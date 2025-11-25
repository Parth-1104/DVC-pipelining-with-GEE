## Corrected FastAPI Code

from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import pandas as pd
import yaml
import os
import requests
from src.model import HydroTransNet
from src.fetch_data import fetch_sentinel2_timeseries
from src.preprocess import preprocess_data
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, 
    PageBreak, Image, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from io import BytesIO
import base64
from fastapi.responses import FileResponse, StreamingResponse
import json
import numpy as np # Add numpy import for std dev handling

# ... (App setup and model loading remain the same) ...

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
config = None

@app.on_event("startup")
async def startup_event():
    # ... (startup logic remains the same) ...
    global model, config
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # NOTE: Assuming HydroTransNet is correctly imported and available
    try:
        model = HydroTransNet(
            input_dim=7,
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_encoder_layers=config['model']['num_encoder_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            output_dim=3
        )
        checkpoint_path = "models/trained_models/best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        # Handle case where model file/config is missing
        print(f"Warning: Model failed to load. Prediction endpoint will fail. Error: {e}")
        model = None


# --- Pydantic Models (The fix is here) ---

class PredictionRequest(BaseModel):
    coordinates: list
    start_date: str
    end_date: str
    input_file: str = 'sentinel2_validated.csv'
    output_file: str = 'processed_features.csv'
    scaler_file: str = 'scaler.pkl'


class GeminiReportRequest(BaseModel):
    lake_name: str
    location: str
    area: float
    chart_data: list
    start_date: str
    end_date: str


class PDFReportRequest(BaseModel):
    lake_name: str
    location: str
    area: float
    chart_data: list
    start_date: str
    end_date: str
    # ✅ FIX 1: Rename to 'ai_report' and expect a dictionary (structured JSON)
    ai_report: dict


# ... (Root and Predict endpoints remain the same) ...

@app.post("/predict")
async def predict(request: PredictionRequest):
    # ... (Predict logic remains the same) ...
    global model, config
    if model is None:
        return {"error": "Model not loaded on startup."}

    df_raw = fetch_sentinel2_timeseries(request.start_date, request.end_date, request.coordinates)
    if df_raw.empty:
        return {"error": "No Sentinel-2 data found for given coordinates and date range."}

    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, request.input_file)
    df_raw.to_csv(raw_path, index=False)

    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    try:
        # NOTE: Assuming preprocess_data is correctly imported and available
        preprocess_data(request.input_file, request.output_file, request.scaler_file)
    except Exception as e:
        return {"error": f"Preprocessing failed: {str(e)}"}

    processed_path = os.path.join(processed_dir, request.output_file)
    if not os.path.exists(processed_path):
        return {"error": "Processed data file not found after preprocessing."}

    df_processed = pd.read_csv(processed_path)
    if df_processed.empty:
        return {"error": "Processed data is empty after preprocessing."}

    seq_len = config['model']['seq_len']
    if len(df_processed) < seq_len:
        return {"error": f"Insufficient processed data length {len(df_processed)} for sequence length {seq_len}"}

    try:
        X = torch.tensor(df_processed.drop(columns=['date'], errors='ignore').values, dtype=torch.float32)
    except Exception as e:
        return {"error": f"Error preparing features for prediction: {str(e)}"}

    preds = []
    # FIX: Ensure prediction window starts correctly
    for i in range(len(X) - seq_len + 1):
        # The model expects [seq_len, batch_size, input_dim]. We use batch_size=1
        seq = X[i:i+seq_len].unsqueeze(1) 
        with torch.no_grad():
            outputs = model(seq)
        preds.append(outputs.cpu().numpy().flatten().tolist())

    cols = ['TSS mg/L', 'Turbidity NTU', 'Chlorophyll ug/L']
    pred_df = pd.DataFrame(preds, columns=cols)
    pred_df['date'] = df_processed['date'].values[seq_len - 1:len(df_processed)]
    pred_df['NDVI'] = df_processed['NDVI'].values[seq_len - 1:len(df_processed)]
    pred_df['NDWI'] = df_processed['NDWI'].values[seq_len - 1:len(df_processed)]
    pred_dicts = pred_df.to_dict(orient='records')
    return {"predictions": pred_dicts}


# ... (Helper functions like parse_gemini_response_to_json, extract_section, etc. remain the same) ...
# NOTE: Keeping the helper functions here for completeness, though they were not the source of the main error.

def parse_gemini_response_to_json(report_text: str) -> dict:
    """
    Parse Gemini's text response into structured JSON format for government reporting.
    Extract key sections: Water Quality Impact, Agricultural Suitability, Recommendations, etc.
    """
    
    # Structure the response into logical government-ready sections
    structured_report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "system": "NadiNetra Water Quality AI"
        },
        "water_quality_impact": {
            "tss_impact": extract_section(report_text, "TSS", "suspended solids"),
            "turbidity_impact": extract_section(report_text, "Turbidity", "water clarity"),
            "chlorophyll_impact": extract_section(report_text, "Chlorophyll", "algae"),
            "vegetation_indices": extract_section(report_text, "NDVI", "NDWI")
        },
        "agricultural_suitability": {
            "suitable_crops": extract_crops(report_text),
            "irrigation_safety": extract_irrigation_assessment(report_text),
            "soil_health_impact": extract_soil_assessment(report_text),
            "risk_factors": extract_risks(report_text)
        },
        "fertilizer_management": {
            "recommended_type": extract_fertilizer_type(report_text),
            "nutrient_leaching_risk": extract_leaching_risk(report_text),
            "application_guidelines": extract_application_guidelines(report_text)
        },
        "environmental_concerns": {
            "contamination_likelihood": extract_contamination_risk(report_text),
            "sedimentation_status": extract_sedimentation(report_text),
            "soil_degradation_risk": extract_degradation_risk(report_text)
        },
        "government_interventions": extract_interventions(report_text),
        "action_items": extract_action_items(report_text),
        "full_analysis": report_text  # Keep original for reference
    }
    
    return structured_report

def extract_section(text: str, *keywords) -> str:
    """Extract a section containing any of the keywords."""
    lines = text.split('\n')
    for line in lines:
        if any(keyword.lower() in line.lower() for keyword in keywords):
            # This is a simplification. For a real report, you'd want the entire paragraph.
            return line.strip() 
    return "No specific data available"

def extract_crops(text: str) -> list:
    crops_keywords = ['rice', 'wheat', 'maize', 'corn', 'vegetables', 'pulses', 'sugarcane', 'cotton', 'soybean']
    mentioned_crops = []
    text_lower = text.lower()
    for crop in crops_keywords:
        if crop in text_lower:
            mentioned_crops.append(crop.capitalize())
    return mentioned_crops if mentioned_crops else ["Requires detailed assessment"]

def extract_irrigation_assessment(text: str) -> str:
    if 'safe' in text.lower():
        return "Suitable for irrigation with standard protocols"
    elif 'treatment' in text.lower():
        return "Requires pre-treatment before irrigation use"
    else:
        return "Assessment required - monitor water quality closely"

def extract_soil_assessment(text: str) -> str:
    if 'fertile' in text.lower() or 'good' in text.lower():
        return "Soil health appears stable and supports productivity"
    elif 'degradation' in text.lower() or 'poor' in text.lower():
        return "Soil degradation risk detected - intervention recommended"
    else:
        return "Soil health status: Monitor with periodic testing"

def extract_risks(text: str) -> list:
    risks = []
    risk_keywords = {
        'contamination': 'Water contamination risk',
        'runoff': 'Nutrient runoff and fertilizer loss',
        'leaching': 'Nutrient leaching into groundwater',
        'sedimentation': 'Excessive sedimentation',
        'algae': 'Algal bloom potential'
    }
    for keyword, risk_label in risk_keywords.items():
        if keyword in text.lower():
            risks.append(risk_label)
    return risks if risks else ["Monitor for emerging risks"]

def extract_fertilizer_type(text: str) -> str:
    if 'organic' in text.lower():
        return "Organic or slow-release fertilizers recommended"
    elif 'nitrogen' in text.lower() or 'phosphate' in text.lower():
        return "Balanced NPK fertilizer with controlled release"
    else:
        return "Consult agronomist for tailored fertilizer plan"

def extract_leaching_risk(text: str) -> str:
    if 'high' in text.lower() and 'leach' in text.lower():
        return "High risk - reduce application rates and increase frequency"
    elif 'low' in text.lower() and 'leach' in text.lower():
        return "Low risk - standard application rates acceptable"
    else:
        return "Moderate risk - implement best management practices"

def extract_application_guidelines(text: str) -> str:
    return "Apply fertilizer in 2-3 splits during growing season. Avoid application during heavy rainfall to minimize runoff."

def extract_contamination_risk(text: str) -> str:
    if 'high' in text.lower() and 'contaminat' in text.lower():
        return "High - Immediate testing and source identification required"
    elif 'low' in text.lower():
        return "Low - Continue routine monitoring"
    else:
        return "Moderate - Implement targeted monitoring program"

def extract_sedimentation(text: str) -> str:
    if 'high' in text.lower() and ('sediment' in text.lower() or 'tss' in text.lower()):
        return "High sedimentation detected - May reduce irrigation efficiency"
    else:
        return "Sedimentation within acceptable range"

def extract_degradation_risk(text: str) -> str:
    if 'risk' in text.lower() and 'degrad' in text.lower():
        return "Risk detected - Implement soil conservation measures"
    else:
        return "Soil degradation risk is minimal"

def extract_interventions(text: str) -> list:
    interventions = []
    intervention_keywords = {
        'monitoring': 'Establish comprehensive water quality monitoring network',
        'treatment': 'Develop water pre-treatment infrastructure',
        'pollution': 'Implement pollution source control measures',
        'coordination': 'Coordinate with agricultural departments for irrigation planning',
        'warning': 'Set up early warning system for contamination events',
        'community': 'Engage farming communities in data-driven decision making'
    }
    for keyword, intervention in intervention_keywords.items():
        if keyword in text.lower():
            interventions.append(intervention)
    
    return interventions if interventions else ["Establish baseline monitoring protocols"]

def extract_action_items(text: str) -> list:
    action_items = [
        "Conduct quarterly soil and water testing to validate satellite indices (NDVI, NDWI)",
        "Coordinate with agricultural departments for irrigation scheduling based on water quality data",
        "Establish early warning system for TSS/Turbidity anomalies indicating contamination"
    ]
    return action_items


@app.post("/gemini_report")
async def gemini_report(request: GeminiReportRequest):
    """Generate structured JSON report from Gemini API analysis."""
    
    # ... (API Key and Prompt construction remain the same) ...
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyAtyDLkN2VXW6G8I00V1KzVeldkmKJ-EnM')
    
    prompt = f"""
    Lake Name: {request.lake_name}
    Location: {request.location}
    Area: {request.area} hectares
    Date Range: {request.start_date} to {request.end_date}
    
    Recent Water Quality Data (Last 60 Days):
    {request.chart_data[-60:] if len(request.chart_data) >= 60 else request.chart_data}
    
    Provide a CLEAR, STRUCTURED analysis for government agricultural planning:
    
    1. WATER QUALITY IMPACT: Analyze TSS, Turbidity, Chlorophyll, NDVI, NDWI. Explain effects on irrigation water safety and soil health.
    
    2. AGRICULTURAL SUITABILITY: Recommend specific crops (rice, wheat, maize, vegetables) based on water quality and soil profiles.
    
    3. SOIL & CROP IMPACT: Assess risks of nutrient leaching, soil fertility loss, and contamination. How will this impact farmer yields?
    
    4. FERTILIZER MANAGEMENT: Recommend fertilizer type and quantity. Account for nutrient loss and leaching potential.
    
    5. CONTAMINATION & SEDIMENTATION: Flag any TSS/Turbidity trends indicating contamination or sedimentation issues.
    
    6. GOVERNMENT INTERVENTIONS: Suggest 3-5 actionable steps for water resource managers and agricultural planners.
    
    Use clear paragraphs, proper spacing, and professional language suitable for government reports. Avoid markdown formatting (no **, __, etc.).
    """
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=50)
        data = response.json()
        candidates = data.get("candidates", [])
        
        if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"] and candidates[0]["content"]["parts"]:
            report_text = candidates[0]["content"]["parts"][0]["text"]
            
            # Parse into structured JSON
            structured_report = parse_gemini_response_to_json(report_text)
            
            # ✅ FIX 2: Return the structured_report as the value of the 'report' key.
            return {
                "status": "success",
                "report": structured_report
            }
        else:
            err = data.get("error", {}).get("message", "No report generated")
            return {"status": "error", "error": err}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ... (current_week_prediction and preprocess_data_inline remain the same) ...
@app.post("/currdate")
async def current_week_prediction(request: dict = {}):
    global model, config
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    coordinates = request.get('coordinates', None)
    if not coordinates:
        return {"error": "Please provide 'coordinates' in the request body."}
    df_raw = fetch_sentinel2_timeseries(start_date, end_date, coordinates)
    if df_raw.empty:
        return {"error": f"No Sentinel-2 data found for coordinates in the last week: {start_date} to {end_date}"}
    try:
        df_processed = preprocess_data_inline(df_raw)
    except Exception as e:
        return {"error": f"Preprocessing failed: {str(e)}"}
    if df_processed.empty:
        return {"error": "Processed data is empty after preprocessing."}
    seq_len = config['model']['seq_len']
    if len(df_processed) < seq_len:
        return {"error": f"Insufficient data length {len(df_processed)} for sequence length {seq_len}"}
    try:
        X = torch.tensor(df_processed.drop(columns=['date'], errors='ignore').values, dtype=torch.float32)
    except Exception as e:
        return {"error": f"Error preparing features for prediction: {str(e)}"}
    with torch.no_grad():
        input_seq = X[-seq_len:].unsqueeze(1)
        output = model(input_seq).numpy().flatten()
    last_date = df_processed['date'].values[-1]
    ndvi_val = float(df_processed['NDVI'].values[-1])
    ndwi_val = float(df_processed['NDWI'].values[-1])
    result = {
        "TSS mg/L": float(output[0]),
        "Turbidity NTU": float(output[1]),
        "Chlorophyll ug/L": float(output[2]),
        "date": str(last_date),
        "NDVI": ndvi_val,
        "NDWI": ndwi_val
    }
    return result

def preprocess_data_inline(df):
    df = df.dropna().drop_duplicates(subset=['date'])
    df['NDVI'] = (df['B8_NIR'] - df['B4_Red']) / (df['B8_NIR'] + df['B4_Red'] + 1e-8)
    df['NDWI'] = (df['B3_Green'] - df['B8_NIR']) / (df['B3_Green'] + df['B8_NIR'] + 1e-8)
    df['Turbidity_Index'] = df['B4_Red'] / (df['B3_Green'] + 1e-8)
    feature_cols = ['B2_Blue', 'B3_Green', 'B4_Red', 'B8_NIR', 'NDVI', 'NDWI', 'Turbidity_Index']
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(df[feature_cols].values)
    processed_df = pd.DataFrame(features_scaled, columns=feature_cols)
    processed_df['date'] = df['date'].values
    processed_df['NDVI'] = df['NDVI'].values
    processed_df['NDWI'] = df['NDWI'].values
    return processed_df


@app.post("/generate_pdf_report")
async def generate_pdf_report(request: PDFReportRequest):
    """Generate professional PDF report with structured layout and proper spacing."""
    
    try:
        pdf_buffer = BytesIO()
        pdf_filename = f"{request.lake_name.replace(' ', '_')}_WQ_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        doc = SimpleDocTemplate(
            pdf_buffer, 
            pagesize=A4, 
            topMargin=0.75*inch, 
            bottomMargin=0.75*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch
        )
        story = []
        
        # ... (Define styles remain the same) ...
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=26,
            textColor=colors.HexColor('#0c4a6e'),
            spaceAfter=0.15*inch,
            spaceBefore=0.1*inch,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#475569'),
            spaceAfter=0.3*inch,
            alignment=TA_CENTER,
            fontName='Helvetica-Oblique'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#0c4a6e'),
            spaceAfter=0.2*inch,
            spaceBefore=0.25*inch,
            fontName='Helvetica-Bold',
            borderPadding=0.1*inch
        )
        
        subheading_style = ParagraphStyle(
            'Subheading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=0.15*inch,
            spaceBefore=0.15*inch,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10.5,
            alignment=TA_JUSTIFY,
            spaceAfter=0.12*inch,
            leading=14
        )
        
        body_left_style = ParagraphStyle(
            'BodyLeft',
            parent=styles['BodyText'],
            fontSize=10.5,
            alignment=TA_LEFT,
            spaceAfter=0.12*inch,
            leading=14
        )
        
        # ===== PAGE 1: TITLE & OVERVIEW =====
        story.append(Paragraph("WATER QUALITY MONITORING REPORT", title_style))
        story.append(Paragraph("AI-Powered Analysis for Agricultural Water Management", subtitle_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Lake Overview
        story.append(Paragraph("LAKE OVERVIEW", heading_style))
        overview_data = [
            ['Parameter', 'Details'],
            ['Lake Name', request.lake_name],
            ['Location', request.location],
            ['Area', f"{request.area} hectares"],
            ['Reporting Period', f"{request.start_date} to {request.end_date}"],
            ['Report Generated', datetime.now().strftime('%d %B %Y at %H:%M:%S')],
            ['System', 'NadiNetra Water Quality AI']
        ]
        # ... (Overview table style and append remain the same) ...
        overview_table = Table(overview_data, colWidths=[1.8*inch, 4.2*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0c4a6e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f9ff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1e293b')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f9ff'), colors.white])
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 0.25*inch))
        
        # Water Quality Measurements Table
        story.append(Paragraph("WATER QUALITY MEASUREMENTS", heading_style))
        
        table_data = [['Date', 'Turbidity (NTU)', 'TSS (mg/L)', 'Chlorophyll (µg/L)', 'NDVI', 'NDWI']]
        for row in request.chart_data[-14:]:
            # Ensure float conversion and error handling for missing keys
            try:
                table_data.append([
                    str(row.get('date', 'N/A')),
                    f"{float(row.get('turbidity', 0)):.5f}",
                    f"{float(row.get('tss', 0)):.5f}",
                    f"{float(row.get('chlorophyll', 0)):.5f}",
                    f"{float(row.get('ndvi', 0)):.5f}",
                    f"{float(row.get('ndwi', 0)):.5f}"
                ])
            except ValueError:
                # Handle non-numeric values gracefully
                table_data.append([
                    str(row.get('date', 'N/A')), 'Error', 'Error', 'Error', 'Error', 'Error'
                ])

        
        # ... (Data table style and append remain the same) ...
        data_table = Table(table_data, colWidths=[1.0*inch, 1.0*inch, 1.0*inch, 1.2*inch, 0.9*inch, 0.9*inch])
        data_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0c4a6e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
        ]))
        story.append(data_table)
        story.append(Spacer(1, 0.25*inch))
        
        # Statistical Summary
        story.append(Paragraph("STATISTICAL SUMMARY", heading_style))
        
        if request.chart_data:
            turbidity_vals = [float(x.get('turbidity', 0)) for x in request.chart_data if isinstance(x.get('turbidity'), (int, float))]
            tss_vals = [float(x.get('tss', 0)) for x in request.chart_data if isinstance(x.get('tss'), (int, float))]
            chlorophyll_vals = [float(x.get('chlorophyll', 0)) for x in request.chart_data if isinstance(x.get('chlorophyll'), (int, float))]
            
            # Helper for stats
            def get_stats(data_list):
                if not data_list:
                    return ["0.00000"] * 4
                s = pd.Series(data_list)
                return [
                    f"{s.min():.5f}",
                    f"{s.max():.5f}",
                    f"{s.mean():.5f}",
                    f"{s.std():.5f}" if len(s) > 1 else "0.00000"
                ]

            turb_stats = get_stats(turbidity_vals)
            tss_stats = get_stats(tss_vals)
            chloro_stats = get_stats(chlorophyll_vals)
            
            stats_data = [
                ['Parameter', 'Minimum', 'Maximum', 'Average', 'Std Dev'],
                ['Turbidity (NTU)'] + turb_stats,
                ['TSS (mg/L)'] + tss_stats,
                ['Chlorophyll (µg/L)'] + chloro_stats,
            ]
            
            # ... (Stats table style and append remain the same) ...
            stats_table = Table(stats_data, colWidths=[1.6*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#06b6d4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9.5),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0fdfa')])
            ]))
            story.append(stats_table)
        
        story.append(Spacer(1, 0.25*inch))
        story.append(PageBreak())
        
        # ===== PAGE 2: AI ANALYSIS & GOVERNMENT REPORT =====
        story.append(Paragraph("AI-POWERED ANALYSIS & GOVERNMENT REPORT", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        # ✅ FIX 3: Use the corrected Pydantic field name `request.ai_report`
        ai_report = request.ai_report if isinstance(request.ai_report, dict) else {}
        
        # Water Quality Impact Section
        if 'water_quality_impact' in ai_report:
            story.append(Paragraph("WATER QUALITY IMPACT ASSESSMENT", subheading_style))
            wq = ai_report['water_quality_impact']
            # ... (Rest of WQ section remains the same) ...
            story.append(Paragraph(f"TSS Impact: {wq.get('tss_impact', 'No data')}", body_left_style))
            story.append(Paragraph(f"Turbidity Impact: {wq.get('turbidity_impact', 'No data')}", body_left_style))
            story.append(Paragraph(f"Chlorophyll Impact: {wq.get('chlorophyll_impact', 'No data')}", body_left_style))
            story.append(Paragraph(f"Vegetation Indices: {wq.get('vegetation_indices', 'No data')}", body_left_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Agricultural Suitability Section
        if 'agricultural_suitability' in ai_report:
            story.append(Paragraph("AGRICULTURAL SUITABILITY", subheading_style))
            ag = ai_report['agricultural_suitability']
            
            if ag.get('suitable_crops'):
                crops_text = ", ".join(ag['suitable_crops']) if isinstance(ag['suitable_crops'], list) else str(ag['suitable_crops'])
                story.append(Paragraph(f"Suitable Crops: {crops_text}", body_left_style))
            
            story.append(Paragraph(f"Irrigation Safety: {ag.get('irrigation_safety', 'Assessment required')}", body_left_style))
            story.append(Paragraph(f"Soil Health Impact: {ag.get('soil_health_impact', 'Monitor closely')}", body_left_style))
            
            if ag.get('risk_factors'):
                risks_text = "; ".join(ag['risk_factors']) if isinstance(ag['risk_factors'], list) else str(ag['risk_factors'])
                story.append(Paragraph(f"Identified Risks: {risks_text}", body_left_style))
            
            story.append(Spacer(1, 0.15*inch))
        
        # Fertilizer Management Section
        if 'fertilizer_management' in ai_report:
            story.append(Paragraph("FERTILIZER MANAGEMENT RECOMMENDATIONS", subheading_style))
            fert = ai_report['fertilizer_management']
            story.append(Paragraph(f"Recommended Type: {fert.get('recommended_type', 'Consult agronomist')}", body_left_style))
            story.append(Paragraph(f"Nutrient Leaching Risk: {fert.get('nutrient_leaching_risk', 'Monitor')}", body_left_style))
            story.append(Paragraph(f"Application Guidelines: {fert.get('application_guidelines', 'Standard practices')}", body_left_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Environmental Concerns Section
        if 'environmental_concerns' in ai_report:
            story.append(Paragraph("ENVIRONMENTAL CONCERNS", subheading_style))
            env = ai_report['environmental_concerns']
            story.append(Paragraph(f"Contamination Likelihood: {env.get('contamination_likelihood', 'Low')}", body_left_style))
            story.append(Paragraph(f"Sedimentation Status: {env.get('sedimentation_status', 'Normal')}", body_left_style))
            story.append(Paragraph(f"Soil Degradation Risk: {env.get('soil_degradation_risk', 'Minimal')}", body_left_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Government Interventions Section
        story.append(Paragraph("RECOMMENDED GOVERNMENT INTERVENTIONS", subheading_style))
        
        interventions = ai_report.get('government_interventions', [])
        if interventions and isinstance(interventions, list):
            for i, intervention in enumerate(interventions, 1):
                story.append(Paragraph(f"{i}. {intervention}", body_left_style))
        else:
            default_interventions = [
                "Establish comprehensive water quality monitoring network with quarterly assessments",
                "Coordinate with agricultural departments for irrigation scheduling and water treatment protocols",
                "Develop early warning system for TSS and Turbidity spikes indicating contamination events"
            ]
            for i, intervention in enumerate(default_interventions, 1):
                story.append(Paragraph(f"{i}. {intervention}", body_left_style))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Action Items Section
        story.append(Paragraph("TOP ACTION ITEMS FOR LOCAL AUTHORITIES", subheading_style))
        
        action_items = ai_report.get('action_items', [])
        if action_items and isinstance(action_items, list):
            for i, action in enumerate(action_items, 1):
                story.append(Paragraph(f"{i}. {action}", body_left_style))
        else:
            default_actions = [
                "Conduct quarterly soil and water testing to validate satellite-derived indices (NDVI, NDWI)",
                "Establish data-sharing protocols with agricultural departments for irrigation management",
                "Implement farmer education programs on water quality impacts and sustainable practices"
            ]
            for i, action in enumerate(default_actions, 1):
                story.append(Paragraph(f"{i}. {action}", body_left_style))
        
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        story.append(PageBreak())
        
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['BodyText'],
            fontSize=9,
            textColor=colors.HexColor('#64748b'),
            alignment=TA_CENTER,
            spaceAfter=0.1*inch
        )
        
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("REPORT FOOTER", heading_style))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "This report was generated by the NadiNetra Water Quality AI System using satellite-derived data and machine learning analysis.",
            footer_style
        ))
        story.append(Paragraph(
            f"Report accuracy as of {datetime.now().strftime('%d %B %Y at %H:%M:%S')}",
            footer_style
        ))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "For questions, data validation, or technical support, contact your Water Resource Management Agency.",
            footer_style
        ))
        
        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        
        return StreamingResponse(
            iter([pdf_buffer.getvalue()]),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={pdf_filename}"}
        )
        
    except Exception as e:
        # Proper error logging for debugging
        print(f"Error during PDF generation: {e}")
        return {"error": f"PDF generation failed: {str(e)}"}