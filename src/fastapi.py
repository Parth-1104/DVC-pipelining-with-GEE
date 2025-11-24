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
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from io import BytesIO
import base64
from fastapi.responses import FileResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use allowed origins in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
config = None

@app.on_event("startup")
async def startup_event():
    global model, config
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
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

class PredictionRequest(BaseModel):
    coordinates: list  # polygon coords [[lng, lat], ...]
    start_date: str    # e.g. "2024-01-01"
    end_date: str      # e.g. "2024-01-10"
    input_file: str = 'sentinel2_validated.csv'
    output_file: str = 'processed_features.csv'
    scaler_file: str = 'scaler.pkl'

class PDFReportRequest(BaseModel):
    lake_name: str
    location: str
    area: float
    chart_data: list
    start_date: str
    end_date: str
    ai_report: str

@app.get("/")
async def root():
    return {"message": "Water Quality Prediction API is running."}

@app.post("/predict")
async def predict(request: PredictionRequest):
    global model, config

    # 1. Fetch raw time series from Sentinel-2 in requested time window
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

    # 2. Prepare feature tensor [all rows, all columns except date]
    try:
        X = torch.tensor(df_processed.drop(columns=['date'], errors='ignore').values, dtype=torch.float32)
    except Exception as e:
        return {"error": f"Error preparing features for prediction: {str(e)}"}

    # 3. Generate predictions for each sequence window
    preds = []
    for i in range(len(X) - seq_len + 1):
        seq = X[i:i+seq_len].unsqueeze(1)
        with torch.no_grad():
            outputs = model(seq)
        preds.append(outputs.numpy().flatten().tolist())

    cols = ['TSS mg/L', 'Turbidity NTU', 'Chlorophyll ug/L']
    pred_df = pd.DataFrame(preds, columns=cols)
    pred_df['date'] = df_processed['date'].values[seq_len - 1:]
    pred_df['NDVI'] = df_processed['NDVI'].values[seq_len - 1:]
    pred_df['NDWI'] = df_processed['NDWI'].values[seq_len - 1:]
    pred_dicts = pred_df.to_dict(orient='records')
    return {"predictions": pred_dicts}

class GeminiReportRequest(BaseModel):
    lake_name: str
    location: str
    area: float
    chart_data: list  # [{date:..., tss:..., turbidity:..., ...}, ...]
    start_date: str
    end_date: str

@app.post("/gemini_report")
async def gemini_report(request: GeminiReportRequest):
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyDy728qlqRd_QYTvF7eJhEG0vcheQd7WEw')
    prompt = f"""
    Lake Name: {request.lake_name}
    Location: {request.location}
    Area: {request.area} ha
    Date Range: {request.start_date} to {request.end_date}
    Recent Data Snapshot (up to 100 recent days):
    {request.chart_data[-60:] if len(request.chart_data) >= 60 else request.chart_data}
    Analyze the above water quality data with respect to agricultural and irrigation suitability. Specifically:
    - Interpret the impact of TSS, Turbidity, Chlorophyll, NDVI, and NDWI on irrigation water safety, soil health, and agricultural productivity.
    - Flag any values or trends that could negatively affect crop yields, soil fertility, or risk fertilizer loss/runoff.
    - Based on these parameters, suggest which crops are best suited for this region given the water and soil profiles (e.g., rice, wheat, maize, vegetables).
    - Recommend optimal fertilizer usage (type and quantity) considering water quality and soil support.
    - Assess the likelihood of nutrient leaching, soil degradation, excessive sedimentation, or contamination.
    - Suggest interventions or precautions: options for water pre-treatment, soil amendments, crop rotation, and integrated water management.
    - Conclude with 2–3 actionable steps for local government/agencies to support sustainable agriculture and irrigation based on current data.

    Provide the reasoning for each recommendation as a concise, policy-ready summary for agricultural planners and water resource managers.
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
            return {"report": report_text}
        else:
            err = data.get("error", {}).get("message", "No report generated (API issue or quota exceeded).")
            return {"error": err}
    except Exception as e:
        return {"error": str(e)}


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
    """
    Generate a professional government-grade PDF report with:
    - Lake overview and metadata
    - Water quality data table (7 decimal precision)
    - Statistical summary (min/max/avg)
    - AI-powered narrative analysis
    - Recommendations for government action
    - Export-ready format for agencies
    """
    
    # Create BytesIO buffer for PDF
    pdf_buffer = BytesIO()
    pdf_filename = f"{request.lake_name.replace(' ', '_')}_WQ_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=0.3*inch,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=0.2*inch,
        spaceBefore=0.2*inch,
        fontName='Helvetica-Bold'
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=0.15*inch
    )
    
    # Title
    story.append(Paragraph("WATER QUALITY MONITORING REPORT", title_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Lake Overview Section
    story.append(Paragraph("LAKE OVERVIEW", heading_style))
    overview_data = [
        ['Lake Name', request.lake_name],
        ['Location', request.location],
        ['Area', f"{request.area} hectares"],
        ['Reporting Period', f"{request.start_date} to {request.end_date}"],
        ['Report Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    ]
    overview_table = Table(overview_data, colWidths=[2*inch, 4*inch])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e0f2fe')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Water Quality Data Table
    story.append(Paragraph("WATER QUALITY MEASUREMENTS (7 Decimal Precision)", heading_style))
    
    table_data = [['Date', 'Turbidity\n(NTU)', 'TSS\n(mg/L)', 'Chlorophyll\n(µg/L)', 'NDVI', 'NDWI']]
    for row in request.chart_data[-14:]:  # Last 14 records
        table_data.append([
            row['date'],
            f"{row.get('turbidity', 0):.7f}",
            f"{row.get('tss', 0):.7f}",
            f"{row.get('chlorophyll', 0):.7f}",
            f"{row.get('ndvi', 0):.7f}",
            f"{row.get('ndwi', 0):.7f}"
        ])
    
    data_table = Table(table_data, colWidths=[1.2*inch, 1*inch, 1*inch, 1.2*inch, 0.8*inch, 0.8*inch])
    data_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f9ff')])
    ]))
    story.append(data_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Statistical Summary
    story.append(Paragraph("STATISTICAL SUMMARY", heading_style))
    
    if request.chart_data:
        turbidity_vals = [x.get('turbidity', 0) for x in request.chart_data]
        tss_vals = [x.get('tss', 0) for x in request.chart_data]
        chlorophyll_vals = [x.get('chlorophyll', 0) for x in request.chart_data]
        
        stats_data = [
            ['Parameter', 'Min', 'Max', 'Average'],
            ['Turbidity (NTU)', f"{min(turbidity_vals):.7f}", f"{max(turbidity_vals):.7f}", f"{sum(turbidity_vals)/len(turbidity_vals):.7f}"],
            ['TSS (mg/L)', f"{min(tss_vals):.7f}", f"{max(tss_vals):.7f}", f"{sum(tss_vals)/len(tss_vals):.7f}"],
            ['Chlorophyll (µg/L)', f"{min(chlorophyll_vals):.7f}", f"{max(chlorophyll_vals):.7f}", f"{sum(chlorophyll_vals)/len(chlorophyll_vals):.7f}"]
        ]
        stats_table = Table(stats_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#06b6d4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')])
        ]))
        story.append(stats_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Page Break
    story.append(PageBreak())
    
    # AI Analysis & Recommendations
    story.append(Paragraph("AI-POWERED ANALYSIS & RECOMMENDATIONS", heading_style))
    story.append(Paragraph(request.ai_report, body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Government Action Items
    story.append(Paragraph("RECOMMENDED GOVERNMENT INTERVENTIONS", heading_style))
    interventions = [
        "Establish monitoring frequency based on detected anomalies and risk levels.",
        "Coordinate with agricultural departments for irrigation scheduling and water treatment protocols.",
        "Implement pollution source tracking and mitigation near upstream industrial/urban areas.",
        "Conduct quarterly soil and water testing to validate satellite-derived indices (NDVI, NDWI).",
        "Set up early warning systems for TSS/Turbidity spikes indicating contamination events.",
        "Engage local farming communities in data-driven decision-making for crop selection and fertilizer use."
    ]
    for i, intervention in enumerate(interventions, 1):
        story.append(Paragraph(f"<b>{i}.</b> {intervention}", body_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['BodyText'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    story.append(Paragraph("---", footer_style))
    story.append(Paragraph(
        f"This report was generated automatically by the Water Quality AI System. "
        f"Data accurate as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.",
        footer_style
    ))
    story.append(Paragraph("For questions or data validation, contact your water resource management agency.", footer_style))
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    
    # Save to file and return
    pdf_path = f"reports/{pdf_filename}"
    os.makedirs("reports", exist_ok=True)
    with open(pdf_path, 'wb') as f:
        f.write(pdf_buffer.getvalue())
    
    return FileResponse(pdf_path, filename=pdf_filename, media_type='application/pdf')