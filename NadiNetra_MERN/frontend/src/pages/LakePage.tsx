import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { waterBodies } from '../data/waterBodies';
import TimeSpanSelector from '../components/TimeSpanSelector';
import { subDays, subMonths, subYears, format } from 'date-fns';
import { Download, AlertCircle, TrendingUp } from 'lucide-react';

const DECIMALS = 7;

const metricConfig = [
  { key: "turbidity", label: "Turbidity NTU", unit: "NTU", color: "text-yellow-600", bgColor: "bg-yellow-50" },
  { key: "tss", label: "TSS mg/L", unit: "mg/L", color: "text-orange-600", bgColor: "bg-orange-50" },
  { key: "chlorophyll", label: "Chlorophyll µg/L", unit: "µg/L", color: "text-green-600", bgColor: "bg-green-50" },
  { key: "ndvi", label: "NDVI", unit: "--", color: "text-emerald-600", bgColor: "bg-emerald-50" },
  { key: "ndwi", label: "NDWI", unit: "--", color: "text-cyan-600", bgColor: "bg-cyan-50" },
];

function formatNumber(value) {
  return value !== undefined ? value.toFixed(DECIMALS) : '--';
}

function getTableRowsToShow(timeSpan) {
  if (timeSpan === '1Y') return 80;
  if (timeSpan === '6M') return 40;
  return 14;
}

// Animated PDF Loader Component
const PDFLoader = () => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-12 shadow-2xl max-w-sm mx-4">
        {/* Animated Circles */}
        <div className="flex justify-center mb-8">
          <div className="relative w-24 h-24">
            <div className="absolute inset-0 rounded-full border-4 border-blue-200 animate-ping" style={{ animationDuration: '1.5s' }}></div>
            <div className="absolute inset-2 rounded-full border-4 border-cyan-400 animate-spin" style={{ animationDuration: '2s' }}></div>
            <div className="absolute inset-4 rounded-full border-4 border-blue-600 animate-pulse"></div>
            
            {/* Center Icon */}
            <div className="absolute inset-0 flex items-center justify-center">
              <Download className="w-10 h-10 text-blue-600 animate-bounce" style={{ animationDuration: '1s' }} />
            </div>
          </div>
        </div>

        {/* Text */}
        <div className="text-center">
          <h3 className="text-xl font-bold text-gray-900 mb-2">Generating PDF Report</h3>
          <p className="text-gray-600 text-sm">Compiling water quality analysis...</p>
          
          {/* Progress Bar */}
          <div className="mt-6 w-full bg-gray-200 rounded-full h-2 overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-blue-500 via-cyan-400 to-blue-600 rounded-full"
              style={{
                animation: 'progress 2s ease-in-out infinite',
              }}
            ></div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes progress {
          0%, 100% { width: 30%; }
          50% { width: 90%; }
        }
      `}</style>
    </div>
  );
};

export default function LakePage() {
  const { name } = useParams();
  const [timeSpan, setTimeSpan] = useState('1M');
  const [chartData, setChartData] = useState([]);
  const [error, setError] = useState(null);
  const [report, setReport] = useState('');
  const [isFetchingData, setIsFetchingData] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [isPdfGenerating, setIsPdfGenerating] = useState(false);
  const [hasAnimated, setHasAnimated] = useState(false);

  const lake = waterBodies.find(body => body.name.toLowerCase() === decodeURIComponent(name || ''));

  useEffect(() => {
    let cancelled = false;
    if (!lake) return;
    setIsFetchingData(true);
    setError(null);

    const today = new Date();
    let startDate = new Date();
    switch (timeSpan) {
      case '1W': startDate = subDays(today, 20); break;
      case '1M': startDate = subMonths(today, 1); break;
      case '6M': startDate = subMonths(today, 6); break;
      case '1Y': startDate = subYears(today, 1); break;
      default: startDate = subMonths(today, 1);
    }
    const formattedStartDate = format(startDate, 'yyyy-MM-dd');
    const formattedEndDate = format(today, 'yyyy-MM-dd');

    fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        coordinates: lake.coordinates,
        start_date: formattedStartDate,
        end_date: formattedEndDate
      }),
    })
      .then(res => res.json())
      .then(data => {
        if (cancelled) return;
        const normalizedData = (data.predictions || []).map(item => ({
          date: item.date,
          turbidity: item["Turbidity NTU"],
          tss: item["TSS mg/L"],
          chlorophyll: item["Chlorophyll ug/L"],
          ndvi: item["NDVI"],
          ndwi: item["NDWI"]
        }));
        setChartData(normalizedData);
        setTimeout(() => setHasAnimated(true), 100);
      })
      .catch(() => setError('Unable to load water quality data'))
      .finally(() => setIsFetchingData(false));

    return () => { cancelled = true };
  }, [lake, timeSpan]);

  const handleGenerateReport = async () => {
    if (!lake || chartData.length === 0) return;
    setIsGeneratingReport(true);
    setReport('');
    const today = new Date();
    let startDate = new Date();
    switch (timeSpan) {
      case '1W': startDate = subDays(today, 20); break;
      case '1M': startDate = subMonths(today, 1); break;
      case '6M': startDate = subMonths(today, 6); break;
      case '1Y': startDate = subYears(today, 1); break;
      default: startDate = subMonths(today, 1);
    }
    const formattedStartDate = format(startDate, 'yyyy-MM-dd');
    const formattedEndDate = format(today, 'yyyy-MM-dd');
    const req = {
      lake_name: lake.name,
      location: lake.location,
      area: lake.area,
      chart_data: chartData,
      start_date: formattedStartDate,
      end_date: formattedEndDate
    };
    try {
      const resp = await fetch('http://127.0.0.1:8000/gemini_report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req)
      });
      const json = await resp.json();
      setReport(json.report || json.error || 'No report generated.');
    } catch {
      setReport('Error calling Gemini API.');
    } finally {
      setIsGeneratingReport(false);
    }
  };

  async function handleDownloadPDF() {
    if (!lake || chartData.length === 0) {
      alert('Please wait for data to load before generating PDF.');
      return;
    }
    setIsPdfGenerating(true);

    const today = new Date();
    let startDate = new Date();
    switch (timeSpan) {
      case '1W': startDate = subDays(today, 20); break;
      case '1M': startDate = subMonths(today, 1); break;
      case '6M': startDate = subMonths(today, 6); break;
      case '1Y': startDate = subYears(today, 1); break;
      default: startDate = subMonths(today, 1);
    }
    const formattedStartDate = format(startDate, 'yyyy-MM-dd');
    const formattedEndDate = format(today, 'yyyy-MM-dd');

    const req = {
      lake_name: lake.name,
      location: lake.location,
      area: lake.area,
      chart_data: chartData,
      start_date: formattedStartDate,
      end_date: formattedEndDate,
      ai_report: report || '' // Send empty string if no report
    };
    try {
      const resp = await fetch('http://127.0.0.1:8000/generate_pdf_report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req)
      });
      if (!resp.ok) throw new Error('PDF generation failed');
      const blob = await resp.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${lake.name}_WQ_Report_${format(today, 'yyyyMMdd')}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert('Error generating PDF: ' + err.message);
    } finally {
      setIsPdfGenerating(false);
    }
  }

  if (!lake) {
    return (
      <div className="text-center py-14">
        <h2 className="text-2xl font-bold text-gray-900">Water body not found</h2>
        <p className="mt-2 text-gray-500">Select a valid lake to view details.</p>
      </div>
    );
  }

  const rowsToShow = getTableRowsToShow(timeSpan);
  const hasWaterQualityIssues = chartData.some(row => row.turbidity > 15);

  return (
    <div className="space-y-10">
      {isPdfGenerating && <PDFLoader />}

      {/* Header */}
      <div>
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 tracking-tight">{lake.name}</h1>
            <p className="mt-2 text-gray-600 text-base flex items-center gap-2">
              📍 {lake.location}
            </p>
          </div>
          {hasWaterQualityIssues && (
            <div className="flex items-center gap-2 px-4 py-2 bg-red-50 border border-red-200 rounded-lg">
              <AlertCircle className="w-5 h-5 text-red-600" />
              <span className="text-sm font-medium text-red-700">Quality Alert</span>
            </div>
          )}
        </div>
        <TimeSpanSelector selectedSpan={timeSpan} onSpanChange={setTimeSpan} className="mt-4" />
      </div>

      {/* Enhanced Table Card */}
      <div style={{ position: 'relative', minHeight: 300 }}>
        {isFetchingData && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-white bg-opacity-70 rounded-xl z-10 backdrop-blur-sm">
            <svg className="animate-spin h-14 w-14 text-blue-500 mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-70" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <span className="text-blue-700 font-medium text-lg tracking-wide">Fetching water quality data...</span>
          </div>
        )}
        
        <div className={`transition-all duration-700 ease-in-out bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden ${
          hasAnimated && chartData.length > 0 && !isFetchingData ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
        }`}>
          {/* Table Header */}
          <div className="bg-gradient-to-r from-blue-600 to-cyan-600 px-8 py-6">
            <h3 className="text-xl font-bold text-white flex items-center gap-2">
              <TrendingUp className="w-6 h-6" />
              Water Quality Metrics
            </h3>
            <p className="text-blue-100 text-sm mt-1">Precision: 7 decimal places | Most recent at bottom</p>
          </div>

          {/* Table Content */}
          <div className="overflow-x-auto">
            {error ? (
              <div className="p-8 text-center text-red-600 bg-red-50 m-4 rounded-lg flex items-center justify-center gap-2">
                <AlertCircle className="w-5 h-5" />
                {error}
              </div>
            ) : chartData.length === 0 ? (
              <div className="py-12 text-gray-500 text-center">No data available for this period.</div>
            ) : (
              <table className="w-full border-collapse" style={{ fontVariantNumeric: 'tabular-nums' }}>
                <thead>
                  <tr className="bg-gray-50 border-b border-gray-200">
                    <th className="px-6 py-4 text-left font-bold text-gray-900">Date</th>
                    {metricConfig.map(m => (
                      <th key={m.key} className={`px-6 py-4 text-center font-bold text-sm ${m.color}`}>
                        <div className="font-semibold">{m.label}</div>
                        <div className="text-xs font-normal text-gray-500">{m.unit}</div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {chartData.slice(-rowsToShow).map((row, idx) => {
                    const isCritical = row.turbidity > 25;
                    const isWarning = row.turbidity > 15;
                    
                    return (
                      <tr 
                        key={row.date + idx}
                        className={`border-b border-gray-100 transition-colors duration-150 ${
                          idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'
                        } hover:bg-blue-50`}
                      >
                        <td className="px-6 py-4 font-semibold text-gray-900">{row.date}</td>
                        {metricConfig.map(m => {
                          const value = row[m.key];
                          const isCriticalMetric = m.key === 'turbidity' && isCritical;
                          const isWarningMetric = m.key === 'turbidity' && isWarning && !isCritical;
                          
                          return (
                            <td 
                              key={m.key} 
                              className={`px-6 py-4 text-center font-mono text-sm font-medium ${
                                isCriticalMetric 
                                  ? 'bg-red-100 text-red-900 rounded' 
                                  : isWarningMetric 
                                  ? 'bg-yellow-100 text-yellow-900 rounded'
                                  : m.color
                              }`}
                            >
                              {formatNumber(value)}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>

          {/* Table Footer */}
          <div className="bg-gray-50 px-8 py-4 border-t border-gray-200 text-xs text-gray-600">
            Showing {Math.min(rowsToShow, chartData.length)} of {chartData.length} records
          </div>
        </div>
      </div>

      {/* AI Gemini Report Section */}
      <div className="rounded-xl shadow-lg p-8 bg-gradient-to-br from-blue-50 via-cyan-50 to-blue-50 border border-blue-200 transition-all duration-700">
        <h3 className="text-xl font-bold mb-4 text-blue-900 flex items-center gap-2">
          <span className={`${isGeneratingReport ? 'animate-spin' : ''}`} style={{ display: 'inline-block' }}>✨</span>
          AI-Powered Gemini Report
        </h3>
        
        <div className="flex gap-4 items-center mb-4 flex-wrap">
          <button
            onClick={handleGenerateReport}
            disabled={isGeneratingReport || !chartData.length}
            className="bg-gradient-to-r from-blue-600 to-cyan-600 text-white px-6 py-3 rounded-lg shadow-md hover:shadow-lg hover:scale-105 transition-all duration-200 font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isGeneratingReport ? "⏳ Generating..." : "✨ Generate AI Report"}
          </button>
          
          <button
            onClick={handleDownloadPDF}
            disabled={isPdfGenerating || !chartData.length}
            className="bg-gradient-to-r from-green-600 to-emerald-600 text-white px-6 py-3 rounded-lg shadow-md hover:shadow-lg hover:scale-105 transition-all duration-200 font-semibold flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Download className="w-4 h-4" />
            {isPdfGenerating ? "Generating PDF..." : "Download PDF Report"}
          </button>
        </div>

        <div 
          className="bg-white rounded-lg p-6 border border-blue-100 min-h-[150px] mt-4"
          style={{
            opacity: report ? 1 : 0.7,
            transition: 'opacity 0.6s',
          }}
        >
          {report ? (
            <div className="whitespace-pre-line text-gray-800 text-sm leading-relaxed font-mono">
              {report}
            </div>
          ) : (
            <div className="text-gray-500 italic text-center py-8">
              Click "Generate AI Report" to get a natural language assessment of this lake's water quality trends.
            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes slide-fade-down {
          from { opacity: 0; transform: translateY(-20px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
