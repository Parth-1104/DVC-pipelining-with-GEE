import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { waterBodies } from '../data/waterBodies';
import TimeSpanSelector from '../components/TimeSpanSelector';
import { subDays, subMonths, subYears, format } from 'date-fns';

const DECIMALS = 7;

const metricConfig = [
  { key: "turbidity", label: "Turbidity NTU" },
  { key: "tss", label: "TSS mg/L" },
  { key: "chlorophyll", label: "Chlorophyll µg/L" },
  { key: "ndvi", label: "NDVI" },
  { key: "ndwi", label: "NDWI" },
];

function formatNumber(value) {
  return value !== undefined ? value.toFixed(DECIMALS) : '--';
}

function getTableRowsToShow(timeSpan) {
  if (timeSpan === '1Y') return 80;
  if (timeSpan === '6M') return 40;
  return 14;
}

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
        setTimeout(() => setHasAnimated(true), 100); // animate table in
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
    if (!lake || chartData.length === 0 || !report) {
      alert('Please generate the AI report first before downloading PDF.');
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
      ai_report: report
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

  // How many rows to show in table based on timespan
  const rowsToShow = getTableRowsToShow(timeSpan);

  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 tracking-tight">{lake.name}</h1>
        <p className="mt-2 text-gray-600 text-md">Location: {lake.location}</p>
        <TimeSpanSelector selectedSpan={timeSpan} onSpanChange={setTimeSpan} className="mt-3" />
      </div>

      {/* Table Card with Loader Overlay */}
      <div style={{ position: 'relative', minHeight: 200, minWidth: 330 }}>
        {/* Loader Overlay */}
        {isFetchingData && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-white bg-opacity-60 rounded-xl z-10">
            <svg className="animate-spin h-14 w-14 text-blue-500 mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-70" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <span className="text-blue-700 font-medium text-lg tracking-wide">Fetching water quality data...</span>
          </div>
        )}
        <div className={`transition-opacity duration-700 ease-in-out bg-white p-7 rounded-xl shadow-md overflow-x-auto ${
          hasAnimated && chartData.length > 0 && !isFetchingData ? 'opacity-100' : 'opacity-0'
        }`}>
          <h3 className="text-lg font-semibold mb-2">Water Quality Table (most recent on bottom, precise to 7 decimals)</h3>
          {error ? (
            <div className="text-center py-6 text-red-500">{error}</div>
          ) : chartData.length === 0 ? (
            <div className="py-8 text-gray-500 text-center">No data for this period.</div>
          ) : (
            <table className="min-w-full border-collapse border" style={{ fontVariantNumeric: 'tabular-nums' }}>
              <thead>
                <tr>
                  <th className="font-bold">Date</th>
                  {metricConfig.map(m => (
                    <th className="font-bold" key={m.key}>{m.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {chartData.slice(-rowsToShow).map((row, idx) => (
                  <tr key={row.date + idx}>
                    <td>{row.date}</td>
                    {metricConfig.map(m => (
                      <td key={m.key}>{formatNumber(row[m.key])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* AI Gemini Report */}
      <div
        className={`rounded-xl shadow-md p-8 mt-6 bg-gradient-to-br from-blue-50 to-cyan-50 border border-blue-100 transition-all duration-700 ${report ? 'animate-[fadein_2s]' : ''}`}
        style={{ minHeight: 250 }}
      >
        <h3 className="text-lg font-bold mb-4 text-blue-800 flex items-center gap-2">
          <span className="animate-spin" style={{ display: isGeneratingReport ? 'inline-block' : 'none' }}>🔄</span>
          AI-Powered Gemini Report
        </h3>
        <div className="flex gap-4 items-center mb-3 flex-wrap">
          <button
            onClick={handleGenerateReport}
            disabled={isGeneratingReport || !chartData.length}
            className="bg-gradient-to-br from-blue-600 to-cyan-500 text-white px-5 py-2 rounded shadow hover:scale-105 transition-transform duration-200 font-medium text-base"
          >
            {isGeneratingReport ? "Generating..." : "Generate AI Report"}
          </button>
          <button
            onClick={handleDownloadPDF}
            disabled={isPdfGenerating || !report}
            className="bg-green-600 text-white px-5 py-2 rounded shadow hover:scale-105 transition-transform duration-200 font-medium text-base"
          >
            {isPdfGenerating ? "Generating PDF..." : "📥 Download PDF Report"}
          </button>
        </div>
        <div className="whitespace-pre-line text-gray-900 text-base font-mono mt-4" style={{
          opacity: report ? 1 : 0.6,
          minHeight: 100,
          transition: 'opacity 0.6s',
          animation: report ? 'slide-fade-down 1.5s cubic-bezier(.8,0,.37,1) 1' : 'none'
        }}>
          {report || <span className="text-gray-500 italic">Click "Generate AI Report" to get a natural language assessment of this lake's water trends.</span>}
        </div>
      </div>

      <style>
        {`
          @keyframes fadein {
            from { opacity: 0; }
            to   { opacity: 1; }
          }
          @keyframes slide-fade-down {
            from { opacity: 0; transform: translateY(-30px);}
            to   { opacity: 1; transform: translateY(0);}
          }
        `}
      </style>
    </div>
  );
}
