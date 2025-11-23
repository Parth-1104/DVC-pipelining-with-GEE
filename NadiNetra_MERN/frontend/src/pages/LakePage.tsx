import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Brush
} from 'recharts';
import { waterBodies } from '../data/waterBodies';
import TimeSpanSelector from '../components/TimeSpanSelector';
import { subDays, subMonths, subYears, format, parseISO } from 'date-fns';

const DECIMALS = 7;

const metricConfig = [
  { key: "turbidity", label: "Turbidity NTU", color: "#ca8a04", icon: "💧" },
  { key: "tss", label: "TSS mg/L", color: "#7c3aed", icon: "🟣" },
  { key: "chlorophyll", label: "Chlorophyll µg/L", color: "#059669", icon: "🌱" },
  { key: "ndvi", label: "NDVI", color: "#2563eb", icon: "🟩" },
  { key: "ndwi", label: "NDWI", color: "#06b6d4", icon: "🟦" },
];

function formatNumber(value: number | undefined) {
  return value !== undefined ? value.toFixed(DECIMALS) : '--';
}

export default function LakePage() {
  const { name } = useParams();
  const [timeSpan, setTimeSpan] = useState('1M');
  const [chartData, setChartData] = useState([]);
  const [currentReading, setCurrentReading] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  const lake = waterBodies.find(body => body.name.toLowerCase() === decodeURIComponent(name || ''));

  useEffect(() => {
    if (!lake) return;
    const fetchData = async () => {
      setLoading(true);
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
      try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            coordinates: lake.coordinates,
            start_date: formattedStartDate,
            end_date: formattedEndDate
          }),
        });
        if (!response.ok) throw new Error('Failed to fetch prediction data');
        const data = await response.json();
        const normalizedData = (data.predictions || []).map(item => ({
          date: item.date,
          turbidity: item["Turbidity NTU"],
          tss: item["TSS mg/L"],
          chlorophyll: item["Chlorophyll ug/L"],
          ndvi: item["NDVI"],
          ndwi: item["NDWI"]
        }));
        setChartData(normalizedData);
        setCurrentReading(normalizedData.length > 0 ? normalizedData[normalizedData.length - 1] : null);
      } catch (err) {
        console.error(err);
        setError('Unable to load water quality data');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [lake, timeSpan]);

  async function handleGenerateReport() {
    if (!lake || chartData.length === 0) return;
    setIsGenerating(true);
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
    } catch (err) {
      setReport('Error calling Gemini API.');
    } finally {
      setIsGenerating(false);
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

  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">{lake.name}</h1>
        <p className="mt-2 text-gray-600">Location: {lake.location}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="bg-white p-7 rounded-xl shadow-md">
          <h3 className="text-lg font-semibold mb-4 text-gray-800">Lake Details</h3>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-gray-500">Area</span>
              <span className="text-xl font-bold">{lake.area} ha</span>
            </div>
          </div>
        </div>

        <div className="bg-white p-7 rounded-xl shadow-md col-span-2 flex flex-col">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-800">Parameter Trends (Scatter)</h3>
            <TimeSpanSelector selectedSpan={timeSpan} onSpanChange={setTimeSpan} />
          </div>
          <div style={{ flex: 1 }}>
            {loading ? (
              <div className="h-full flex items-center justify-center text-gray-500">Loading chart data...</div>
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="4 4" />
                  <XAxis 
                    dataKey="date"
                    type="category"
                    tickFormatter={(date) => {
                      try { return format(parseISO(date), 'MMM d'); } catch { return date; }
                    }}
                  />
                  <YAxis
                    tickFormatter={(value: number) => value.toFixed(7)}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip
                    formatter={(val: number, name: string) => formatNumber(val)}
                    labelFormatter={(date: string) => {
                      try { return format(parseISO(date), 'MMM d, yyyy'); }
                      catch { return date; }
                    }}
                  />
                  <Legend />
                  <Brush dataKey="date" height={30} stroke="#aaa" />
                  {metricConfig.slice(0, 3).map(metric => (
                    <Scatter
                      key={metric.key}
                      name={metric.label}
                      data={chartData}
                      fill={metric.color}
                      dataKey={metric.key}
                      shape="circle"
                    />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      {/* Analytics Table */}
      <div className="bg-white p-7 rounded-xl shadow-md overflow-x-auto">
        <h3 className="text-lg font-semibold mb-2">Recent Measurements (7 decimals)</h3>
        <table className="min-w-full border-collapse border">
          <thead>
            <tr>
              <th>Date</th>
              {metricConfig.map(m => (
                <th key={m.key} style={{color: m.color}}>{m.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {chartData.slice(-7).map((row, idx) => (
              <tr key={row.date + idx}>
                <td>{row.date}</td>
                {metricConfig.map(m => (
                  <td key={m.key}>{formatNumber(row[m.key])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Info Cards */}
      <div className="bg-white p-7 rounded-xl shadow-md">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-800">Latest Readings</h3>
          {currentReading && (
            <span className="text-sm text-gray-500">
              Date: {currentReading.date}
            </span>
          )}
        </div>
        {loading ? (
          <div className="text-center py-6">Loading latest values...</div>
        ) : error ? (
          <div className="text-center py-6 text-red-500">{error}</div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-7">
            {metricConfig.map((metric) => (
              <div key={metric.key} className="p-4 rounded-lg flex flex-col items-center bg-gradient-to-tr shadow-sm"
                style={{
                  background: `linear-gradient(130deg, ${metric.color}30 40%, #fff 100%)`
                }}>
                <span className="text-3xl mb-2">{metric.icon}</span>
                <span className={`text-sm font-medium`} style={{ color: metric.color }}>
                  {metric.label}
                </span>
                <span className="text-lg font-bold text-gray-900 mt-2">
                  {formatNumber(currentReading ? currentReading[metric.key] : undefined)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Gemini AI Report Section */}
      <div className="bg-gray-50 p-6 rounded-lg mt-8">
        <h3 className="text-lg font-bold mb-4">AI-Powered Narrative Report</h3>
        <button
          onClick={handleGenerateReport}
          disabled={isGenerating || !chartData.length}
          className="bg-blue-600 text-white px-4 py-2 rounded shadow hover:bg-blue-700 mb-4"
        >
          {isGenerating ? "Generating..." : "Generate AI Report"}
        </button>
        <div className="whitespace-pre-line text-gray-900 text-base mt-4 max-w-2xl mx-auto">{report}</div>
      </div>
    </div>
  );
}
