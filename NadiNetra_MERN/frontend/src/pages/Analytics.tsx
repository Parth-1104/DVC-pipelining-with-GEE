import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { MapContainer, TileLayer, Circle } from 'react-leaflet';
import { Activity, AlertTriangle, MessageSquare, TrendingUp } from 'lucide-react';
import { format, parseISO } from 'date-fns';
import { historicalData, predictions, communityReports, pollutionHeatmapData } from '../data/analyticsData';

export default function Analytics() {
  const [selectedMetric, setSelectedMetric] = useState<'ph' | 'dissolvedOxygen' | 'turbidity' | 'pollutants' | 'biodiversity'>('ph');

  const metricOptions = [
    { value: 'ph', label: 'pH Levels', color: '#2563eb' },
    { value: 'dissolvedOxygen', label: 'Dissolved Oxygen', color: '#16a34a' },
    { value: 'turbidity', label: 'Turbidity', color: '#ca8a04' },
    { value: 'pollutants', label: 'Pollutant Levels', color: '#dc2626' },
    { value: 'biodiversity', label: 'Biodiversity Index', color: '#7c3aed' }
  ];

  const calculateWQI = (data: typeof historicalData[0]) => {
    return (
      (data.ph * 0.2 +
        data.dissolvedOxygen * 0.25 +
        (100 - data.turbidity) * 0.15 +
        (100 - data.pollutants) * 0.25 +
        data.biodiversity * 0.15)
    ).toFixed(1);
  };

  const latestWQI = calculateWQI(historicalData[historicalData.length - 1]);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Analytics Dashboard</h1>
        <p className="mt-2 text-gray-600">Comprehensive analysis of water quality parameters and predictions</p>
      </div>

      {/* Live WQI Section */}
      {/* <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Live Water Quality Index</h2>
            <p className="text-3xl font-bold mt-2">{latestWQI}</p>
          </div>
          <div className={`p-4 rounded-full ${Number(latestWQI) >= 75 ? 'bg-green-100' : 'bg-yellow-100'}`}>
            <Activity className={`h-8 w-8 ${Number(latestWQI) >= 75 ? 'text-green-600' : 'text-yellow-600'}`} />
          </div>
        </div>
      </div> */}

      {/* Historical Data Section */}
      {/* <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Historical Trends</h2>
        <div className="space-x-2 mb-4">
          {metricOptions.map(option => (
            <button
              key={option.value}
              onClick={() => setSelectedMetric(option.value as any)}
              className={`px-4 py-2 rounded-lg text-sm ${
                selectedMetric === option.value
                  ? 'bg-gray-900 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickFormatter={date => format(parseISO(date), 'MMM d')}
              />
              <YAxis />
              <Tooltip
                labelFormatter={date => format(parseISO(date as string), 'MMM d, yyyy')}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey={selectedMetric}
                stroke={metricOptions.find(o => o.value === selectedMetric)?.color}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div> */}

      {/* Pollution Heatmap */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Pollution Intensity Map</h2>
        <div className="h-[400px]">
          <MapContainer
            center={[28.6139, 77.2090]}
            zoom={11}
            className="h-full w-full"
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {pollutionHeatmapData.data.map((point, index) => (
              <Circle
                key={index}
                center={[point.lat, point.lng]}
                radius={1000}
                pathOptions={{
                  color: `hsl(${(1 - point.value / 100) * 120}, 70%, 50%)`,
                  fillOpacity: 0.6
                }}
              />
            ))}
          </MapContainer>
        </div>
      </div>

      {/* Predictive Analytics */}
      {/* <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Water Quality Predictions</h2>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={predictions}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickFormatter={date => format(parseISO(date), 'MMM d')}
              />
              <YAxis domain={[60, 100]} />
              <Tooltip
                labelFormatter={date => format(parseISO(date as string), 'MMM d, yyyy')}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="actual"
                stroke="#2563eb"
                strokeWidth={2}
                name="Actual WQI"
              />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#dc2626"
                strokeWidth={2}
                name="Predicted WQI"
                strokeDasharray="5 5"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div> */}

      {/* Community Reports */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Community Reports</h2>
        <div className="space-y-4">
          {communityReports.map(report => (
            <div key={report.id} className="border-b pb-4 last:border-0">
              <div className="flex items-start justify-between">
                <div>
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize" style={{
                    backgroundColor: report.status === 'resolved' ? '#dcfce7' : report.status === 'investigating' ? '#fef9c3' : '#fee2e2',
                    color: report.status === 'resolved' ? '#16a34a' : report.status === 'investigating' ? '#ca8a04' : '#dc2626'
                  }}>
                    {report.status}
                  </span>
                  <p className="mt-2 text-sm text-gray-900">{report.description}</p>
                  <p className="mt-1 text-xs text-gray-500">
                    Reported on {format(parseISO(report.createdAt), 'MMM d, yyyy')}
                  </p>
                </div>
                <div className="p-2 rounded-full bg-gray-100">
                  {report.reportType === 'pollution' ? (
                    <AlertTriangle className="h-5 w-5 text-red-600" />
                  ) : (
                    <MessageSquare className="h-5 w-5 text-blue-600" />
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}