import React from 'react';
import { AlertTriangle, ArrowUpRight, Activity, Droplets } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { waterBodies } from '../data/waterBodies';

// Utility function to determine if a water body is in critical state
const isCritical = (quality: typeof waterBodies[0]['quality']) => {
  return (
    quality.ph < 6.5 || 
    quality.ph > 8.5 ||
    quality.dissolvedOxygen < 5 ||
    quality.turbidity > 15 ||
    quality.tds > 500 ||
    quality.tss > 30 ||
    quality.chlorophyll > 4
  );
};

// Get critical water bodies
const criticalBodies = waterBodies.filter(body => isCritical(body.quality));

export default function CriticalStatus() {
  const navigate = useNavigate();

  const getStatusSeverity = (quality: typeof waterBodies[0]['quality']) => {
    const violations = [
      quality.ph < 6.5 || quality.ph > 8.5,
      quality.dissolvedOxygen < 5,
      quality.turbidity > 15,
      quality.tds > 500,
      quality.tss > 30,
      quality.chlorophyll > 4
    ].filter(Boolean).length;

    return violations >= 3 ? 'severe' : violations >= 2 ? 'moderate' : 'mild';
  };

  const getViolatedParameters = (quality: typeof waterBodies[0]['quality']) => {
    const violations = [];
    if (quality.ph < 6.5 || quality.ph > 8.5) violations.push('pH Level');
    if (quality.dissolvedOxygen < 5) violations.push('Dissolved Oxygen');
    if (quality.turbidity > 15) violations.push('Turbidity');
    if (quality.tds > 500) violations.push('TDS');
    if (quality.tss > 30) violations.push('TSS');
    if (quality.chlorophyll > 4) violations.push('Chlorophyll');
    return violations;
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Critical Status Report</h1>
          <p className="mt-2 text-gray-600">Water bodies requiring immediate attention</p>
        </div>
        <div className="flex items-center gap-2 bg-red-50 px-4 py-2 rounded-lg">
          <AlertTriangle className="h-5 w-5 text-red-600" />
          <span className="text-red-700 font-medium">{criticalBodies.length} Critical Issues</span>
        </div>
      </div>

      <div className="grid gap-6">
        {criticalBodies.map(body => {
          const severity = getStatusSeverity(body.quality);
          const violations = getViolatedParameters(body.quality);
          
          return (
            <div
              key={body.id}
              className={`bg-white rounded-lg shadow-md p-6 border-l-4 ${
                severity === 'severe' 
                  ? 'border-red-500' 
                  : severity === 'moderate'
                  ? 'border-orange-500'
                  : 'border-yellow-500'
              }`}
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">{body.name}</h2>
                  <p className="text-gray-600">{body.location}</p>
                </div>
                <button
                  onClick={() => navigate(`/lake/${encodeURIComponent(body.name.toLowerCase())}`)}
                  className="flex items-center gap-1 text-blue-600 hover:text-blue-800"
                >
                  View Details
                  <ArrowUpRight className="h-4 w-4" />
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <p className="text-sm text-gray-500 mb-1">Status Severity</p>
                    <div className="flex items-center gap-2">
                      <Activity className={`h-5 w-5 ${
                        severity === 'severe' 
                          ? 'text-red-600' 
                          : severity === 'moderate'
                          ? 'text-orange-600'
                          : 'text-yellow-600'
                      }`} />
                      <span className={`font-medium capitalize ${
                        severity === 'severe' 
                          ? 'text-red-700' 
                          : severity === 'moderate'
                          ? 'text-orange-700'
                          : 'text-yellow-700'
                      }`}>
                        {severity}
                      </span>
                    </div>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 mb-1">Water Level</p>
                    <div className="flex items-center gap-2">
                      <Droplets className="h-5 w-5 text-blue-600" />
                      <span className="font-medium">{body.waterLevel}%</span>
                    </div>
                  </div>
                </div>

                <div>
                  <p className="text-sm text-gray-500 mb-2">Critical Parameters</p>
                  <div className="space-y-2">
                    {violations.map((param, index) => (
                      <div
                        key={index}
                        className="flex items-center gap-2 text-sm bg-red-50 text-red-700 px-3 py-1.5 rounded"
                      >
                        <AlertTriangle className="h-4 w-4" />
                        {param}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t">
                <h3 className="text-sm font-medium text-gray-900 mb-3">Current Readings</h3>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                  <div className="text-sm">
                    <p className="text-gray-500">pH Level</p>
                    <p className={`font-medium ${body.quality.ph < 6.5 || body.quality.ph > 8.5 ? 'text-red-600' : 'text-gray-900'}`}>
                      {body.quality.ph}
                    </p>
                  </div>
                  <div className="text-sm">
                    <p className="text-gray-500">DO (mg/L)</p>
                    <p className={`font-medium ${body.quality.dissolvedOxygen < 5 ? 'text-red-600' : 'text-gray-900'}`}>
                      {body.quality.dissolvedOxygen}
                    </p>
                  </div>
                  <div className="text-sm">
                    <p className="text-gray-500">Turbidity (NTU)</p>
                    <p className={`font-medium ${body.quality.turbidity > 15 ? 'text-red-600' : 'text-gray-900'}`}>
                      {body.quality.turbidity}
                    </p>
                  </div>
                  <div className="text-sm">
                    <p className="text-gray-500">TDS (mg/L)</p>
                    <p className={`font-medium ${body.quality.tds > 500 ? 'text-red-600' : 'text-gray-900'}`}>
                      {body.quality.tds}
                    </p>
                  </div>
                  <div className="text-sm">
                    <p className="text-gray-500">TSS (mg/L)</p>
                    <p className={`font-medium ${body.quality.tss > 30 ? 'text-red-600' : 'text-gray-900'}`}>
                      {body.quality.tss}
                    </p>
                  </div>
                  <div className="text-sm">
                    <p className="text-gray-500">Chlorophyll (µg/L)</p>
                    <p className={`font-medium ${body.quality.chlorophyll > 4 ? 'text-red-600' : 'text-gray-900'}`}>
                      {body.quality.chlorophyll}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}