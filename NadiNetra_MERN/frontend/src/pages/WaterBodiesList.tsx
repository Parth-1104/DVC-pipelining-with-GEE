import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { waterBodies } from '../data/waterBodies';
import { Activity } from 'lucide-react';

// Define the shape of the API response
interface APIResponse {
  "TSS mg/L": number;
  "Turbidity NTU": number;
  "Chlorophyll ug/L": number;
  "date": string;
  "NDVI": number;
  "NDWI": number;
}

export default function WaterBodiesList() {
  const navigate = useNavigate();
  
  // Store measurements keyed by water body ID
  const [measurements, setMeasurements] = useState<Record<string, APIResponse>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAllData = async () => {
      const newMeasurements: Record<string, APIResponse> = {};
      
      // Create an array of fetch promises for all water bodies
      const promises = waterBodies.map(async (body) => {
        try {
          const response = await fetch('http://127.0.0.1:8000/currdate', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              coordinates: body.coordinates
            }),
          });
          
          if (response.ok) {
            const data: APIResponse = await response.json();
            newMeasurements[body.id] = data;
          }
        } catch (error) {
          console.error(`Failed to fetch data for ${body.name}`, error);
        }
      });

      // Wait for all requests to finish
      await Promise.all(promises);
      setMeasurements(newMeasurements);
      setLoading(false);
    };

    fetchAllData();
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Water Bodies</h1>
        <p className="mt-2 text-gray-600">Complete list of monitored water bodies in Delhi</p>
      </div>

      <div className="grid gap-6">
        {waterBodies.map(body => {
          const data = measurements[body.id];

          return (
            <div
              key={body.id}
              className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow cursor-pointer"
              onClick={() => navigate(`/dashboard/lake/${encodeURIComponent(body.name.toLowerCase())}`)}
            >
              <div className="flex items-start justify-between">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">{body.name}</h2>
                  <p className="text-gray-600">{body.location}</p>
                </div>
                {/* Icon is green if we have live data, otherwise gray */}
                <div className={`p-2 rounded-full ${data ? 'bg-green-50' : 'bg-gray-50'}`}>
                  <Activity className={`h-6 w-6 ${data ? 'text-green-600' : 'text-gray-400'}`} />
                </div>
              </div>
              
              <div className="mt-4 grid grid-cols-3 gap-4">
                {/* REPLACED: Area -> Turbidity */}
                <div>
                  <p className="text-sm text-gray-500">Turbidity</p>
                  <p className="font-medium">
                    {data ? `${data["Turbidity NTU"].toFixed(2)} NTU` : loading ? 'Loading...' : '--'}
                  </p>
                </div>

                {/* REPLACED: Water Level -> TSS */}
                <div>
                  <p className="text-sm text-gray-500">TSS</p>
                  <p className="font-medium">
                    {data ? `${data["TSS mg/L"].toFixed(2)} mg/L` : loading ? 'Loading...' : '--'}
                  </p>
                </div>

                {/* REPLACED: pH Level -> Chlorophyll */}
                <div>
                  <p className="text-sm text-gray-500">Chlorophyll</p>
                  <p className="font-medium">
                    {data ? `${data["Chlorophyll ug/L"].toFixed(2)} µg/L` : loading ? 'Loading...' : '--'}
                  </p>
                </div>
              </div>

              <div className="mt-4 text-sm text-gray-500">
                Last updated: {data ? data.date : 'Fetching live data...'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}