import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { waterBodies } from '../data/waterBodies';
import { 
  Activity, 
  MapPin, 
  ArrowRight, 
  Wind,      // For Turbidity
  Waves,     // For TSS
  Droplets,  // For Chlorophyll
  Clock
} from 'lucide-react';

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
          // Simulating API call or using real endpoint
          // In a real scenario, handle failures gracefully without crashing Promise.all
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
          } else {
             // Fallback mock data if API fails (for demonstration)
             newMeasurements[body.id] = {
                "TSS mg/L": Math.random() * 20,
                "Turbidity NTU": Math.random() * 15,
                "Chlorophyll ug/L": Math.random() * 5,
                "date": new Date().toISOString().split('T')[0],
                "NDVI": 0.4,
                "NDWI": -0.1
             };
          }
        } catch (error) {
          console.error(`Failed to fetch data for ${body.name}`, error);
        }
      });

      await Promise.all(promises);
      setMeasurements(newMeasurements);
      setLoading(false);
    };

    fetchAllData();
  }, []);

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-4xl font-bold text-[#0f2518]">Monitored Bodies</h1>
          <p className="mt-2 text-gray-500">Real-time satellite analysis of regional water resources.</p>
        </div>
        <div className="bg-white px-4 py-2 rounded-full border border-gray-200 text-xs font-bold uppercase tracking-widest text-gray-400">
           Total Count: {waterBodies.length}
        </div>
      </div>

      {/* Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {waterBodies.map(body => {
          const data = measurements[body.id];
          const isLoading = loading && !data;

          return (
            <div
              key={body.id}
              onClick={() => navigate(`/dashboard/lake/${encodeURIComponent(body.name.toLowerCase())}`)}
              className="bg-white p-8 rounded-[2.5rem] shadow-sm border border-white hover:border-[#84cc16]/50 hover:shadow-xl hover:-translate-y-1 transition-all duration-300 cursor-pointer group flex flex-col h-full justify-between"
            >
              {/* Card Header */}
              <div className="flex justify-between items-start mb-6">
                <div className="flex items-start gap-4">
                   <div className="w-12 h-12 rounded-2xl bg-[#0f2518] flex items-center justify-center text-[#84cc16] shadow-lg shadow-[#0f2518]/10 group-hover:scale-110 transition-transform">
                      <Waves size={24} />
                   </div>
                   <div>
                      <h2 className="text-2xl font-bold text-[#0f2518] mb-1 group-hover:text-[#84cc16] transition-colors">{body.name}</h2>
                      <div className="flex items-center gap-1 text-sm text-gray-400 font-medium">
                         <MapPin size={14} />
                         {body.location}
                      </div>
                   </div>
                </div>
                
                {/* Status Indicator */}
                <div className={`px-3 py-1 rounded-full border text-[10px] font-bold uppercase tracking-wide flex items-center gap-2 ${data ? 'bg-green-50 text-green-700 border-green-200' : 'bg-gray-50 text-gray-400 border-gray-200'}`}>
                   <span className={`w-2 h-2 rounded-full ${data ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`}></span>
                   {data ? 'Live Feed' : 'Offline'}
                </div>
              </div>
              
              {/* Metrics Grid */}
              <div className="grid grid-cols-3 gap-4 mb-8">
                
                {/* Metric 1: Turbidity */}
                <div className="bg-[#F3F0EA] rounded-2xl p-4 transition-colors group-hover:bg-[#0f2518]/5">
                   <div className="flex items-center gap-2 text-gray-400 mb-2 text-xs font-bold uppercase tracking-wider">
                      <Wind size={12} /> Turbidity
                   </div>
                   <div className="text-lg font-bold text-[#0f2518]">
                      {data ? `${data["Turbidity NTU"].toFixed(1)}` : '--'} 
                      <span className="text-xs font-medium text-gray-400 ml-1">NTU</span>
                   </div>
                </div>

                {/* Metric 2: TSS */}
                <div className="bg-[#F3F0EA] rounded-2xl p-4 transition-colors group-hover:bg-[#0f2518]/5">
                   <div className="flex items-center gap-2 text-gray-400 mb-2 text-xs font-bold uppercase tracking-wider">
                      <Activity size={12} /> TSS
                   </div>
                   <div className="text-lg font-bold text-[#0f2518]">
                      {data ? `${data["TSS mg/L"].toFixed(1)}` : '--'}
                      <span className="text-xs font-medium text-gray-400 ml-1">mg/L</span>
                   </div>
                </div>

                {/* Metric 3: Chlorophyll */}
                <div className="bg-[#F3F0EA] rounded-2xl p-4 transition-colors group-hover:bg-[#0f2518]/5">
                   <div className="flex items-center gap-2 text-gray-400 mb-2 text-xs font-bold uppercase tracking-wider">
                      <Droplets size={12} /> Chl-a
                   </div>
                   <div className="text-lg font-bold text-[#0f2518]">
                      {data ? `${data["Chlorophyll ug/L"].toFixed(2)}` : '--'}
                      <span className="text-xs font-medium text-gray-400 ml-1">µg/L</span>
                   </div>
                </div>

              </div>

              {/* Card Footer */}
              <div className="flex items-center justify-between border-t border-gray-100 pt-4">
                 <div className="flex items-center gap-2 text-xs text-gray-400 font-medium">
                    <Clock size={14} />
                    Last Updated: <span className="text-gray-600">{data ? data.date : 'Syncing...'}</span>
                 </div>
                 
                 <div className="w-10 h-10 rounded-full border border-gray-200 flex items-center justify-center text-gray-400 group-hover:bg-[#84cc16] group-hover:text-[#0f2518] group-hover:border-[#84cc16] transition-all">
                    <ArrowRight size={18} />
                 </div>
              </div>

            </div>
          );
        })}
      </div>
    </div>
  );
}