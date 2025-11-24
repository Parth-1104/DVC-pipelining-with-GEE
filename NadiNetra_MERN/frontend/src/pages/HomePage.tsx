import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { waterBodies } from '../data/waterBodies';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend, AreaChart, Area
} from 'recharts';
import { 
  Droplets, AlertTriangle, Activity, Map as MapIcon, 
  Wind, Sprout, RefreshCw, Leaf, CloudRain, Info
} from 'lucide-react';

// --- Interfaces ---
interface LakeData {
  id: string;
  name: string;
  location: string;
  turbidity: number;
  tss: number;
  chlorophyll: number;
  ndvi: number;
  ndwi: number;
  status: 'Good' | 'Warning' | 'Critical';
}

interface StatCardProps {
  title: string;
  value: string | number;
  unit?: string;
  icon: React.ReactNode;
  colorClass: string;
  tooltip?: string;
}

// --- Helper Components ---

const StatCard = ({ title, value, unit, icon, colorClass, tooltip }: StatCardProps) => (
  <div className="bg-white p-6 rounded-[2rem] shadow-sm border border-white hover:shadow-lg transition-all duration-300 group relative">
    <div className="flex items-center justify-between mb-4">
      <div className={`p-4 rounded-full ${colorClass} bg-opacity-10`}>
        {icon}
      </div>
      {tooltip && <Info size={16} className="text-gray-300 hover:text-gray-500 cursor-help transition-colors" />}
    </div>
    
    <div>
      <p className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-1">
        {title}
      </p>
      <p className="text-3xl font-bold text-[#0f2518] truncate" title={String(value)}>
        {value} <span className="text-sm text-gray-400 font-medium ml-1">{unit}</span>
      </p>
    </div>

    {tooltip && (
      <div className="absolute top-4 right-8 z-10 hidden group-hover:block w-48 p-3 bg-[#0f2518] text-white text-xs rounded-xl shadow-xl">
        {tooltip}
      </div>
    )}
  </div>
);

const StatusBadge = ({ status }: { status: string }) => {
  const colors = {
    Good: 'bg-green-100 text-green-800 border-green-200',
    Warning: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    Critical: 'bg-red-100 text-red-800 border-red-200'
  };
  
  return (
    <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide border ${colors[status as keyof typeof colors]}`}>
      {status}
    </span>
  );
};

// --- Caching Utilities ---
const CACHE_KEY = 'lakeDataCache';
const CACHE_EXPIRY_MINUTES = 15;

const saveToCache = (data: LakeData[]) => {
  localStorage.setItem(CACHE_KEY, JSON.stringify({
    timestamp: Date.now(),
    lakeData: data,
  }));
};

const getFromCache = (): LakeData[] | null => {
  const cache = localStorage.getItem(CACHE_KEY);
  if (cache) {
    try {
      const parsed = JSON.parse(cache);
      const age = (Date.now() - parsed.timestamp) / 1000 / 60;
      if (age < CACHE_EXPIRY_MINUTES) {
        return parsed.lakeData;
      }
      localStorage.removeItem(CACHE_KEY);
    } catch {
      localStorage.removeItem(CACHE_KEY);
    }
  }
  return null;
};

// --- Main Component ---

export default function HomePage() {
  const navigate = useNavigate();
  const [lakeData, setLakeData] = useState<LakeData[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchAllLakeData = async () => {
    setLoading(true);
    const promises = waterBodies.map(async (body) => {
      try {
        const response = await fetch('http://127.0.0.1:8000/currdate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ coordinates: body.coordinates }),
        });
        
        if (response.ok) {
          const data = await response.json();
          let status: 'Good' | 'Warning' | 'Critical' = 'Good';
          if (data["Turbidity NTU"] > 25) status = 'Critical';
          else if (data["Turbidity NTU"] > 15) status = 'Warning';

          return {
            id: body.id,
            name: body.name,
            location: body.location,
            turbidity: data["Turbidity NTU"],
            tss: data["TSS mg/L"],
            chlorophyll: data["Chlorophyll ug/L"],
            ndvi: data["NDVI"],
            ndwi: data["NDWI"],
            status
          };
        }
        return null;
      } catch (e) {
        console.error(`Error fetching ${body.name}`, e);
        return null;
      }
    });

    const results = await Promise.all(promises);
    const validResults = results.filter((item): item is LakeData => item !== null);
    setLakeData(validResults);
    saveToCache(validResults);
    setLastUpdated(new Date());
    setLoading(false);
  };

  useEffect(() => {
    const cachedData = getFromCache();
    if (cachedData && cachedData.length) {
      setLakeData(cachedData);
      setLastUpdated(new Date());
      setLoading(false);
    } else {
      fetchAllLakeData();
    }
  }, []);

  const handleRefresh = () => {
    localStorage.removeItem(CACHE_KEY);
    fetchAllLakeData();
  };

  // Stats
  const avgTurbidity = lakeData.length ? (lakeData.reduce((acc, curr) => acc + curr.turbidity, 0) / lakeData.length).toFixed(5) : '0.00000';
  const avgNDVI = lakeData.length ? (lakeData.reduce((acc, curr) => acc + curr.ndvi, 0) / lakeData.length).toFixed(5) : '0.00000';
  const avgNDWI = lakeData.length ? (lakeData.reduce((acc, curr) => acc + curr.ndwi, 0) / lakeData.length).toFixed(5) : '0.00000';
  const worstLakes = [...lakeData].sort((a, b) => b.turbidity - a.turbidity).slice(0, 3);
  
  const statusDistribution = [
    { name: 'Good', value: lakeData.filter(l => l.status === 'Good').length, color: '#22c55e' },
    { name: 'Warning', value: lakeData.filter(l => l.status === 'Warning').length, color: '#eab308' },
    { name: 'Critical', value: lakeData.filter(l => l.status === 'Critical').length, color: '#ef4444' },
  ].filter(item => item.value > 0);

  if (loading && !lakeData.length) {
    return (
      <div className="flex flex-col items-center justify-center h-[80vh]">
        <div className="relative">
           <div className="w-16 h-16 border-4 border-[#0f2518]/20 border-t-[#84cc16] rounded-full animate-spin"></div>
           <Leaf className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-[#0f2518]" size={20} />
        </div>
        <h2 className="text-xl font-bold text-[#0f2518] mt-6">Calibrating Satellite Data</h2>
        <p className="text-gray-500 mt-2">Connecting to Sentinel-2 feed...</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in pb-12 max-w-7xl mx-auto">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <h1 className="text-4xl font-bold text-[#0f2518]">Command Center</h1>
          <p className="mt-2 text-gray-500 flex items-center gap-2">
            System Status: <span className="text-[#84cc16] font-bold bg-[#0f2518] px-2 py-0.5 rounded text-xs">ONLINE</span>
            <span className="text-gray-300">|</span>
            Last Scan: {lastUpdated?.toLocaleTimeString()}
          </p>
        </div>
        <div className="flex gap-3">
            <button 
                onClick={handleRefresh}
                className="flex items-center gap-2 bg-white text-gray-700 border border-gray-200 px-6 py-3 rounded-full hover:bg-gray-50 hover:border-gray-300 transition-all font-medium text-sm shadow-sm"
            >
                <RefreshCw size={18} />
                Refresh Data
            </button>
            <button 
            onClick={() => navigate('/dashboard/map')}
            className="flex items-center gap-2 bg-[#0f2518] text-white px-6 py-3 rounded-full hover:bg-[#84cc16] hover:text-[#0f2518] transition-all font-bold text-sm shadow-lg shadow-[#0f2518]/20"
            >
            <MapIcon size={18} />
            Live Map
            </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard 
          title="Monitored Lakes" 
          value={lakeData.length} 
          icon={<Droplets className="w-6 h-6 text-blue-600" />}
          colorClass="bg-blue-100 text-blue-600"
        />
        <StatCard 
          title="Avg Turbidity" 
          value={avgTurbidity} 
          unit="NTU"
          icon={<Wind className="w-6 h-6 text-yellow-600" />}
          colorClass="bg-yellow-100 text-yellow-600"
        />
        <StatCard 
          title="Avg NDVI (Crop)" 
          value={avgNDVI} 
          tooltip="Assesses plant vigor"
          icon={<Leaf className="w-6 h-6 text-green-600" />}
          colorClass="bg-green-100 text-green-600"
        />
        <StatCard 
          title="Avg NDWI (Water)" 
          value={avgNDWI} 
          tooltip="Measures moisture"
          icon={<CloudRain className="w-6 h-6 text-cyan-600" />}
          colorClass="bg-cyan-100 text-cyan-600"
        />
      </div>

      {/* Agricultural Intelligence Section */}
      <div className="bg-[#0f2518] rounded-[2.5rem] p-8 md:p-10 shadow-xl relative overflow-hidden group text-white">
        {/* Abstract Background Shapes */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-[#84cc16] rounded-full blur-[120px] opacity-20 -mr-20 -mt-20 pointer-events-none group-hover:opacity-30 transition-opacity duration-700"></div>
        
        <div className="relative z-10 flex flex-col lg:flex-row gap-10">
            {/* Info Column */}
            <div className="lg:w-1/3 space-y-6">
                <div className="flex items-center gap-3 mb-2">
                    <div className="p-3 bg-white/10 rounded-2xl backdrop-blur-md border border-white/10">
                        <Sprout className="w-8 h-8 text-[#84cc16]" />
                    </div>
                    <div>
                        <h2 className="text-2xl font-bold">Agri-Intel</h2>
                        <p className="text-[#84cc16] text-sm font-medium">Real-time Crop & Soil Analysis</p>
                    </div>
                </div>

                <div className="space-y-4">
                    <div className="bg-white/5 p-5 rounded-2xl border border-white/5 hover:bg-white/10 transition-colors">
                        <h4 className="font-bold text-[#84cc16] mb-2 flex items-center gap-2 text-sm uppercase tracking-wide">
                            <Leaf size={14}/> Crop Health (NDVI)
                        </h4>
                        <p className="text-sm text-gray-300 leading-relaxed">
                            Detect early signs of crop stress, nutrient deficiencies, or disease before they become visible to the naked eye.
                        </p>
                    </div>
                    
                    <div className="bg-white/5 p-5 rounded-2xl border border-white/5 hover:bg-white/10 transition-colors">
                        <h4 className="font-bold text-cyan-400 mb-2 flex items-center gap-2 text-sm uppercase tracking-wide">
                            <Droplets size={14}/> Water Stress (NDWI)
                        </h4>
                        <p className="text-sm text-gray-300 leading-relaxed">
                            Optimize irrigation by measuring vegetation water content and soil moisture deficits.
                        </p>
                    </div>
                </div>
            </div>

            {/* Chart Column */}
            <div className="lg:w-2/3 bg-white/5 rounded-3xl p-6 border border-white/10 backdrop-blur-sm">
                <h3 className="font-bold text-white mb-6 flex items-center gap-2">
                    <Activity size={18} className="text-[#84cc16]" />
                    Vegetation vs. Moisture Correlation
                </h3>
                <div className="h-[320px]">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={lakeData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                    <defs>
                        <linearGradient id="colorNdvi" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#84cc16" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#84cc16" stopOpacity={0}/>
                        </linearGradient>
                        <linearGradient id="colorNdwi" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                        </linearGradient>
                    </defs>
                    <XAxis dataKey="name" tick={{fontSize: 10, fill: '#9ca3af'}} interval={0} angle={-15} textAnchor="end" height={50} stroke="#4b5563"/>
                    <YAxis tick={{fontSize: 11, fill: '#9ca3af'}} stroke="#4b5563" />
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#374151" />
                    <Tooltip 
                        contentStyle={{borderRadius: '16px', border: 'none', backgroundColor: '#1f2937', color: '#fff'}}
                        itemStyle={{color: '#fff'}}
                        formatter={(val: number) => val.toFixed(5)}
                    />
                    <Legend verticalAlign="top" height={36}/>
                    <Area type="monotone" dataKey="ndvi" stroke="#84cc16" fillOpacity={1} fill="url(#colorNdvi)" name="NDVI (Crop)" />
                    <Area type="monotone" dataKey="ndwi" stroke="#06b6d4" fillOpacity={1} fill="url(#colorNdwi)" name="NDWI (Moisture)" />
                    </AreaChart>
                </ResponsiveContainer>
                </div>
            </div>
        </div>
      </div>

      {/* Main Water Quality Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left: Detailed Bar Chart */}
        <div className="lg:col-span-2 bg-white p-8 rounded-[2.5rem] shadow-sm border border-white">
  <div className="flex justify-between items-center mb-8">
    <h3 className="text-xl font-bold text-[#0f2518]">Turbidity Analysis</h3>
    <div className="text-xs font-bold bg-red-100 text-red-600 px-3 py-1 rounded-full uppercase tracking-wide">Threshold: 25 NTU</div>
  </div>
  <div className="h-[400px] w-full">
    <ResponsiveContainer width="100%" height="100%" >
      <BarChart data={lakeData} margin={{ top: 20, right: 30, left: 20, bottom: 40 }} className='p-4'>
        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
        <XAxis 
          dataKey="name" 
          tick={{fontSize: 11, fill: '#6b7280'}} 
          interval={0} 
          angle={-20} 
          textAnchor="end"
        />
        <YAxis
          label={{ value: 'Turbidity (NTU)', angle: -90, position: 'insideLeft', fontSize: 12, fill: '#6b7280' }}
          domain={['auto', 'auto']}
          tickFormatter={val => Number(val).toFixed(9)}
        />
        <Tooltip
          cursor={{fill: '#F3F0EA'}}
          formatter={(value: number) => [value.toFixed(9), 'Turbidity']}
          contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)'}}
        />
        <Bar dataKey="turbidity" name="Turbidity" radius={[6, 6, 0, 0]}>
          {lakeData.map((entry, index) => (
            <Cell 
                key={`cell-${index}`}
                fill={entry.turbidity > 25 ? '#ef4444' : entry.turbidity > 15 ? '#eab308' : '#3b82f6'} 
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  </div>
</div>


        {/* Right: Health Distribution Pie Chart */}
        <div className="bg-white p-8 rounded-[2.5rem] shadow-sm border border-white flex flex-col">
          <h3 className="text-xl font-bold text-[#0f2518] mb-2">Health Distribution</h3>
          <p className="text-sm text-gray-400 mb-8">Safety across monitored bodies</p>
          
          <div className="flex-1 min-h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={statusDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={70}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                  cornerRadius={6}
                >
                  {statusDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} stroke="none" />
                  ))}
                </Pie>
                <Tooltip contentStyle={{borderRadius: '12px', border: 'none'}} />
                <Legend verticalAlign="bottom" height={36} iconType="circle"/>
              </PieChart>
            </ResponsiveContainer>
          </div>
          
          <div className="mt-6 p-5 bg-[#F3F0EA] rounded-2xl border border-gray-200/50 text-center">
             <div className="text-3xl font-bold text-[#0f2518] mb-1">
                {((lakeData.filter(l => l.status === 'Good').length / lakeData.length || 0) * 100).toFixed(0)}%
             </div>
             <div className="text-xs font-bold text-gray-500 uppercase tracking-widest">Safe for Irrigation</div>
          </div>
        </div>
      </div>

      {/* Pollution Watchlist */}
      <div className="bg-white p-8 rounded-[2.5rem] shadow-sm border border-white">
          <div className="flex items-center gap-3 mb-8">
              <div className="p-2 bg-red-100 text-red-600 rounded-lg">
                <AlertTriangle size={24} />
              </div>
              <h3 className="text-xl font-bold text-[#0f2518]">Priority Watchlist</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {worstLakes.map((lake, idx) => (
                  <div 
                      key={lake.id} 
                      onClick={() => navigate(`/dashboard/lake/${encodeURIComponent(lake.name.toLowerCase())}`)}
                      className="flex items-center justify-between p-5 bg-[#F3F0EA] rounded-2xl cursor-pointer hover:bg-white hover:shadow-md hover:scale-[1.02] transition-all duration-300 border border-transparent hover:border-gray-100"
                  >
                      <div className="flex items-center gap-4">
                          <span className="flex items-center justify-center w-8 h-8 rounded-full bg-red-100 text-red-600 font-bold text-sm">#{idx + 1}</span>
                          <div>
                              <h4 className="font-bold text-[#0f2518] text-lg">{lake.name}</h4>
                              <p className="text-xs text-gray-500 uppercase tracking-wide font-medium">{lake.location}</p>
                          </div>
                      </div>
                      <div className="text-right">
                          <span className="block font-bold text-red-600 text-lg mb-1">{lake.turbidity.toFixed(9)}</span>
                          <StatusBadge status="Critical" />
                      </div>
                  </div>
              ))}
          </div>
      </div>
    </div>
  );
}