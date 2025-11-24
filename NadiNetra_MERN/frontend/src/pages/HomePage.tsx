import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { waterBodies } from '../data/waterBodies';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend, AreaChart, Area
} from 'recharts';
import { 
  Droplets, AlertTriangle, Activity, Map as MapIcon, 
  Wind, Waves, Sprout, RefreshCw, Leaf, CloudRain, Info
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
  <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow group relative">
    <div className="flex items-center justify-between">
      <div className="overflow-hidden">
        <p className="text-sm font-medium text-gray-500 flex items-center gap-1">
          {title}
          {tooltip && <Info size={14} className="text-gray-400 cursor-help" />}
        </p>
        <p className="text-2xl font-bold text-gray-900 mt-2 truncate" title={String(value)}>
          {value} <span className="text-sm text-gray-500 font-normal">{unit}</span>
        </p>
      </div>
      <div className={`p-3 rounded-full ${colorClass} bg-opacity-10 shrink-0 ml-4`}>
        {icon}
      </div>
    </div>
    {tooltip && (
      <div className="absolute top-full left-0 mt-2 z-10 hidden group-hover:block w-64 p-3 bg-gray-800 text-white text-xs rounded shadow-lg">
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
    <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium border ${colors[status as keyof typeof colors]}`}>
      {status}
    </span>
  );
};

// --- Caching Utilities ---
const CACHE_KEY = 'lakeDataCache';
const CACHE_EXPIRY_MINUTES = 15; // Cache expiry time in minutes

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
      localStorage.removeItem(CACHE_KEY); // Remove expired cache
    } catch {
      localStorage.removeItem(CACHE_KEY); // Remove corrupted cache
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

  // On mount, use cached data if available, otherwise fetch
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

  // Manual refresh handler to clear cache & refetch
  const handleRefresh = () => {
    localStorage.removeItem(CACHE_KEY);
    fetchAllLakeData();
  };

  // --- Derived Statistics ---
  const avgTurbidity = lakeData.length ? (lakeData.reduce((acc, curr) => acc + curr.turbidity, 0) / lakeData.length).toFixed(5) : '0.00000';
  const avgNDVI = lakeData.length ? (lakeData.reduce((acc, curr) => acc + curr.ndvi, 0) / lakeData.length).toFixed(5) : '0.00000';
  const avgNDWI = lakeData.length ? (lakeData.reduce((acc, curr) => acc + curr.ndwi, 0) / lakeData.length).toFixed(5) : '0.00000';
  const criticalCount = lakeData.filter(l => l.status === 'Critical').length;

  const worstLakes = [...lakeData].sort((a, b) => b.turbidity - a.turbidity).slice(0, 3);
  
  const statusDistribution = [
    { name: 'Good', value: lakeData.filter(l => l.status === 'Good').length, color: '#22c55e' },
    { name: 'Warning', value: lakeData.filter(l => l.status === 'Warning').length, color: '#eab308' },
    { name: 'Critical', value: lakeData.filter(l => l.status === 'Critical').length, color: '#ef4444' },
  ].filter(item => item.value > 0);

  if (loading && !lakeData.length) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh]">
        <Activity className="w-12 h-12 text-blue-600 animate-spin mb-4" />
        <h2 className="text-xl font-semibold text-gray-700">Calibrating Satellite Data...</h2>
        <p className="text-gray-500">Fetching agricultural & water quality metrics</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Command Center</h1>
          <p className="mt-2 text-gray-600 flex items-center gap-2">
            System Status: <span className="text-green-600 font-semibold">Online</span>
            <span className="text-gray-300">|</span>
            Last Scan: {lastUpdated?.toLocaleTimeString()}
          </p>
        </div>
        <div className="flex gap-3">
            <button 
                onClick={handleRefresh}
                className="flex items-center gap-2 bg-white text-gray-700 border border-gray-300 px-4 py-2 rounded-lg hover:bg-gray-50 transition-colors"
            >
                <RefreshCw size={18} />
                Refresh
            </button>
            <button 
            onClick={() => navigate('/dashboard/map')}
            className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors shadow-sm"
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
          tooltip="Assesses plant vigor and biomass"
          icon={<Leaf className="w-6 h-6 text-green-600" />}
          colorClass="bg-green-100 text-green-600"
        />
        <StatCard 
          title="Avg NDWI (Water)" 
          value={avgNDWI} 
          tooltip="Measures vegetation water content & soil moisture"
          icon={<CloudRain className="w-6 h-6 text-cyan-600" />}
          colorClass="bg-cyan-100 text-cyan-600"
        />
      </div>

      {/* Agricultural Intelligence Section */}
      <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-8 border border-green-100">
        <div className="flex items-start gap-4 mb-6">
          <div className="p-3 bg-green-100 rounded-lg">
            <Sprout className="w-8 h-8 text-green-700" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-green-900">Agricultural Intelligence</h2>
            <p className="text-green-700 mt-1">Real-time crop health and irrigation insights</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-white/60 p-5 rounded-lg border border-green-100 shadow-sm">
              <h4 className="font-semibold text-green-800 mb-2 flex items-center gap-2">
                <Leaf size={16}/> Crop Health (NDVI)
              </h4>
              <p className="text-sm text-gray-700 leading-relaxed">
                NDVI is used to assess overall plant vigor, biomass, and health. It helps farmers detect early signs of crop stress, nutrient deficiencies, or disease. It also supports yield estimation and tracking crop development.
              </p>
            </div>
            
            <div className="bg-white/60 p-5 rounded-lg border border-blue-100 shadow-sm">
              <h4 className="font-semibold text-blue-800 mb-2 flex items-center gap-2">
                <Droplets size={16}/> Water Stress (NDWI)
              </h4>
              <p className="text-sm text-gray-700 leading-relaxed">
                NDWI measures vegetation water content and soil moisture, valuable for detecting water stress before visible symptoms. It assists in irrigation management by showing moisture deficits.
              </p>
            </div>

            <div className="p-4 bg-yellow-50/80 rounded-lg border border-yellow-100">
              <p className="text-xs text-yellow-800 font-medium">
                <strong>Impact:</strong> Together, these indices enable precision agriculture (variable-rate irrigation, fertilization) to improve resource efficiency and yields.
              </p>
            </div>
          </div>

          <div className="lg:col-span-2 bg-white p-6 rounded-lg shadow-sm border border-gray-100">
            <h3 className="font-bold text-gray-800 mb-4">Vegetation vs. Water Content Analysis</h3>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={lakeData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorNdvi" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#22c55e" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#22c55e" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorNdwi" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="name" tick={{fontSize: 10}} interval={0} angle={-15} textAnchor="end" height={50}/>
                  <YAxis tick={{fontSize: 11}} />
                  <CartesianGrid strokeDasharray="3 3" vertical={false}/>
                  <Tooltip 
                    contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}}
                    formatter={(val: number) => val.toFixed(5)}
                  />
                  <Legend verticalAlign="top" height={36}/>
                  <Area type="monotone" dataKey="ndvi" stroke="#22c55e" fillOpacity={1} fill="url(#colorNdvi)" name="NDVI (Vegetation)" />
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
        <div className="lg:col-span-2 bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-lg font-bold text-gray-900">Turbidity Levels by Water Body</h3>
            <div className="text-sm text-gray-500">Threshold: 25 NTU (Critical)</div>
          </div>
          <div className="h-[350px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={lakeData} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                <XAxis 
                    dataKey="name" 
                    tick={{fontSize: 11}} 
                    interval={0} 
                    angle={-20} 
                    textAnchor="end"
                />
                <YAxis label={{ value: 'Turbidity (NTU)', angle: -90, position: 'insideLeft', fontSize: 12 }} />
                <Tooltip 
                  cursor={{fill: '#f9fafb'}}
                  formatter={(value: number) => [value.toFixed(5), 'Turbidity']}
                  contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)'}}
                />
                <Bar dataKey="turbidity" name="Turbidity" radius={[4, 4, 0, 0]}>
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
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex flex-col">
          <h3 className="text-lg font-bold text-gray-900 mb-2">Overall Water Health</h3>
          <p className="text-sm text-gray-500 mb-6">Based on pollution thresholds</p>
          
          <div className="flex-1 min-h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={statusDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {statusDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend verticalAlign="bottom" height={36}/>
              </PieChart>
            </ResponsiveContainer>
          </div>
          
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
             <div className="text-sm text-gray-600">
                <strong>Insight:</strong> {((lakeData.filter(l => l.status === 'Good').length / lakeData.length || 0) * 100).toFixed(0)}% of water bodies are safe for general use.
             </div>
          </div>
        </div>
      </div>

      {/* Pollution Watchlist */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <div className="flex items-center gap-2 mb-6">
              <AlertTriangle className="text-red-500" size={20} />
              <h3 className="text-lg font-bold text-gray-900">Priority Attention Required</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {worstLakes.map((lake, idx) => (
                  <div 
                      key={lake.id} 
                      onClick={() => navigate(`/dashboard/lake/${encodeURIComponent(lake.name.toLowerCase())}`)}
                      className="flex items-center justify-between p-4 bg-red-50 rounded-lg cursor-pointer hover:bg-red-100 transition-colors"
                  >
                      <div className="flex items-center gap-4">
                          <span className="font-bold text-red-300 text-xl">#{idx + 1}</span>
                          <div>
                              <h4 className="font-semibold text-gray-900">{lake.name}</h4>
                              <p className="text-xs text-gray-500">{lake.location}</p>
                          </div>
                      </div>
                      <div className="text-right">
                          <span className="block font-bold text-red-700">{lake.turbidity.toFixed(5)} NTU</span>
                          <StatusBadge status="Critical" />
                      </div>
                  </div>
              ))}
          </div>
      </div>
    </div>
  );
}
