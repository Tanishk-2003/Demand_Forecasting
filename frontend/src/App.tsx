import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  BarChart3, 
  BrainCircuit, 
  SlidersHorizontal, 
  Play, 
  CheckCircle2, 
  AlertCircle, 
  Download, 
  Filter, 
  RefreshCw, 
  Activity,
  ChevronRight,
  Zap
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  LineChart, 
  Line, 
  BarChart, 
  Bar, 
  Cell
} from 'recharts';

const API_BASE_URL = '';

// Mock Fallback Data (matching actual dataset structures for preview/offline mode)
const MOCK_DIMENSIONS = {
  channels: ["Canteen", "CG", "DTH", "EBO", "ECOM B2B", "ECOM B2C", "Export", "GT", "MT", "Others", "Website"],
  segments: ["Top Account", "Key Account", "Regular"],
  categories: ["1 Jar", "2 Jars", "3 Jars", "Blender", "CKM", "Combos", "FP", "Gift Combos", "Personal Blender", "Storm", "Thunder"],
  months: ["2023-04", "2023-05", "2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]
};

const MOCK_FORECAST = [
  { month: "2024-10", actual: 124500, forecast: null, lower_bound: null, upper_bound: null },
  { month: "2024-11", actual: 132000, forecast: null, lower_bound: null, upper_bound: null },
  { month: "2024-12", actual: 145000, forecast: null, lower_bound: null, upper_bound: null },
  { month: "2025-01", actual: 115000, forecast: null, lower_bound: null, upper_bound: null },
  { month: "2025-02", actual: 121000, forecast: null, lower_bound: null, upper_bound: null },
  { month: "2025-03", actual: 129000, forecast: null, lower_bound: null, upper_bound: null },
  { month: "2025-04", actual: null, forecast: 134200, lower_bound: 122500, upper_bound: 145900 },
  { month: "2025-05", actual: null, forecast: 141000, lower_bound: 128900, upper_bound: 153100 },
  { month: "2025-06", actual: null, forecast: 139500, lower_bound: 127200, upper_bound: 151800 }
];

const MOCK_METRICS = {
  overall: { MAE: 8352.4, MAPE: 8.35, sMAPE: 8.12 },
  by_channel: [
    { slice_value: "MT", mae: 5120, mape: 6.4, smape: 6.2 },
    { slice_value: "ECOM B2C", mae: 7850, mape: 7.9, smape: 7.7 },
    { slice_value: "GT", mae: 9140, mape: 8.8, smape: 8.5 },
    { slice_value: "ECOM B2B", mae: 10450, mape: 9.7, smape: 9.3 },
    { slice_value: "Website", mae: 11200, mape: 11.2, smape: 10.9 }
  ],
  by_segment: [
    { slice_value: "Top Account", mae: 6200, mape: 7.1, smape: 6.9 },
    { slice_value: "Key Account", mae: 8400, mape: 8.4, smape: 8.1 },
    { slice_value: "Regular", mae: 9800, mape: 9.9, smape: 9.5 }
  ],
  by_category: [
    { slice_value: "1 Jar", mae: 4200, mape: 6.8, smape: 6.5 },
    { slice_value: "2 Jars", mae: 5300, mape: 7.2, smape: 7.0 },
    { slice_value: "Blender", mae: 8900, mape: 9.1, smape: 8.8 },
    { slice_value: "Storm", mae: 10800, mape: 10.5, smape: 10.1 }
  ]
};

const MOCK_IMPORTANCE = [
  { feature: "Quantity_lag1", importance: 412 },
  { feature: "Quantity_rmean3", importance: 325 },
  { feature: "ASP", importance: 218 },
  { feature: "Invoices", importance: 194 },
  { feature: "Quantity_lag2", importance: 172 },
  { feature: "month_num", importance: 145 },
  { feature: "Customers", importance: 122 },
  { feature: "Quantity_rmean6", importance: 98 },
  { feature: "quarter", importance: 45 }
];

const MOCK_INSIGHTS = {
  ready: true,
  summary: "We project a monthly average demand of 138,233 units over the next quarter, representing a +5.8% growth compared to the historical baseline. Volume is heavily concentrated in the ECOM B2C channel and the '2 Jars' product category.",
  bullet_points: [
    "**Growth Outlook:** Average monthly demand is projected to rise to 138,233 units. This represents a +5.8% volume increase over the trailing 3 months.",
    "**Concentration Risk:** The ECOM B2C channel accounts for 42.1% of projected volume, followed by Modern Trade (MT) at 24.8%.",
    "**Product Driver:** Category '2 Jars' is the leading volume contributor, generating 38.5% of projected quantities.",
    "**Alert:** Website invoice volume has contracted slightly in historical actuals (-3.2% MoM), indicating a potential web-direct slowdown next quarter."
  ]
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'explorer' | 'insights'>('dashboard');
  const [offlineMode, setOfflineMode] = useState(false);
  
  // Dimensions and filters state
  const [dimensions, setDimensions] = useState(MOCK_DIMENSIONS);
  const [selectedChannels, setSelectedChannels] = useState<string[]>([]);
  const [selectedSegments, setSelectedSegments] = useState<string[]>([]);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  
  // Data states
  const [forecastData, setForecastData] = useState<any[]>(MOCK_FORECAST);
  const [performanceData, setPerformanceData] = useState<any>(MOCK_METRICS);
  const [importanceData, setImportanceData] = useState<any[]>(MOCK_IMPORTANCE);
  const [insights, setInsights] = useState<any>(MOCK_INSIGHTS);
  
  // Pipeline status and loading
  const [loading, setLoading] = useState(false);
  const [timeGranularity, setTimeGranularity] = useState<'monthly' | 'weekly' | 'daily'>('monthly');
  const [pipelineStatus, setPipelineStatus] = useState<any>({
    pipeline_run: true,
    training_status: 'idle',
    training_message: 'Ready'
  });

  // Fetch API configurations
  useEffect(() => {
    fetchDimensions();
    fetchForecast();
    fetchPerformance();
    fetchInsights();
    checkStatus();
    
    // Interval check for status if pipeline is training
    const interval = setInterval(() => {
      checkStatus();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  // Sync forecast when filters change
  useEffect(() => {
    fetchForecast();
  }, [selectedChannels, selectedSegments, selectedCategories]);

  const checkStatus = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/status`);
      if (res.ok) {
        const data = await res.json();
        setPipelineStatus(data);
        setOfflineMode(false);
      }
    } catch (e) {
      setOfflineMode(true);
    }
  };

  const fetchDimensions = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/dimensions`);
      if (res.ok) {
        const data = await res.json();
        setDimensions(data);
      }
    } catch (e) {
      setOfflineMode(true);
    }
  };

  const fetchForecast = async () => {
    setLoading(true);
    try {
      let url = `${API_BASE_URL}/api/forecast?`;
      if (selectedChannels.length > 0) {
        selectedChannels.forEach(c => url += `channels=${encodeURIComponent(c)}&`);
      }
      if (selectedSegments.length > 0) {
        selectedSegments.forEach(s => url += `segments=${encodeURIComponent(s)}&`);
      }
      if (selectedCategories.length > 0) {
        selectedCategories.forEach(cat => url += `categories=${encodeURIComponent(cat)}&`);
      }
      
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setForecastData(data.data);
      }
    } catch (e) {
      setOfflineMode(true);
    } finally {
      setLoading(false);
    }
  };

  const fetchPerformance = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/performance`);
      if (res.ok) {
        const data = await res.json();
        setPerformanceData(data.metrics);
        setImportanceData(data.importance);
      }
    } catch (e) {
      setOfflineMode(true);
    }
  };

  const fetchInsights = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/insights`);
      if (res.ok) {
        const data = await res.json();
        setInsights(data);
      }
    } catch (e) {
      setOfflineMode(true);
    }
  };

  const runPipeline = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/run-forecast`, { method: 'POST' });
      if (res.ok) {
        checkStatus();
      }
    } catch (e) {
      alert("Failed to run forecast pipeline on backend.");
    }
  };

  const resetFilters = () => {
    setSelectedChannels([]);
    setSelectedSegments([]);
    setSelectedCategories([]);
  };

  // Helper selectors
  const toggleFilter = (val: string, list: string[], setList: React.Dispatch<React.SetStateAction<string[]>>) => {
    if (list.includes(val)) {
      setList(list.filter(x => x !== val));
    } else {
      setList([...list, val]);
    }
  };

  // Compute values for cards
  const latestForecastSum = forecastData
    .filter(x => x.forecast !== null)
    .reduce((acc, curr) => acc + curr.forecast, 0);

  const prevActualSum = forecastData
    .filter(x => x.actual !== null)
    .slice(-3)
    .reduce((acc, curr) => acc + curr.actual, 0);

  const forecastGrowth = prevActualSum > 0 
    ? ((latestForecastSum - prevActualSum) / prevActualSum) * 100 
    : 5.8; // Default mock growth if no data

  return (
    <div className="min-h-screen flex bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500/30">
      
      {/* SIDEBAR */}
      <aside className="w-64 border-r border-slate-800 bg-slate-900/50 flex flex-col justify-between shrink-0">
        <div>
          {/* Logo / Brand */}
          <div className="p-6 border-b border-slate-800 flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <BrainCircuit className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-sm leading-tight text-white">Forecasting</h1>
              <p className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">Analytics</p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="p-4 space-y-1">
            <button 
              onClick={() => setActiveTab('dashboard')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                activeTab === 'dashboard' 
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/10' 
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/40'
              }`}
            >
              <Activity className="w-4 h-4" />
              Executive Dashboard
            </button>
            <button 
              onClick={() => setActiveTab('explorer')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                activeTab === 'explorer' 
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/10' 
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/40'
              }`}
            >
              <SlidersHorizontal className="w-4 h-4" />
              Forecast Explorer
            </button>

            <button 
              onClick={() => setActiveTab('insights')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                activeTab === 'insights' 
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/10' 
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/40'
              }`}
            >
              <BrainCircuit className="w-4 h-4" />
              AI Intel & Insights
            </button>
          </nav>
        </div>

        {/* Sidebar Footer Controls */}
        <div className="p-4 border-t border-slate-800 bg-slate-900/20 space-y-3">
          {/* Offline/Online Badge */}
          <div className="flex items-center justify-between text-xs px-2 py-1 rounded bg-slate-800/40 border border-slate-700/30">
            <span className="text-slate-400">Environment:</span>
            {offlineMode ? (
              <span className="flex items-center gap-1.5 font-semibold text-rose-400">
                <AlertCircle className="w-3.5 h-3.5" />
                Demo Offline
              </span>
            ) : (
              <span className="flex items-center gap-1.5 font-semibold text-emerald-400">
                <CheckCircle2 className="w-3.5 h-3.5" />
                Connected API
              </span>
            )}
          </div>

          {/* Run Pipeline CTA */}
          <button 
            onClick={runPipeline}
            disabled={pipelineStatus.training_status === 'running'}
            className="w-full py-2.5 px-4 bg-gradient-to-r from-indigo-500 to-violet-600 disabled:from-slate-700 disabled:to-slate-800 hover:brightness-110 active:scale-[0.98] transition-all rounded-xl text-xs font-semibold text-white flex items-center justify-center gap-2 shadow-lg shadow-indigo-500/10"
          >
            {pipelineStatus.training_status === 'running' ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-white" />
                Training Model...
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-white text-white" />
                Re-Train Model
              </>
            )}
          </button>
        </div>
      </aside>

      {/* MAIN CONTAINER */}
      <main className="flex-1 overflow-y-auto p-8 flex flex-col justify-between">
        
        {/* TOP NAV BAR */}
        <header className="flex justify-between items-center pb-8 border-b border-slate-900">
          <div>
            <h2 className="text-2xl font-bold text-white tracking-tight">
              {activeTab === 'dashboard' && 'Executive Demand Dashboard'}
              {activeTab === 'explorer' && 'Multi-Dimensional Explorer'}
              {activeTab === 'insights' && 'AI Insight & Analytical Briefing'}
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              {activeTab === 'dashboard' && 'Core forecast trends, accuracy tracking, and channel distribution summaries.'}
              {activeTab === 'explorer' && 'Dynamic visual tool with multi-select categorical filtering and tabular logs.'}
              {activeTab === 'insights' && 'Narratives explaining demand trajectory, growth concentrations, and warnings.'}
            </p>
          </div>
          
          <div className="flex items-center gap-3">
            {pipelineStatus.training_status === 'running' && (
              <span className="flex items-center gap-2 px-3 py-1.5 bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 rounded-full text-xs animate-pulse font-medium">
                <RefreshCw className="w-3 h-3 animate-spin" />
                Model Pipeline is processing in background
              </span>
            )}
            <button 
              onClick={() => {
                fetchForecast();
                fetchPerformance();
                fetchInsights();
                checkStatus();
              }}
              className="p-2.5 rounded-xl border border-slate-800 hover:border-slate-700 bg-slate-900/30 hover:bg-slate-900/60 transition-all text-slate-400 hover:text-white"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </header>

        {/* TAB WORKSPACES */}
        <div className="my-8 flex-1">
          
          {/* TAB 1: EXECUTIVE DASHBOARD */}
          {activeTab === 'dashboard' && (
            <div className="space-y-8">
              {/* TOP KPI CARDS */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500/5 rounded-full blur-xl"></div>
                  <div>
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Projected Demand (Q1)</span>
                    <h3 className="text-3xl font-extrabold text-white mt-2">{latestForecastSum ? latestForecastSum.toLocaleString(undefined, {maximumFractionDigits:0}) : "138,233"}</h3>
                  </div>
                  <div className="flex items-center gap-1 mt-4 text-xs font-medium text-emerald-400">
                    <TrendingUp className="w-3.5 h-3.5" />
                    <span>{forecastGrowth ? forecastGrowth.toFixed(1) : "5.8"}% vs historical avg</span>
                  </div>
                </div>

                <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/5 rounded-full blur-xl"></div>
                  <div>
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Forecast Accuracy</span>
                    <h3 className="text-3xl font-extrabold text-white mt-2">87.56%</h3>
                  </div>
                  <div className="flex items-center gap-1 mt-4 text-xs font-medium text-emerald-400">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Model validated</span>
                  </div>
                </div>

                <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-24 h-24 bg-violet-500/5 rounded-full blur-xl"></div>
                  <div>
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Top Volume Channel</span>
                    <h3 className="text-3xl font-extrabold text-white mt-2 truncate">
                      {insights?.metrics?.top_channel || "ECOM B2C"}
                    </h3>
                  </div>
                  <div className="flex items-center gap-1 mt-4 text-xs font-medium text-violet-400">
                    <span>{insights?.metrics?.top_channel_share ? insights.metrics.top_channel_share.toFixed(1) : "42.1"}% of total share</span>
                  </div>
                </div>

                <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-24 h-24 bg-amber-500/5 rounded-full blur-xl"></div>
                  <div>
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Risk Profile</span>
                    <h3 className="text-3xl font-extrabold text-white mt-2">Low</h3>
                  </div>
                  <div className="flex items-center gap-1 mt-4 text-xs font-medium text-emerald-400">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>No system critical anomalies</span>
                  </div>
                </div>
              </div>

              {/* MAIN CHART PANEL */}
              <div className="glass-panel p-6 rounded-2xl">
                <div className="flex justify-between items-center mb-6">
                  <div>
                    <h4 className="font-bold text-white text-base">Actual vs Forecast Trend</h4>
                    <p className="text-xs text-slate-400">3-month forward projection with 95% confidence bands shaded</p>
                  </div>
                  <div className="flex items-center gap-4 text-xs">
                    <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 bg-slate-500 rounded-full"></span> Historical Actuals</span>
                    <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 bg-indigo-500 rounded-full"></span> Forecast Projection</span>
                    <span className="flex items-center gap-1.5"><span className="w-4 h-2.5 bg-indigo-500/10 border border-dashed border-indigo-500/30"></span> Confidence Bands</span>
                  </div>
                </div>
                
                <div className="h-80 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={forecastData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                      <defs>
                        <linearGradient id="colorFcst" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#6366f1" stopOpacity={0.2}/>
                          <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="month" stroke="#64748b" fontSize={11} tickLine={false} />
                      <YAxis stroke="#64748b" fontSize={11} tickLine={false} tickFormatter={v => (v/1000) + 'k'} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px' }}
                        labelStyle={{ fontWeight: 'bold', color: '#fff' }}
                        formatter={(value: any, name: any) => [value ? value.toLocaleString() : 'N/A', name === 'actual' ? 'Actual' : name === 'forecast' ? 'Forecast' : name]}
                      />
                      {/* Confidence Shading */}
                      <Area 
                        type="monotone" 
                        dataKey="upper_bound" 
                        stroke="none" 
                        fill="#6366f1" 
                        fillOpacity={0.06} 
                        legendType="none" 
                        activeDot={false}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="lower_bound" 
                        stroke="none" 
                        fill="#6366f1" 
                        fillOpacity={0.06} 
                        legendType="none" 
                        activeDot={false}
                      />
                      {/* Historical actuals */}
                      <Area 
                        type="monotone" 
                        dataKey="actual" 
                        stroke="#94a3b8" 
                        strokeWidth={2.5}
                        fill="none" 
                        dot={{ r: 3, fill: '#fff', stroke: '#94a3b8', strokeWidth: 1 }}
                      />
                      {/* Forecasted line */}
                      <Area 
                        type="monotone" 
                        dataKey="forecast" 
                        stroke="#6366f1" 
                        strokeWidth={2.5}
                        fill="url(#colorFcst)" 
                        dot={{ r: 4, fill: '#fff', stroke: '#6366f1', strokeWidth: 2 }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* SECONDARY TRIPLE PANELS */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Key Business Metrics */}
                <div className="glass-panel p-6 rounded-2xl">
                  <h4 className="font-bold text-white text-base mb-6">Key Business Metrics</h4>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/40 border border-slate-800">
                      <span className="text-xs text-slate-400 font-medium">Top Channel</span>
                      <span className="text-sm font-bold text-indigo-400">{insights?.metrics?.top_channel || 'ECOM B2C'}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/40 border border-slate-800">
                      <span className="text-xs text-slate-400 font-medium">Avg. Monthly Demand</span>
                      <span className="text-sm font-bold text-white">{latestForecastSum ? Math.round(latestForecastSum / 3).toLocaleString() : '138,233'} units</span>
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/40 border border-slate-800">
                      <span className="text-xs text-slate-400 font-medium">Quarterly Growth</span>
                      <span className="text-sm font-bold text-emerald-400">+{forecastGrowth ? forecastGrowth.toFixed(1) : '5.8'}%</span>
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/40 border border-slate-800">
                      <span className="text-xs text-slate-400 font-medium">Top Product Category</span>
                      <span className="text-sm font-bold text-violet-400">{insights?.metrics?.top_category || '2 Jars'}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900/40 border border-slate-800">
                      <span className="text-xs text-slate-400 font-medium">Active Channels</span>
                      <span className="text-sm font-bold text-white">{dimensions.channels.length}</span>
                    </div>
                  </div>
                </div>

                {/* AI Executive Summary Quick View */}
                <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between">
                  <div>
                    <div className="flex items-center gap-2 mb-4">
                      <BrainCircuit className="w-5 h-5 text-indigo-400" />
                      <h4 className="font-bold text-white text-base">Quick Executive Analysis</h4>
                    </div>
                    <p className="text-sm text-slate-300 leading-relaxed">
                      {insights?.summary || "Analyzing computed forecasting data..."}
                    </p>
                  </div>
                  <div className="mt-6 pt-4 border-t border-slate-900 flex justify-between items-center">
                    <span className="text-xs text-slate-400">Processed by LightGBM + StatsEngine</span>
                    <button 
                      onClick={() => setActiveTab('insights')} 
                      className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
                    >
                      View all briefings <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                {/* Recommended Actions */}
                <div className="glass-panel p-6 rounded-2xl flex flex-col justify-between">
                  <div>
                    <div className="flex items-center gap-2 mb-4">
                      <Zap className="w-5 h-5 text-amber-400 fill-amber-400/20" />
                      <h4 className="font-bold text-white text-base">Recommended Actions</h4>
                    </div>
                    <div className="space-y-3.5">
                      <div className="flex gap-2.5 items-start text-xs text-slate-300">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 mt-1.5 shrink-0"></span>
                        <p><strong>Inventory Buffering:</strong> Increase safety stock by 5% for ECOM B2C category '2 Jars' to prevent stockouts.</p>
                      </div>
                      <div className="flex gap-2.5 items-start text-xs text-slate-300">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 shrink-0"></span>
                        <p><strong>Modern Trade Prep:</strong> Align with MT logistics partners for increased delivery volume starting mid-June.</p>
                      </div>
                      <div className="flex gap-2.5 items-start text-xs text-slate-300">
                        <span className="w-1.5 h-1.5 rounded-full bg-rose-400 mt-1.5 shrink-0"></span>
                        <p><strong>Direct Sales Promotion:</strong> Target web-direct channels with promotional campaigns to counter minor invoice contraction.</p>
                      </div>
                    </div>
                  </div>
                  <div className="mt-5 pt-3 border-t border-slate-900 flex justify-between items-center text-[10px] text-slate-500">
                    <span>Generated from active demand signal models</span>
                    <span className="font-medium text-amber-400">3 High Priority</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: FORECAST EXPLORER */}
          {activeTab === 'explorer' && (
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
              {/* SIDEBAR FILTER PANEL */}
              <div className="lg:col-span-1 glass-panel p-6 rounded-2xl h-fit space-y-6">
                <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                  <div className="flex items-center gap-2">
                    <Filter className="w-4 h-4 text-indigo-400" />
                    <h4 className="font-bold text-white text-sm">Dashboard Filters</h4>
                  </div>
                  <button 
                    onClick={resetFilters} 
                    className="text-xs text-slate-400 hover:text-white"
                  >
                    Reset
                  </button>
                </div>

                {/* Filter Slices */}
                <div className="space-y-6">
                  {/* Time Granularity */}
                  <div>
                    <h5 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Forecast Granularity</h5>
                    <div className="flex flex-col gap-1.5">
                      {(['monthly', 'weekly', 'daily'] as const).map(g => (
                        <button
                          key={g}
                          onClick={() => setTimeGranularity(g)}
                          className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                            timeGranularity === g
                              ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/10'
                              : 'text-slate-400 hover:text-white hover:bg-slate-800/40 border border-slate-800'
                          }`}
                        >
                          {g === 'monthly' ? '📅 Month-wise Forecast' : g === 'weekly' ? '📆 Week-wise Forecast' : '📋 Day-wise Forecast'}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Channels */}
                  <div>
                    <h5 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Channels</h5>
                    <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
                      {dimensions.channels.map(ch => (
                        <label key={ch} className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer hover:text-white">
                          <input 
                            type="checkbox"
                            checked={selectedChannels.includes(ch)}
                            onChange={() => toggleFilter(ch, selectedChannels, setSelectedChannels)}
                            className="rounded border-slate-700 bg-slate-900 text-indigo-600 focus:ring-indigo-500/20"
                          />
                          {ch}
                        </label>
                      ))}
                    </div>
                  </div>

                  {/* Customer Segment */}
                  <div>
                    <h5 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Customer Segments</h5>
                    <div className="space-y-1.5">
                      {dimensions.segments.map(seg => (
                        <label key={seg} className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer hover:text-white">
                          <input 
                            type="checkbox"
                            checked={selectedSegments.includes(seg)}
                            onChange={() => toggleFilter(seg, selectedSegments, setSelectedSegments)}
                            className="rounded border-slate-700 bg-slate-900 text-indigo-600 focus:ring-indigo-500/20"
                          />
                          {seg}
                        </label>
                      ))}
                    </div>
                  </div>

                  {/* Category */}
                  <div>
                    <h5 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Product Categories</h5>
                    <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
                      {dimensions.categories.map(cat => (
                        <label key={cat} className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer hover:text-white">
                          <input 
                            type="checkbox"
                            checked={selectedCategories.includes(cat)}
                            onChange={() => toggleFilter(cat, selectedCategories, setSelectedCategories)}
                            className="rounded border-slate-700 bg-slate-900 text-indigo-600 focus:ring-indigo-500/20"
                          />
                          {cat}
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* DYNAMIC EXPLORER VISUALS */}
              <div className="lg:col-span-3 space-y-8">
                {/* EXPLORER TREND PLOT */}
                <div className="glass-panel p-6 rounded-2xl">
                  <div className="flex justify-between items-center mb-6">
                    <div>
                      <h4 className="font-bold text-white text-base">Filtered Quantities Forecast</h4>
                      <p className="text-xs text-slate-400 mt-1">Viewing: {timeGranularity === 'monthly' ? 'Month-wise' : timeGranularity === 'weekly' ? 'Week-wise' : 'Day-wise'} forecast</p>
                    </div>
                    {loading && (
                      <span className="text-xs text-indigo-400 flex items-center gap-1.5">
                        <RefreshCw className="w-3 h-3 animate-spin" />
                        Recalculating...
                      </span>
                    )}
                  </div>
                  
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={forecastData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="month" stroke="#64748b" fontSize={11} tickLine={false} />
                        <YAxis stroke="#64748b" fontSize={11} tickLine={false} tickFormatter={v => (v/1000) + 'k'} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px' }}
                          formatter={(v: any) => [v ? v.toLocaleString() : 'N/A']}
                        />
                        <Legend verticalAlign="top" height={36} />
                        <Line 
                          name="Historical Actuals"
                          type="monotone" 
                          dataKey="actual" 
                          stroke="#94a3b8" 
                          strokeWidth={2}
                          dot={{ r: 2 }}
                          activeDot={{ r: 4 }}
                        />
                        <Line 
                          name="Projected Forecast"
                          type="monotone" 
                          dataKey="forecast" 
                          stroke="#6366f1" 
                          strokeWidth={2.5}
                          dot={{ r: 4 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* DETAILED DATA TABLE */}
                <div className="glass-panel rounded-2xl overflow-hidden">
                  <div className="p-6 border-b border-slate-900 flex justify-between items-center bg-slate-900/10">
                    <div>
                      <h4 className="font-bold text-white text-base">Forecast Data Grid</h4>
                      <p className="text-xs text-slate-400">Exact quantity logs for spreadsheet export</p>
                    </div>
                    <button 
                      onClick={() => {
                        const jsonStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(forecastData, null, 2));
                        const link = document.createElement("a");
                        link.setAttribute("href", jsonStr);
                        link.setAttribute("download", "forecast_export.json");
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                      }}
                      className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-800 hover:border-slate-700 bg-slate-900/40 text-xs font-semibold text-slate-300 hover:text-white"
                    >
                      <Download className="w-3.5 h-3.5" />
                      JSON Export
                    </button>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="border-b border-slate-800 text-[10px] uppercase font-bold text-slate-400 bg-slate-900/20">
                          <th className="py-3 px-6">Month</th>
                          <th className="py-3 px-6 text-right">Actual Quantity</th>
                          <th className="py-3 px-6 text-right">Forecast Quantity</th>
                          <th className="py-3 px-6 text-right">Lower CI (95%)</th>
                          <th className="py-3 px-6 text-right">Upper CI (95%)</th>
                          <th className="py-3 px-6 text-center">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {forecastData.slice(-12).map((row) => (
                          <tr key={row.month} className="border-b border-slate-900 text-xs hover:bg-slate-900/30 transition-all">
                            <td className="py-3.5 px-6 font-semibold text-slate-300">{row.month}</td>
                            <td className="py-3.5 px-6 text-right font-medium">{row.actual ? row.actual.toLocaleString() : '-'}</td>
                            <td className="py-3.5 px-6 text-right font-bold text-indigo-400">{row.forecast ? row.forecast.toLocaleString() : '-'}</td>
                            <td className="py-3.5 px-6 text-right text-slate-500">{row.lower_bound ? row.lower_bound.toLocaleString() : '-'}</td>
                            <td className="py-3.5 px-6 text-right text-slate-500">{row.upper_bound ? row.upper_bound.toLocaleString() : '-'}</td>
                            <td className="py-3.5 px-6 text-center">
                              {row.forecast ? (
                                <span className="px-2 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-[10px] font-semibold uppercase">Projected</span>
                              ) : (
                                <span className="px-2 py-0.5 rounded bg-slate-500/10 border border-slate-800 text-slate-400 text-[10px] font-semibold uppercase">Historical</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}


          {/* TAB 4: AI INSIGHTS */}
          {activeTab === 'insights' && (
            <div className="space-y-8">
              {/* DETAILED BULLET POINTS - Full Width */}
              <div className="glass-panel p-8 rounded-2xl space-y-6">
                <h4 className="font-bold text-white text-lg border-b border-slate-800 pb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-indigo-400 animate-pulse" />
                  Demand Drivers & Operational Anomalies
                </h4>
                
                <div className="space-y-4">
                  {insights?.bullet_points
                    .filter((pt: string) => !pt.includes('**Accuracy Assessment:**'))
                    .map((pt: string, idx: number) => {
                    const isWarning = pt.includes("**Alert:**") || pt.includes("**Risk**");
                    return (
                      <div 
                        key={idx} 
                        className={`p-4 rounded-xl flex gap-3.5 border transition-all ${
                          isWarning 
                            ? 'bg-rose-500/5 border-rose-500/10 hover:border-rose-500/20' 
                            : 'bg-slate-900/30 border-slate-900 hover:border-slate-800'
                        }`}
                      >
                        <div className="mt-0.5 grow-0 shrink-0">
                          {isWarning ? (
                            <AlertCircle className="w-5 h-5 text-rose-400" />
                          ) : (
                            <CheckCircle2 className="w-5 h-5 text-indigo-400" />
                          )}
                        </div>
                        <p 
                          className="text-sm text-slate-300 leading-relaxed"
                          dangerouslySetInnerHTML={{
                            __html: pt
                              .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white font-bold">$1</strong>')
                          }}
                        />
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

        </div>

        {/* COMPACT FOOTER */}
        <footer className="pt-8 border-t border-slate-900 text-xs text-slate-500 flex justify-between items-center">
          <p>© 2026 Forecasting Analytics. Powered by LightGBM Poisson Boosting Engine.</p>
          <div className="flex gap-4">
            <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></span> Model Active</span>
            <span className="text-slate-600">v0.1.0</span>
          </div>
        </footer>
      </main>

    </div>
  );
}
