import React, { useState, useEffect } from 'react';
import { 
  User, 
  LogOut, 
  Upload, 
  Database, 
  Cpu, 
  History,
  Loader2, 
  Lock, 
  Mail, 
  CheckCircle2, 
  FileText,
  UserPlus,
  Download,
  Trash2,
  Eye
} from 'lucide-react';


const API_BASE_URL = "http://localhost:8000";

// --- 1. Sub-Components (Defined outside App to prevent focus loss) ---

const AuthView = ({ mode, setMode, handleAuth, authData, setAuthData, error, authLoading }) => (
  <div className="min-h-screen flex items-center justify-center bg-slate-50 px-4">
    <div className="max-w-md w-full bg-white rounded-3xl shadow-xl shadow-red-100 border border-red-50 p-10 text-left">
      <div className="text-center mb-10">
        <div className="bg-red-600 w-16 h-16 rounded-2xl flex items-center justify-center text-white mx-auto mb-4 shadow-lg shadow-red-200">
          <Cpu size={32} />
        </div>
        <h2 className="text-3xl font-extrabold text-slate-900">Vesuvius AI Pro</h2>
        <p className="text-slate-500 mt-2">
          {mode === 'login' ? 'Sign in to start surface analysis' : 'Create a new account to start'}
        </p>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 text-red-600 text-sm font-bold rounded-xl border border-red-100 animate-pulse">
          {error}
        </div>
      )}

      <form onSubmit={handleAuth} className="space-y-4">
        {mode === 'register' && (
          <div className="relative">
            <User className="absolute left-3 top-3.5 text-slate-400" size={18} />
            <input 
              type="text" 
              placeholder="Full Name" 
              required
              value={authData.name}
              className="w-full pl-10 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-500 transition-all"
              onChange={(e) => setAuthData({...authData, name: e.target.value})}
            />
          </div>
        )}
        <div className="relative">
          <Mail className="absolute left-3 top-3.5 text-slate-400" size={18} />
          <input 
            type="email" 
            placeholder="Email Address" 
            required
            value={authData.email}
            className="w-full pl-10 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-500 transition-all"
            onChange={(e) => setAuthData({...authData, email: e.target.value})}
          />
        </div>
        <div className="relative">
          <Lock className="absolute left-3 top-3.5 text-slate-400" size={18} />
          <input 
            type="password" 
            placeholder="Password" 
            required
            value={authData.password}
            className="w-full pl-10 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-500 transition-all"
            onChange={(e) => setAuthData({...authData, password: e.target.value})}
          />
        </div>
        <button 
          type="submit"
          disabled={authLoading}
          className="w-full bg-red-600 text-white py-4 rounded-xl font-bold hover:bg-red-700 transition-all shadow-lg shadow-red-200 active:scale-95 flex items-center justify-center gap-2"
        >
          {authLoading ? <Loader2 className="animate-spin" size={18} /> : (mode === 'login' ? 'Login Now' : 'Create Account')}
        </button>
      </form>

      <div className="mt-8 text-center border-t pt-6">
        <button 
          onClick={() => setMode(mode === 'login' ? 'register' : 'login')} 
          className="text-sm font-bold text-red-600 hover:underline flex items-center justify-center gap-2 mx-auto"
        >
          {mode === 'login' ? <><UserPlus size={16}/> New here? Register</> : <><Lock size={16}/> Already have account? Login</>}
        </button>
      </div>
    </div>
  </div>
);

const VisualOutput = ({ prediction }) => (
  <div className="mt-10 space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700 text-left">
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      <div className="space-y-4">
          <h3 className="text-xl font-bold text-slate-800 text-left">Original CT Scan Slice</h3>
          <div className="aspect-square bg-black rounded-lg overflow-hidden border-4 border-slate-900 relative shadow-2xl">
              <img 
                src={prediction.original_img} 
                alt="Raw Scan" 
                className="w-full h-full object-contain" 
              />
          </div>
      </div>

      <div className="space-y-4">
          <h3 className="text-xl font-bold text-slate-800 text-left">AI Detected Surface (Overlay)</h3>
          <div className="aspect-square bg-black rounded-lg overflow-hidden border-4 border-slate-900 relative shadow-2xl">
              <img 
                src={prediction.overlay_img} 
                alt="AI Overlay" 
                className="w-full h-full object-contain"
              />
          </div>
      </div>
    </div>

    {/* Result Bar showing exact pixels from model output */}
    <div className="bg-[#e9f2ff] p-4 rounded-xl border border-blue-100 shadow-sm flex items-center gap-2">
        <CheckCircle2 className="text-[#3b82f6]" size={18} />
        <p className="text-[#3b82f6] font-bold text-sm tracking-tight text-left">
            Detected Surface Pixels: {prediction.surface_pixels?.toLocaleString() || 0}
        </p>
    </div>
  </div>
);

const DashboardContent = ({ uploading, handleFileUpload, prediction, historyData, user }) => (
  <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 text-left">
    <div className="lg:col-span-2 space-y-6">
      <div className="bg-white rounded-3xl border border-slate-200 p-8 shadow-sm text-left">
        <div className="flex items-center justify-between mb-8">
          <h3 className="text-xl font-bold flex items-center gap-2 text-slate-800">
            <Upload size={24} className="text-red-600" />
            Analyze Volumetric Slice
          </h3>
          <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">TIFF Slices Only</span>
        </div>
        
        <label className="relative border-2 border-dashed border-red-100 rounded-3xl p-16 flex flex-col items-center justify-center bg-red-50/30 hover:bg-red-50 transition-all cursor-pointer group">
          <input type="file" className="hidden" accept=".tif,.tiff" onChange={handleFileUpload} disabled={uploading} />
          {uploading ? (
            <div className="text-center">
              <Loader2 className="animate-spin text-red-600 mx-auto mb-4" size={56} />
              <p className="text-xl font-bold text-slate-700">Predicting Surface...</p>
              <p className="text-slate-500 text-sm mt-2 italic text-center">Processing SurfaceNet via PyTorch</p>
            </div>
          ) : (
            <>
              <div className="bg-red-600 p-5 rounded-full text-white mb-6 shadow-lg group-hover:scale-110 transition-transform"><Upload size={36} /></div>
              <p className="text-slate-700 text-lg font-bold text-center">Select Dataset Slice</p>
              <p className="text-slate-400 mt-2 text-center">Click to browse or drag & drop</p>
            </>
          )}
        </label>

        {prediction && (
          <div className="mt-8 p-6 bg-white border-2 border-red-600 rounded-2xl shadow-md relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4">
              <CheckCircle2 className="text-red-600" size={32} />
            </div>
            <h4 className="text-red-600 text-xs font-black uppercase mb-2 tracking-widest">Analysis Complete</h4>
            <h2 className="text-3xl font-black text-slate-900 mb-6 truncate">{prediction.file_name}</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
              <div className="bg-red-50 p-4 rounded-xl text-center">
                <p className="text-red-400 text-xs font-bold mb-1 uppercase tracking-tighter">Surface Area</p>
                <p className="text-2xl font-black text-red-700">{prediction.surface_pixels?.toLocaleString() || 0}</p>
              </div>
              <div className="bg-red-50 p-4 rounded-xl text-center">
                <p className="text-red-400 text-xs font-bold mb-1 uppercase tracking-tighter">Model Score</p>
                <p className="text-2xl font-black text-red-700">{prediction.confidence}</p>
              </div>
              <div className="bg-red-50 p-4 rounded-xl text-center">
                <p className="text-red-400 text-xs font-bold mb-1 uppercase tracking-tighter">Scan Depth</p>
                <p className="text-2xl font-black text-red-700">3-Slice Envel.</p>
              </div>
            </div>

            <VisualOutput prediction={prediction} />
          </div>
        )}
      </div>
    </div>

    <div className="space-y-6">
      <div className="bg-red-600 rounded-3xl p-8 text-white shadow-xl shadow-red-200 text-left">
        <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-white/20 rounded-lg"><User size={20} /></div>
            <p className="font-bold text-lg">{user.name}</p>
        </div>
        <p className="text-red-100 text-sm leading-relaxed mb-6">
          Account authenticated via PostgreSQL. All scan history is securely stored on your local server.
        </p>
        <div className="flex justify-between items-end border-t border-red-500 pt-6">
          <div>
            <p className="text-[10px] uppercase font-bold text-red-300 tracking-tighter">Database Records</p>
            <p className="text-3xl font-black">{historyData.length}</p>
          </div>
          <div className="text-right">
            <p className="text-[10px] uppercase font-bold text-red-300 tracking-tighter">SQL Status</p>
            <p className="text-sm font-bold italic">CONNECTED</p>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-3xl border border-slate-200 p-6 shadow-sm text-left">
        <h4 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
          <History size={18} className="text-red-600" /> Recent History
        </h4>
        <div className="space-y-4">
          {historyData.length === 0 ? (
            <p className="text-slate-400 text-sm italic text-center py-4">No records found.</p>
          ) : (
            historyData.slice(0, 3).map(item => (
              <div key={item.id || Math.random()} className="flex items-center gap-3 p-3 bg-slate-50 rounded-xl hover:bg-red-50 transition-colors">
                <div className="bg-red-100 p-2 rounded-lg text-red-600"><FileText size={16} /></div>
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-bold truncate text-slate-700">{item.file_name}</p>
                  <p className="text-[10px] text-slate-400">{new Date(item.timestamp).toLocaleDateString()}</p>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  </div>
);

// --- 2. Main App Component ---

export default function App() {
  const [view, setView] = useState('dashboard');
  const [authMode, setAuthMode] = useState('login');
  const [user, setUser] = useState(null);
  const [authData, setAuthData] = useState({ name: '', email: '', password: '' });
  const [authLoading, setAuthLoading] = useState(false);
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [historyData, setHistoryData] = useState([]);

  // Chip Favicon and Title logic
  useEffect(() => {
    const link = document.querySelector("link[rel*='icon']") || document.createElement('link');
    link.type = 'image/svg+xml';
    link.rel = 'icon';
    link.href = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23dc2626" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="15" x2="23" y2="15"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="15" x2="4" y2="15"/></svg>';
    document.getElementsByTagName('head')[0].appendChild(link);
    document.title = "Vesuvius AI Pro";
  }, []);

  // Sync session
  useEffect(() => {
    const savedUser = localStorage.getItem('vesuvius_user');
    if (savedUser) setUser(JSON.parse(savedUser));
  }, []);

  // Sync History from DB
  useEffect(() => {
    if (!user) return;
    fetch(`${API_BASE_URL}/get-history/${user.uid}`)
      .then(res => res.json())
      .then(data => setHistoryData(data.reverse()))
      .catch(err => console.error("Database sync error:", err));
  }, [user]);

  const handleAuth = async (e) => {
    e.preventDefault();
    setError(''); setAuthLoading(true);
    const endpoint = authMode === 'login' ? '/login' : '/register';
    const payload = authMode === 'login' 
        ? { email: authData.email, password: authData.password }
        : { full_name: authData.name, email: authData.email, password: authData.password };

    try {
      const res = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (res.ok) {
        if (authMode === 'login') {
          setUser(data.user);
          localStorage.setItem('vesuvius_user', JSON.stringify(data.user));
        } else {
          setAuthMode('login'); alert("Registration Successful! Please login.");
        }
      } else { setError(data.detail || "Authentication Failed"); }
    } catch { setError("Cannot connect to backend server"); } finally { setAuthLoading(false); }
  };

  const handleLogout = () => {
    localStorage.removeItem('vesuvius_user');
    setUser(null);
    setAuthMode('login');
    setPrediction(null);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file || !user) return;
    setUploading(true); setPrediction(null);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', user.uid);

    try {
      const res = await fetch(`${API_BASE_URL}/predict`, { 
        method: 'POST', 
        body: formData 
      });
      if (res.ok) {
        const result = await res.json();
        // ENSURE REAL PIXEL COUNT FROM BACKEND IS SET
        setPrediction(result);
        const newHist = { ...result, timestamp: new Date().toISOString() };
        setHistoryData(prev => [newHist, ...prev]);
      } else {
        const errData = await res.json();
        alert(`Error: ${errData.detail}`);
      }
    } catch { alert("Backend offline."); } finally { setUploading(false); }
  };

  const handleDeleteHistory = async (id) => {
    if (!window.confirm("Delete this analysis record?")) return;
    try {
      const res = await fetch(`${API_BASE_URL}/delete-history/${id}`, { method: 'DELETE' });
      if (res.ok) {
        setHistoryData(prev => prev.filter(item => item.id !== id));
      } else { alert("Could not delete from database."); }
    } catch { alert("Server error."); }
  };

  const downloadImage = (base64Data, filename) => {
    const link = document.createElement('a');
    link.href = base64Data;
    link.download = `VESUVIUS_${filename.split('.')[0]}_RESULT.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (!user) return <AuthView mode={authMode} setMode={setAuthMode} handleAuth={handleAuth} authData={authData} setAuthData={setAuthData} error={error} authLoading={authLoading} />;

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 pb-20">
      {/* Navigation matching branding snippet */}
      <nav className="bg-white border-b border-slate-200 px-8 py-5 flex justify-between items-center sticky top-0 z-50 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="bg-red-600 p-2.5 rounded-xl text-white shadow-lg shadow-red-100">
            <Cpu size={26} />
          </div>
          <h1 className="text-2xl font-black tracking-tighter text-slate-900 uppercase">
            VESUVIUS <span className="text-red-600">web</span>
          </h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <button onClick={() => setView('dashboard')} className={`text-sm font-bold px-4 py-2 rounded-lg transition-all ${view === 'dashboard' ? 'bg-red-50 text-red-600' : 'text-slate-500 hover:text-red-600'}`}>Analyzer</button>
            <button onClick={() => setView('history')} className={`text-sm font-bold px-4 py-2 rounded-lg transition-all ${view === 'history' ? 'bg-red-50 text-red-600' : 'text-slate-500 hover:text-red-600'}`}>History</button>
          </div>
          <div className="flex items-center gap-3 bg-slate-100 p-1 pr-4 rounded-full border border-slate-200 shadow-sm">
            <div className="w-10 h-10 bg-red-600 rounded-full flex items-center justify-center text-white font-black text-sm uppercase shadow-md border-2 border-white">
              {user.name[0]}
            </div>
            <div className="text-left leading-none hidden sm:block">
              <p className="text-xs font-black text-slate-800 uppercase">{user.name}</p>
              <p className="text-[10px] text-slate-400 font-bold uppercase">{user.uid}</p>
            </div>
            <button onClick={handleLogout} className="ml-2 p-1 text-slate-400 hover:text-red-600 transition-colors"><LogOut size={16} /></button>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto p-8 pt-12">
        {view === 'dashboard' ? (
          <DashboardContent uploading={uploading} handleFileUpload={handleFileUpload} prediction={prediction} historyData={historyData} user={user} />
        ) : (
          <div className="bg-white rounded-3xl border border-slate-200 p-10 shadow-sm min-h-[500px] text-left">
             <div className="flex items-center gap-4 mb-10"><div className="bg-red-600 p-3 rounded-2xl text-white"><History size={32} /></div><h3 className="text-3xl font-black text-slate-900 tracking-tighter uppercase">Analysis History</h3></div>
             {historyData.length === 0 ? <p className="text-slate-400 italic py-20 text-center">No records found.</p> : (
               <div className="overflow-hidden rounded-2xl border border-slate-100">
                 <table className="w-full text-left border-collapse text-left">
                   <thead className="bg-slate-50 border-b border-slate-100"><tr className="text-slate-400 text-[10px] uppercase font-black tracking-widest"><th className="px-6 py-5">Scan Document</th><th className="px-6 py-5">Date</th><th className="px-6 py-5">Model Score</th><th className="px-6 py-5 text-right">Result</th><th className="px-6 py-5 text-right">Actions</th></tr></thead>
                   <tbody className="divide-y divide-slate-50">
                     {historyData.map((item) => (
                       <tr key={item.id} className="hover:bg-red-50/20 transition-colors group">
                         <td className="px-6 py-5 font-bold text-slate-800">{item.file_name}</td>
                         <td className="px-6 py-5 text-sm text-slate-500">{new Date(item.timestamp).toLocaleDateString()}</td>
                         <td className="px-6 py-5"><span className="bg-red-100 text-red-700 px-3 py-1 rounded-full text-[10px] font-black tracking-widest">{item.confidence}</span></td>
                         <td className="px-6 py-5 text-right font-black text-red-600 whitespace-nowrap">{item.surface_pixels?.toLocaleString() || 0} Pixels</td>
                         <td className="px-6 py-5 text-right"><div className="flex items-center justify-end gap-2">
                           {item.overlay_img && <button onClick={() => downloadImage(item.overlay_img, item.file_name)} className="p-2 bg-slate-100 text-slate-500 hover:bg-red-600 hover:text-white rounded-xl shadow-sm"><Download size={14} /></button>}
                           <button onClick={() => handleDeleteHistory(item.id)} className="p-2 bg-slate-100 text-slate-500 hover:bg-red-600 hover:text-white rounded-xl shadow-sm"><Trash2 size={14} /></button>
                         </div></td>
                       </tr>
                     ))}
                   </tbody>
                 </table>
               </div>
             )}
          </div>
        )}
      </main>

      <footer className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-white/80 backdrop-blur-md px-10 py-3 rounded-2xl border border-slate-200 shadow-xl z-50">
        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">CIS 6005 COMPUTATIONAL INTELLIGENCE PROJECT © 2026 | POSTGRESQL CONNECTED</p>
      </footer>
    </div>
  );
}