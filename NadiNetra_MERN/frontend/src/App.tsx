import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "./contexts/ThemeContext";
import { AuthProvider } from "./contexts/AuthContext";
import LandingPage from "./pages/LandingPage";
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";
import HomePage from "./pages/HomePage";
import LakePage from "./pages/LakePage";
import MapView from "./pages/MapView";
import WaterBodiesList from "./pages/WaterBodiesList";
import Settings from "./pages/Settings";
import Help from "./pages/Help";
import ContactPage from "./pages/ContactPage";

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <Router>
          <Routes>
            {/* Public Routes */}
            <Route path="/" element={<LandingPage />} />
            <Route path="/contact" element={<ContactPage/>} />
            
            {/* Protected Routes - Dashboard */}
            <Route
              path="/dashboard/*"
              element={
                <div className="min-h-screen bg-[#F3F0EA] flex flex-col md:flex-row font-sans">
                  {/* Sidebar Fixed */}
                  <div className="hidden md:block w-72 fixed h-full z-40">
                    <Sidebar />
                  </div>

                  {/* Main Content Area */}
                  <div className="flex-1 md:ml-72 flex flex-col min-h-screen relative">
                    {/* Floating Navbar */}
                    <div className="sticky top-0 z-30 px-6 pt-6 pb-2">
                       <Navbar />
                    </div>

                    <main className="flex-1 p-6 overflow-y-auto">
                      <Routes>
                        <Route path="/" element={<HomePage />} />
                        <Route path="lake/:name" element={<LakePage />} />
                        <Route path="map" element={<MapView />} />
                        <Route path="water-bodies" element={<WaterBodiesList />} />
                        <Route path="settings" element={<Settings />} />
                        <Route path="help" element={<Help />} />
                      </Routes>
                    </main>
                  </div>
                </div>
              }
            />
          </Routes>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;