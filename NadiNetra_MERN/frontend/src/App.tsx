import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "./contexts/ThemeContext";
import { AuthProvider } from "./contexts/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import LandingPage from "./pages/LandingPage";
import AuthPage from "./pages/AuthPage";
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";
import HomePage from "./pages/HomePage";
import LakePage from "./pages/LakePage";
import MapView from "./pages/MapView";
import WaterBodiesList from "./pages/WaterBodiesList";
import Analytics from "./pages/Analytics";
import Settings from "./pages/Settings";
import Help from "./pages/Help";
import CriticalStatus from "./pages/CriticalStatus";
import ContactPage from "./pages/ContactPage";

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <Router>
          <Routes>
            {/* Public Routes */}
            <Route path="/" element={<LandingPage />} />
            {/* <Route path="/auth" element={<AuthPage />} /> */}
            <Route path="/contact" element={<ContactPage/>} />
            
            {/* Protected Routes - Dashboard */}
            <Route
              path="/dashboard/*"
              element={
                // <ProtectedRoute>
                  <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
                    <Navbar />
                    <Sidebar />
                    <main className="ml-64 pt-16 min-h-screen">
                      <div className="p-8">
                        <Routes>
                          <Route path="/" element={<HomePage />} />
                          <Route path="lake/:name" element={<LakePage />} />
                          <Route path="map" element={<MapView />} />
                          <Route path="water-bodies" element={<WaterBodiesList />} />
                          {/* <Route path="analytics" element={<Analytics />} /> */}
                          <Route path="settings" element={<Settings />} />
                          <Route path="help" element={<Help />} />
                          {/* <Route path="critical-status" element={<CriticalStatus />} /> */}
                        </Routes>
                      </div>
                    </main>
                  </div>
                // </ProtectedRoute>
              }
            />
          </Routes>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
