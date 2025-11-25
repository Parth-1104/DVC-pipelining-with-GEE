"use client";
import React, { useState, useRef } from "react";
import { 
  ArrowRight, 
  ArrowLeft, 
  Globe, 
  Leaf, 
  Zap,       
  Crosshair, 
  Sprout,
  Cpu,          // Added for AI section
  Settings,     // Added for Processing
  Database,     // Added for Data
  BarChart3,    // Added for Charts
  Layers        // Added for Feature Extraction
} from "lucide-react"; 
import {
  Navbar,
  NavBody,
  NavItems,
  MobileNav,
  MobileNavHeader,
  MobileNavToggle,
  MobileNavMenu,
  NavbarLogo,
  NavbarButton,
} from "../components/ui/resizable-navbar";
import HydroTransNetCard from "../components/Hydro";

const LandingPage = () => {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0); // State for the slider

  const navItems = [
    { name: "Home", link: "/" },
    { name: "How it Works", link: "#how-it-works" },
    { name: "Contact", link: "/contact" },
  ];

  // Data for the "Harnessing Satellite Data" slider
  const features = [
    { 
      id: "01",
      title: "Coverage at scale", 
      desc: "Instantly visualize water clarity levels across vast river networks. Our transformer models detect suspended particles with 94% accuracy compared to ground sensors.",
      img: "https://images.unsplash.com/photo-1524661135-423995f22d0b?q=80&w=800&auto=format&fit=crop" 
    },
    { 
      id: "02",
      title: "Real-time Dashboard",
      desc: "Monitor algae growth in real-time. We use spectral bands to isolate chlorophyll concentrations, helping you predict blooms before they become toxic.",
      img: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=800&auto=format&fit=crop" 
    },
    { 
      id: "03",
      title: "Historical Trends", 
      desc: "Access 5 years of historical satellite data to understand seasonal patterns and long-term degradation trends in your water bodies.",
      img: "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=800&auto=format&fit=crop" 
    },
  ];

  // Slider Handlers
  const handleNext = () => setActiveIndex((prev) => (prev + 1) % features.length);
  const handlePrev = () => setActiveIndex((prev) => (prev - 1 + features.length) % features.length);

  // Data for the new "How It Works" Section
  const workflowSteps = [
    {
      step: "01",
      phase: "Phase 1-2: Acquisition",
      title: "Satellite Data Ingestion",
      description: "User defines the Region of Interest (ROI). The system instantly queries the Sentinel-2 ImageCollection via Google Earth Engine (GEE), filtering for cloud cover (<60%) and selecting specific spectral bands (B2, B3, B4, B8).",
      icon: <Database className="text-[#84cc16]" size={28} />,
      techSpecs: ["Sentinel-2 L1C", "Google Earth Engine", "Cloud Masking"]
    },
    {
      step: "02",
      phase: "Phase 3-4: Preprocessing",
      title: "Calibration & Feature Extraction",
      description: "Raw data undergoes Sen2Cor atmospheric correction to produce Bottom-Of-Atmosphere (BOA) reflectance. We then calculate critical spectral indices (NDVI, NDWI) and assemble a validated feature matrix.",
      icon: <Layers className="text-[#84cc16]" size={28} />,
      techSpecs: ["Radiometric Calibration", "Sen2Cor", "NDVI / NDWI Calculation"]
    },
    {
      step: "03",
      phase: "Phase 8: The Core",
      title: "HydroTransNet Architecture",
      description: "Our custom Transformer neural network processes temporal sequences of 10 timesteps. Using Multi-Head Self-Attention, it captures complex time-series dependencies that simple regression models miss.",
      icon: <Cpu className="text-[#84cc16]" size={28} />,
      techSpecs: ["Transformer Encoder", "Multi-Head Attention", "Temporal Sequences"]
    },
    {
      step: "04",
      phase: "Phase 9-10: Output",
      title: "Inference & Visualization",
      description: "The model predicts TSS (mg/L), Turbidity (NTU), and Chlorophyll (µg/L) levels. Results are post-processed and delivered via our real-time dashboard or API endpoints for immediate action.",
      icon: <BarChart3 className="text-[#84cc16]" size={28} />,
      techSpecs: ["FastAPI Inference", "Real-time Metrics", "JSON/CSV Export"]
    }
  ];

  return (
    <div className="w-full font-sans bg-gray-50">
      
      {/* ================= HERO SECTION ================= */}
      <div
        className="min-h-screen w-full bg-cover bg-center bg-no-repeat relative flex flex-col justify-between"
        style={{ backgroundImage: "url('/Farm.png')" }} 
      >
        {/* ---------------- NAVBAR ---------------- */}
        <div className="pt-6 px-4 md:px-12 w-full z-50 relative">
          <Navbar className="w-full max-w-7xl mx-auto">
            <NavBody className="py-3 px-6 text-sm bg-white rounded-full shadow-md flex justify-between items-center">
              <div className="flex items-center gap-2 text-green-700 font-bold text-xl">
                <div className="p-1 border-2 border-green-600 rounded">
                  <Leaf size={16} fill="currentColor" />
                </div>
                Nadi Netra
              </div>

              <div className="hidden md:block">
                <NavItems items={navItems} className="text-sm font-medium text-gray-700 hover:text-green-700 transition-colors" />
              </div>

              <div className="hidden md:flex items-center gap-3">
                <NavbarButton className="text-sm px-5 py-2 rounded-full bg-black text-white hover:bg-gray-800" href="/dashboard">
                  Dashboard
                </NavbarButton>
              </div>

              <div className="md:hidden">
                <MobileNavToggle isOpen={open} onClick={() => setOpen(!open)} />
              </div>
            </NavBody>

            <MobileNav>
              <MobileNavMenu isOpen={open} onClose={() => setOpen(false)} className="bg-white rounded-2xl mt-2 p-4">
                {navItems.map((item, idx) => (
                  <a key={idx} href={item.link} onClick={() => setOpen(false)} className="block text-gray-800 text-lg py-2 border-b border-gray-100">
                    {item.name}
                  </a>
                ))}
              </MobileNavMenu>
            </MobileNav>
          </Navbar>
        </div>

        {/* ---------------- MAIN HERO CONTENT ---------------- */}
        <main className="relative z-10 max-w-7xl mx-auto px-4 md:px-12 pt-12 pb-16 w-full flex flex-col justify-between grow">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-12 md:mb-8">
            <div className="max-w-2xl">
              <h1 className="text-6xl md:text-8xl font-medium text-white leading-[1.1] drop-shadow-sm">
                Part of future <br />
                Agriculture <span className="inline-block border-2 border-white rounded-full p-1 text-3xl align-middle mb-2"><Leaf size={32} /></span>
              </h1>
            </div>

            <div className="mt-8 md:mt-0 bg-[#84cc16] p-8 rounded-[2.5rem] text-white w-full md:w-72 shadow-xl relative group flex flex-col justify-between min-h-[300px]">
                <div className="flex justify-between items-start">
                    <div className="w-12 h-12 bg-[#1a2e1a] rounded-full flex items-center justify-center text-[#84cc16]">
                      <Globe size={24} />
                    </div>
                    <div className="flex items-center gap-2 bg-white/20 backdrop-blur-md px-3 py-1 rounded-full">
                      <span className="relative flex h-2 w-2">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-200 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-white"></span>
                      </span>
                      <span className="text-xs font-bold tracking-wide uppercase">Live Feed</span>
                    </div>
                </div>
                <div>
                    <h2 className="text-3xl font-bold mb-3 leading-none">Zero Lab <br /> Delays</h2>
                    <p className="text-sm opacity-90 leading-relaxed font-medium">Instant spectral analysis for river systems.</p>
                </div>
                <div>
                    <div className="flex justify-between items-end mb-2">
                    <span className="text-xs font-bold uppercase tracking-widest opacity-80">Remote Coverage</span>
                    <span className="text-xl font-bold">100%</span>
                    </div>
                    <div className="h-2 w-full bg-[#1a2e1a]/30 rounded-full overflow-hidden">
                    <div className="h-full bg-white w-full rounded-full shadow-[0_0_10px_rgba(255,255,255,0.5)]"></div>
                    </div>
                </div>
            </div>
          </div>

          <div className="flex flex-col md:flex-row gap-6 items-end">
            <div className="bg-[#F3F0EA] p-6 rounded-[2.5rem] flex-1 flex flex-col md:flex-row items-center relative group min-h-[300px]">
            <div className="w-full md:w-1/2 h-58 md:h-full flex items-center justify-center">
  <video 
    src="/videoplayback.mp4"
    className="object-cover w-full h-full drop-shadow-xl rounded-lg"
    autoPlay
    loop
    muted
  />
</div>

              <div className="w-full md:w-1/2 pl-4 flex flex-col justify-center">
                <h3 className="text-2xl font-bold text-gray-900 mb-2">Sentinal 2</h3>
                <p className="text-gray-600 text-sm mb-6 leading-relaxed">
                  Sentinel-2 satellite provides high-resolution, multispectral imagery used to monitor water quality parameters.
                </p>
                <a href="https://livingatlas.arcgis.com/sentinel2explorer" target="_blank" className="px-6 py-2 border border-gray-800 rounded-full text-sm text-white font-semibold bg-gray-800 hover:text-white transition self-start">
                  VIEW DETAILS
                </a>
              </div>
            </div>
            <HydroTransNetCard />
          </div>
        </main>
      </div>

      {/* ================= SLIDER SECTION ================= */}
      <section className="bg-white py-24 px-4 md:px-12 rounded-t-[3rem] -mt-10 relative z-20">
        <div className="max-w-7xl mx-auto">
          
          <div className="flex flex-col lg:flex-row justify-between items-end gap-10 mb-20">
            <h2 className="text-5xl md:text-7xl font-medium text-[#0f2518] leading-[1.1] tracking-tight">
              <span className="inline-flex items-center gap-2 align-middle mr-4 mb-2">
                 <span className="bg-[#F3F0EA] p-2 rounded-full"><Zap size={24} className="text-black" /></span>
                 <span className="bg-[#0f2518] p-2 rounded-full"><Crosshair size={24} className="text-white" /></span>
              </span>
              Harnessing <br />
              Satellite Data <span className="inline-block bg-[#F3F0EA] p-2 rounded-full align-middle mb-2"><Sprout size={32} className="text-[#84cc16]"/></span> <br />
              for Water Quality
            </h2>
          </div>

          <div className="flex flex-col lg:flex-row gap-12 items-stretch h-full min-h-[500px]">
            <div className="lg:w-1/3 flex flex-col justify-between order-2 lg:order-1">
               <div className="transition-all duration-500 ease-in-out">
                  <span className="text-8xl font-bold text-gray-100 block -ml-2 mb-4">
                     {features[activeIndex].id}
                  </span>
                  <h3 className="text-3xl font-bold text-[#0f2518] mb-4">
                     {features[activeIndex].title}
                  </h3>
                  <p className="text-gray-500 text-sm leading-7 font-medium mb-8">
                     {features[activeIndex].desc}
                  </p>
               </div>
               <div className="space-y-6">
                 <div className="flex items-center gap-2 text-sm font-bold text-gray-400">
                    <span>01</span>
                    <div className="w-full h-[2px] bg-gray-200 rounded-full relative">
                        <div 
                          className="absolute h-full bg-[#0f2518] transition-all duration-500 ease-out"
                          style={{ width: `${((activeIndex + 1) / features.length) * 100}%` }}
                        />
                    </div>
                    <span>03</span>
                 </div>
                 <div className="flex gap-4">
                    <button onClick={handlePrev} className="w-16 h-16 rounded-full border border-gray-200 flex items-center justify-center text-gray-600 hover:bg-[#0f2518] hover:text-white hover:border-[#0f2518] transition duration-300">
                      <ArrowLeft size={24}/>
                    </button>
                    <button onClick={handleNext} className="w-16 h-16 rounded-full border border-gray-200 flex items-center justify-center text-gray-600 hover:bg-[#0f2518] hover:text-white hover:border-[#0f2518] transition duration-300">
                      <ArrowRight size={24}/>
                    </button>
                 </div>
               </div>
            </div>
            <div className="lg:w-2/3 relative order-1 lg:order-2">
                <div className="w-full h-[500px] rounded-[2.5rem] overflow-hidden relative shadow-2xl z-20">
                    <img 
                      src={features[activeIndex].img} 
                      alt={features[activeIndex].title} 
                      className="w-full h-full object-cover transition-transform duration-700 hover:scale-105"
                      key={activeIndex} 
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-60" />
                    <div className="absolute top-8 right-8 bg-white/20 backdrop-blur-md text-white px-4 py-2 rounded-full text-xs font-bold uppercase tracking-wider border border-white/20">
                       Nadi Netra AI
                    </div>
                </div>
            </div>
          </div>
        </div>
      </section>

      {/* ================= HOW IT WORKS SECTION (NEW) ================= */}
      <section id="how-it-works" className="bg-[#0f2518] py-24 px-4 md:px-12 rounded-t-[3rem] -mt-10 relative z-30">
        <div className="max-w-7xl mx-auto">
          
          {/* Section Header */}
          <div className="text-center mb-20">
            <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur-md px-4 py-2 rounded-full text-xs font-bold uppercase tracking-widest text-[#84cc16] mb-6 border border-white/10">
              <Settings size={14} /> 
              System Workflow
            </div>
            <h2 className="text-4xl md:text-6xl font-medium text-white mb-6">
              How Nadi Netra Works
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto text-lg">
              From raw satellite ingestion to actionable insights, our pipeline leverages advanced Transformer models for unparalleled accuracy.
            </p>
          </div>

          {/* Workflow Steps Grid */}
          <div className="flex flex-col gap-8">
            {workflowSteps.map((item, idx) => (
              <div 
                key={idx} 
                className={`flex flex-col md:flex-row gap-8 items-center p-8 rounded-[2.5rem] border border-white/5 transition-colors hover:border-[#84cc16]/30 hover:bg-white/5 group ${
                  idx % 2 === 1 ? 'md:flex-row-reverse' : '' // Zig-zag layout
                }`}
              >
                {/* Visual Icon Box */}
                <div className="w-full md:w-1/3 flex justify-center">
                   <div className="w-32 h-32 rounded-full bg-[#1a2e1a] flex items-center justify-center border border-white/10 group-hover:scale-110 transition-transform duration-500 relative">
                      {/* Decorative ring */}
                      <div className="absolute inset-0 rounded-full border border-[#84cc16] opacity-0 group-hover:opacity-100 group-hover:animate-ping transition-opacity" />
                      {item.icon}
                   </div>
                </div>

                {/* Content */}
                <div className="w-full md:w-2/3 text-center md:text-left">
                   <div className="text-[#84cc16] text-sm font-bold uppercase tracking-widest mb-2">
                      {item.phase}
                   </div>
                   <h3 className="text-3xl font-bold text-white mb-4">
                      {item.step}. {item.title}
                   </h3>
                   <p className="text-gray-400 text-base leading-relaxed mb-6 max-w-xl mx-auto md:mx-0">
                      {item.description}
                   </p>
                   
                   {/* Tech Specs Tags */}
                   <div className="flex flex-wrap gap-2 justify-center md:justify-start">
                      {item.techSpecs.map((spec, sIdx) => (
                        <span key={sIdx} className="px-3 py-1 rounded-full bg-white/5 text-gray-300 text-xs font-medium border border-white/5">
                           {spec}
                        </span>
                      ))}
                   </div>
                </div>
              </div>
            ))}
          </div>

          {/* Call to Action Footer of Section */}
          <div className="mt-20 text-center">
            <button className="bg-[#84cc16] text-[#0f2518] rounded-full px-10 py-4 text-sm font-bold uppercase tracking-widest hover:bg-white transition-colors duration-300 inline-flex items-center gap-2">
               Start Monitoring <ArrowRight size={18}/>
            </button>
          </div>

        </div>
      </section>

    </div>
  );
};

export default LandingPage;