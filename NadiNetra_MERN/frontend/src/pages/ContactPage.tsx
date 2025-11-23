"use client";
import React, { useState } from "react";
import { 
  Leaf, 
  Mail, 
  Phone, 
  MapPin, 
  ArrowRight, 
  Send,
  MessageSquare,
  Globe
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
} from "../components/ui/resizable-navbar"; // Adjust path if needed

const ContactPage = () => {
  const [open, setOpen] = useState(false);

  const navItems = [
    { name: "Home", link: "/" },
    { name: "How it Works", link: "#" },
    { name: "Contact", link: "/contact" },
  ];

  return (
    <div className="min-h-screen w-full bg-[#F3F0EA] font-sans flex flex-col">
      
      {/* ---------------- NAVBAR ---------------- */}
      {/* Reusing your exact Navbar structure for consistency */}
      <div className="pt-6 px-4 md:px-12 w-full z-50 relative">
        <Navbar className="w-full max-w-7xl mx-auto">
          <NavBody className="py-3 px-6 text-sm bg-white rounded-full shadow-sm flex justify-between items-center">
            {/* Logo */}
            <div className="flex items-center gap-2 text-green-700 font-bold text-xl">
              <div className="p-1 border-2 border-green-600 rounded">
                <Leaf size={16} fill="currentColor" />
              </div>
              Nadi Netra
            </div>

            {/* Desktop Nav */}
            <div className="hidden md:block">
              <NavItems items={navItems} className="text-sm font-medium text-gray-700 hover:text-green-700 transition-colors" />
            </div>

            {/* Right Side Button */}
            <div className="hidden md:flex items-center gap-3">
              <NavbarButton className="text-sm px-5 py-2 rounded-full bg-black text-white hover:bg-gray-800" href="/dashboard">
                Dashboard
              </NavbarButton>
            </div>

            {/* Mobile Toggle */}
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

      {/* ---------------- MAIN CONTENT ---------------- */}
      <main className="flex-grow flex items-center justify-center px-4 md:px-12 py-12">
        <div className="w-full max-w-7xl mx-auto">
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch">
            
            {/* LEFT COLUMN: Contact Info Card (Dark Theme) */}
            <div className="bg-[#0f2518] rounded-[3rem] p-10 md:p-14 text-white flex flex-col justify-between relative overflow-hidden min-h-[600px]">
               
               {/* Background Pattern Decoration */}
               <div className="absolute top-0 right-0 w-64 h-64 bg-[#84cc16] rounded-full blur-[100px] opacity-20 -mr-16 -mt-16 pointer-events-none"></div>

               <div>
                  <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur-md px-4 py-2 rounded-full text-xs font-bold uppercase tracking-widest mb-8 border border-white/10">
                    <MessageSquare size={14} className="text-[#84cc16]" /> 
                    Contact Us
                  </div>
                  
                  <h1 className="text-5xl md:text-6xl font-medium leading-[1.1] mb-6">
                    Let's discuss <br />
                    <span className="text-[#84cc16]">Satellite Data.</span>
                  </h1>
                  
                  <p className="text-gray-400 text-lg max-w-sm leading-relaxed">
                    Have questions about HydroTransNet or Sentinel-2 integration? We are here to help you scale your water monitoring.
                  </p>
               </div>

               {/* Contact Details Grid */}
               <div className="space-y-8 mt-12 relative z-10">
                  <div className="flex items-start gap-5 group cursor-pointer">
                     <div className="w-14 h-14 rounded-full border border-gray-600 flex items-center justify-center group-hover:bg-[#84cc16] group-hover:text-[#0f2518] group-hover:border-[#84cc16] transition-all duration-300">
                        <Mail size={24} />
                     </div>
                     <div>
                        <p className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-1">Email Support</p>
                        <p className="text-xl font-medium">hello@nadinetra.ai</p>
                     </div>
                  </div>

                  <div className="flex items-start gap-5 group cursor-pointer">
                     <div className="w-14 h-14 rounded-full border border-gray-600 flex items-center justify-center group-hover:bg-[#84cc16] group-hover:text-[#0f2518] group-hover:border-[#84cc16] transition-all duration-300">
                        <Phone size={24} />
                     </div>
                     <div>
                        <p className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-1">Call Us</p>
                        <p className="text-xl font-medium">+91 (800) 123-4567</p>
                     </div>
                  </div>

                  <div className="flex items-start gap-5 group cursor-pointer">
                     <div className="w-14 h-14 rounded-full border border-gray-600 flex items-center justify-center group-hover:bg-[#84cc16] group-hover:text-[#0f2518] group-hover:border-[#84cc16] transition-all duration-300">
                        <MapPin size={24} />
                     </div>
                     <div>
                        <p className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-1">HQ Location</p>
                        <p className="text-xl font-medium">Tech Park, Bangalore, India</p>
                     </div>
                  </div>
               </div>

            </div>

            {/* RIGHT COLUMN: Interactive Form (White Theme) */}
            <div className="bg-white rounded-[3rem] p-10 md:p-14 shadow-xl flex flex-col justify-center">
              
              <h3 className="text-3xl font-bold text-[#0f2518] mb-8">Send a Message</h3>
              
              <form className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                   <div className="space-y-2">
                      <label className="text-sm font-bold text-gray-500 uppercase tracking-wider ml-2">First Name</label>
                      <input 
                        type="text" 
                        placeholder="John" 
                        className="w-full bg-[#F3F0EA] rounded-2xl px-6 py-4 text-[#0f2518] placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[#84cc16] transition-all"
                      />
                   </div>
                   <div className="space-y-2">
                      <label className="text-sm font-bold text-gray-500 uppercase tracking-wider ml-2">Last Name</label>
                      <input 
                        type="text" 
                        placeholder="Doe" 
                        className="w-full bg-[#F3F0EA] rounded-2xl px-6 py-4 text-[#0f2518] placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[#84cc16] transition-all"
                      />
                   </div>
                </div>

                <div className="space-y-2">
                    <label className="text-sm font-bold text-gray-500 uppercase tracking-wider ml-2">Email Address</label>
                    <input 
                      type="email" 
                      placeholder="john@company.com" 
                      className="w-full bg-[#F3F0EA] rounded-2xl px-6 py-4 text-[#0f2518] placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[#84cc16] transition-all"
                    />
                </div>

                <div className="space-y-2">
                    <label className="text-sm font-bold text-gray-500 uppercase tracking-wider ml-2">Organization Type</label>
                    <select className="w-full bg-[#F3F0EA] rounded-2xl px-6 py-4 text-[#0f2518] focus:outline-none focus:ring-2 focus:ring-[#84cc16] transition-all appearance-none cursor-pointer">
                       <option>Agricultural Enterprise</option>
                       <option>Government Body</option>
                       <option>Research Institute</option>
                       <option>Other</option>
                    </select>
                </div>

                <div className="space-y-2">
                    <label className="text-sm font-bold text-gray-500 uppercase tracking-wider ml-2">Message</label>
                    <textarea 
                      rows={4}
                      placeholder="Tell us about your water monitoring needs..." 
                      className="w-full bg-[#F3F0EA] rounded-2xl px-6 py-4 text-[#0f2518] placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[#84cc16] transition-all resize-none"
                    ></textarea>
                </div>

                <div className="pt-4">
                  <button className="group w-full bg-[#0f2518] text-white rounded-full py-4 text-sm font-bold uppercase tracking-widest hover:bg-[#84cc16] hover:text-[#0f2518] transition-all duration-300 flex items-center justify-center gap-2">
                    Send Inquiry 
                    <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
                  </button>
                </div>

              </form>
            </div>

          </div>

        </div>
      </main>

    </div>
  );
};

export default ContactPage;