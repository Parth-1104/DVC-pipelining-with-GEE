import React from 'react';
import { Home, Map, Settings, HelpCircle, Info, Leaf } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';

export default function Sidebar() {
  const location = useLocation();

  const menuItems = [
    { icon: Home, label: 'Dashboard', path: '/dashboard' },
    { icon: Info, label: 'Water Bodies', path: '/dashboard/water-bodies' },
    { icon: Map, label: 'Map View', path: '/dashboard/map' },
    { icon: Settings, label: 'Settings', path: '/dashboard/settings' },
    { icon: HelpCircle, label: 'Help', path: '/dashboard/help' },
  ];

  return (
    <div className="h-full w-full p-4">
      <div className="h-full w-full bg-[#0f2518] rounded-[2.5rem] shadow-2xl flex flex-col p-6 text-white relative overflow-hidden">
        
        {/* Background Decor */}
        <div className="absolute top-0 right-0 w-40 h-40 bg-[#84cc16] rounded-full blur-[80px] opacity-10 pointer-events-none -mr-10 -mt-10"></div>
        <div className="absolute bottom-0 left-0 w-40 h-40 bg-blue-500 rounded-full blur-[80px] opacity-10 pointer-events-none -ml-10 -mb-10"></div>

        {/* Logo Area */}
        <div className="flex items-center gap-3 mb-10 px-2 relative z-10">
          <div className="p-2 bg-white/10 rounded-xl backdrop-blur-sm border border-white/10 text-[#84cc16]">
            <Leaf className="h-6 w-6 fill-current" />
          </div>
          <span className="text-2xl font-bold tracking-tight">Nadi Netra</span>
        </div>

        {/* Menu Items */}
        <div className="space-y-2 flex-1 relative z-10">
          {menuItems.map(({ icon: Icon, label, path }) => {
            const isActive = location.pathname === path;
            return (
              <Link
                key={path}
                to={path}
                className={`flex items-center space-x-3 px-5 py-4 rounded-2xl transition-all duration-300 group ${
                  isActive
                    ? 'bg-[#84cc16] text-[#0f2518] shadow-lg shadow-[#84cc16]/20 font-bold translate-x-1'
                    : 'text-gray-400 hover:bg-white/10 hover:text-white hover:translate-x-1'
                }`}
              >
                <Icon className={`h-5 w-5 ${isActive ? 'text-[#0f2518]' : 'text-gray-400 group-hover:text-white'}`} />
                <span>{label}</span>
              </Link>
            );
          })}
        </div>

        {/* Footer / Status */}
        <div className="mt-auto px-2 relative z-10">
           <div className="bg-white/5 rounded-2xl p-4 border border-white/5 backdrop-blur-sm">
              <p className="text-xs text-gray-400 uppercase tracking-wider font-bold mb-1">System Status</p>
              <div className="flex items-center gap-2">
                 <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                 </span>
                 <span className="text-sm font-medium text-white">Online</span>
              </div>
           </div>
        </div>

      </div>
    </div>
  );
}