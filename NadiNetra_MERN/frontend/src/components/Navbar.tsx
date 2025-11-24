import React, { useState } from 'react';
import { Search, Leaf } from 'lucide-react'; // Changed icon to Leaf to match landing
import { Link, useNavigate } from 'react-router-dom';
import { waterBodies } from '../data/waterBodies';

export default function Navbar() {
  const [searchQuery, setSearchQuery] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const navigate = useNavigate();

  const filteredBodies = waterBodies.filter(body =>
    body.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleSelect = (name: string) => {
    navigate(`/dashboard/lake/${encodeURIComponent(name.toLowerCase())}`);
    setShowDropdown(false);
    setSearchQuery('');
  };

  return (
    <nav className="bg-white/80 backdrop-blur-md shadow-sm rounded-full px-6 py-3 border border-white/50 transition-all duration-200">
      <div className="flex items-center justify-between">
        
        {/* Brand (Visible on Mobile/Tablet if sidebar hidden, or just extra context) */}
        <div className="flex items-center md:hidden">
          <Link to="/" className="p-1.5 border-2 border-green-600 rounded-lg mr-2 text-green-700">
             <Leaf className="h-5 w-5 fill-current" />
          </Link>
          <span className="text-xl font-bold text-[#0f2518]">Nadi Netra</span>
        </div>
        
        {/* Search Bar */}
        <div className="relative flex-1 max-w-2xl mx-auto md:mx-0">
          <div className="relative group">
            <input
              type="text"
              className="w-full pl-12 pr-6 py-2.5 rounded-full border-none bg-[#F3F0EA] text-[#0f2518] placeholder-gray-400 focus:ring-2 focus:ring-[#84cc16] focus:bg-white transition-all duration-300 shadow-inner"
              placeholder="Search water bodies..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setShowDropdown(true);
              }}
              onFocus={() => setShowDropdown(true)}
            />
            <Search className="absolute left-4 top-3 h-5 w-5 text-gray-400 group-focus-within:text-[#84cc16] transition-colors" />
          </div>
          
          {/* Dropdown Results */}
          {showDropdown && searchQuery && (
            <div className="absolute w-full mt-2 bg-white rounded-[1.5rem] shadow-xl border border-gray-100 overflow-hidden z-50">
              {filteredBodies.length > 0 ? (
                filteredBodies.map(body => (
                  <div
                    key={body.id}
                    className="px-6 py-3 hover:bg-[#F3F0EA] cursor-pointer text-[#0f2518] font-medium transition-colors duration-200 border-b border-gray-50 last:border-0"
                    onClick={() => handleSelect(body.name)}
                  >
                    {body.name}
                  </div>
                ))
              ) : (
                <div className="px-6 py-3 text-gray-400">No results found</div>
              )}
            </div>
          )}
        </div>

        {/* User Profile / Context (Optional) */}
        {/* <div className="hidden md:flex items-center gap-4 ml-6">
           <div className="h-10 w-10 rounded-full bg-[#0f2518] text-[#84cc16] flex items-center justify-center font-bold shadow-md">
              JD
           </div>
        </div> */}
      </div>
    </nav>
  );
}