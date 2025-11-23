import React from "react";
import { 
  Droplets, 
  Waves, 
  FlaskConical, 
  Sprout, 
  ScanLine, 
  Sparkles 
} from "lucide-react";

const HydroTransNetCard = () => {
  return (
    <div className="bg-[#F3F0EA] p-8 rounded-[2.5rem] w-full md:w-[450px] min-h-[300px] flex flex-col relative z-10">
      
      {/* Top Section */}
      <div className="mb-6">
        <div className="flex justify-between items-start mb-2">
            <h3 className="text-2xl font-bold text-gray-900 leading-tight">
            HydroTransNet
            </h3>
            <Sparkles size={20} className="text-gray-400" />
        </div>
        <p className="text-gray-500 text-sm leading-relaxed">
          A transformer-based model predicting 5 key water quality indices in real-time.
        </p>
      </div>

      {/* Divider */}
      <div className="w-full h-px bg-gray-300/50 mb-6" />

      {/* Prediction Parameters List - No Boxes, Just Clean UI */}
      <div className="flex flex-col gap-4 mt-auto">
        
        {/* Row 1 */}
        <div className="flex items-center gap-6">
            <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-100/50 rounded-full text-blue-600">
                    <Droplets size={18} />
                </div>
                <span className="text-sm font-semibold text-gray-700">Turbidity</span>
            </div>
            
            <div className="flex items-center gap-3">
                <div className="p-2 bg-amber-100/50 rounded-full text-amber-700">
                    <Waves size={18} />
                </div>
                <span className="text-sm font-semibold text-gray-700">TSS</span>
            </div>
        </div>

        {/* Row 2 */}
        <div className="flex items-center gap-6">
            <div className="flex items-center gap-3">
                <div className="p-2 bg-emerald-100/50 rounded-full text-emerald-600">
                    <FlaskConical size={18} />
                </div>
                <span className="text-sm font-semibold text-gray-700">Chlorophyll</span>
            </div>

            <div className="flex items-center gap-3">
                <div className="p-2 bg-green-100/50 rounded-full text-green-600">
                    <Sprout size={18} />
                </div>
                <span className="text-sm font-semibold text-gray-700">NDVI & NDWI</span>
            </div>
        </div>

        {/* Row 3 */}
        {/* <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-100/50 rounded-full text-cyan-600">
                <ScanLine size={18} />
            </div>
            <span className="text-sm font-semibold text-gray-700">NDWI</span>
        </div> */}

      </div>
    </div>
  );
};

export default HydroTransNetCard;