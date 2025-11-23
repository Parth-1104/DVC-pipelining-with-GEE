import React from 'react';
import { useNavigate } from 'react-router-dom';

interface StatsCardProps {
  title: string;
  value: number;
  unit?: string;
  icon: React.ReactNode;
  color: string;
  link?: string;
  criticalLink?: boolean;
}

export default function StatsCard({ title, value, unit, icon, color, link, criticalLink }: StatsCardProps) {
  const navigate = useNavigate();
  
  return (
    <div 
      className={`bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 ${(link || criticalLink) ? 'cursor-pointer hover:shadow-lg transition-shadow' : ''} transition-colors duration-200`}
      onClick={() => {
        if (criticalLink) {
          navigate('/critical-status');
        } else if (link) {
          navigate(link);
        }
      }}
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400">{title}</p>
          <p className="text-2xl font-semibold mt-1 text-gray-900 dark:text-white">
            {value}
            {unit && <span className="text-sm ml-1">{unit}</span>}
          </p>
        </div>
        <div className={`p-3 rounded-full ${color} transition-colors duration-200`}>
          {icon}
        </div>
      </div>
    </div>
  );
}