import React from 'react';
import { Calendar } from 'lucide-react';

interface TimeSpanSelectorProps {
  selectedSpan: string;
  onSpanChange: (span: string) => void;
}

export default function TimeSpanSelector({ selectedSpan, onSpanChange }: TimeSpanSelectorProps) {
  const timeSpans = [
    { value: '1W', label: '1 Week' },
    { value: '1M', label: '1 Month' },
    { value: '6M', label: '6 Months' },
    { value: '1Y', label: '1 Year' }
  ];

  return (
    <div className="flex items-center gap-2 mb-4">
      <Calendar className="h-5 w-5 text-gray-500" />
      <div className="flex gap-2">
        {timeSpans.map(span => (
          <button
            key={span.value}
            onClick={() => onSpanChange(span.value)}
            className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
              selectedSpan === span.value
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {span.label}
          </button>
        ))}
      </div>
    </div>
  );
}