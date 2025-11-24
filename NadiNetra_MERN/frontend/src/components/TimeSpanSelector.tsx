

interface TimeSpanSelectorProps {
  selectedSpan: string;
  onSpanChange: (span: string) => void;
  className?: string;
}


export default function TimeSpanSelector({ selectedSpan, onSpanChange, className = '' } : TimeSpanSelectorProps) {
  const spans = [
    { label: '1W', value: '1W' },
    { label: '1M', value: '1M' },
    { label: '6M', value: '6M' },
    { label: '1Y', value: '1Y' },
  ];

  return (
    <div className={`inline-flex bg-white p-1 rounded-full border border-gray-200 shadow-sm ${className}`}>
      {spans.map((span) => (
        <button
          key={span.value}
          onClick={() => onSpanChange(span.value)}
          className={`px-6 py-2 rounded-full text-sm font-bold transition-all duration-300 ${
            selectedSpan === span.value
              ? 'bg-[#0f2518] text-[#84cc16] shadow-md'
              : 'text-gray-500 hover:text-[#0f2518] hover:bg-gray-50'
          }`}
        >
          {span.label}
        </button>
      ))}
    </div>
  );
}