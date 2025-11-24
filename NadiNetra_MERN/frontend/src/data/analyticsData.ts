import { HistoricalData, CommunityReport, WQIPrediction } from '../types';
import { addDays, subDays, format } from 'date-fns';

const today = new Date();

export const historicalData: HistoricalData[] = Array.from({ length: 30 }, (_, i) => ({
  date: format(subDays(today, 29 - i), 'yyyy-MM-dd'),
  turbidity: 8 + Math.random() * 4,
  pollutants: 20 + Math.random() * 10,
  biodiversity: 75 + Math.random() * 10
}));

export const predictions: WQIPrediction[] = [
  ...historicalData.slice(-7).map(data => ({
    date: data.date,
    predicted: 75 + Math.random() * 5,
    actual: 74 + Math.random() * 6
  })),
  ...Array.from({ length: 7 }, (_, i) => ({
    date: format(addDays(today, i + 1), 'yyyy-MM-dd'),
    predicted: 76 + Math.random() * 4
  }))
];

export const communityReports: CommunityReport[] = [
  {
    id: '1',
    waterBodyId: '1',
    reportType: 'pollution',
    description: 'Observed unusual water coloration and debris',
    status: 'investigating',
    createdAt: '2024-03-14T08:30:00Z'
  },
  {
    id: '2',
    waterBodyId: '2',
    reportType: 'illegal_activity',
    description: 'Unauthorized fishing activity spotted',
    status: 'resolved',
    createdAt: '2024-03-13T15:45:00Z'
  },
  {
    id: '3',
    waterBodyId: '3',
    reportType: 'infrastructure',
    description: 'Damaged monitoring equipment',
    status: 'pending',
    createdAt: '2024-03-15T11:20:00Z'
  }
];

export const pollutionHeatmapData = {
  max: 100,
  data: [
    { lat: 28.7357, lng: 77.1674, value: 85 }, // Bhalswa Lake
    { lat: 28.7495, lng: 77.1183, value: 45 }, // DTU Lake
    { lat: 28.5478, lng: 77.1920, value: 65 }, // Hauz Khas Lake
    { lat: 28.7019, lng: 77.2016, value: 55 }, // Naini Lake
    { lat: 28.6139, lng: 77.3063, value: 75 }  // Sanjay Lake
  ]
};