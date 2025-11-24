import { WaterBody, WaterQualityStats } from '../types';


export const waterBodies: WaterBody[] = [
  {
    id: '1',
    name: 'Bhalswa Lake',
    location: 'North Delhi',
    area: 34,
    lastUpdated: '2024-03-15T10:30:00Z',
    coordinates: [
      [77.1560, 28.7435], // Southwest corner
      [77.1650, 28.7435], // Southeast corner
      [77.1650, 28.7500], // Northeast corner
      [77.1560, 28.7500] 
    ]


  },
  {
    id: '2',
    name: 'DTU Lake',
    location: 'Northwest Delhi',
    area: 12,

    lastUpdated: '2024-03-15T10:30:00Z',
    coordinates: [
      [77.1060, 28.7035],
      [77.1130, 28.7035],
      [77.1130, 28.7090],
      [77.1060, 28.7090],
      [77.1060, 28.7035]
    ]

  },
  {
    id: '3',
    name: 'Hauz Khas Lake',
    location: 'South Delhi',
    area: 26,

    lastUpdated: '2024-03-15T10:30:00Z',
    coordinates: [
      [77.1950, 28.5460],
      [77.2020, 28.5460],
      [77.2020, 28.5525],
      [77.1950, 28.5525],
      [77.1950, 28.5460]
    ]

  },
  {
    id: '4',
    name: 'Naini Lake',
    location: 'North Delhi',
    area: 18,
    lastUpdated: '2024-03-15T10:30:00Z',
    coordinates: [
      [81.7910, 25.4390],
      [81.8000, 25.4390],
      [81.8000, 25.4450],
      [81.7910, 25.4450],
      [81.7910, 25.4390]
    ]


  },
  {
    id: '5',
    name: 'Sanjay Lake',
    location: 'East Delhi',
    area: 42,
    lastUpdated: '2024-03-15T10:30:00Z',
    coordinates: [
      [77.3190, 28.6370],
      [77.3270, 28.6370],
      [77.3270, 28.6430],
      [77.3190, 28.6430],
      [77.3190, 28.6370]
    ]
  }
];

export const waterQualityStats: WaterQualityStats = {
  totalBodies: 7,
  averageQuality: 78,
  criticalBodies: 2,
  monitoredBodies: 7
};