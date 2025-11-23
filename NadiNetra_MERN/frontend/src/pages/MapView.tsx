import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polygon } from 'react-leaflet';
import { useNavigate } from 'react-router-dom';
import { waterBodies } from '../data/waterBodies';
import 'leaflet/dist/leaflet.css';
import { Icon } from 'leaflet';
import { LatLngExpression } from 'leaflet';

// Fix for default marker icon
const icon = new Icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

export default function MapView() {
  const navigate = useNavigate();

  // Helper to calculate the center of the polygon for the Marker position
  // Also swaps [Lng, Lat] (GeoJSON) to [Lat, Lng] (Leaflet)
  const getCenter = (coords: number[][]): LatLngExpression => {
    if (!coords || coords.length === 0) return [28.6139, 77.2090];
    
    let latSum = 0;
    let lngSum = 0;
    
    coords.forEach(coord => {
      lngSum += coord[0]; // Sum Longitudes
      latSum += coord[1]; // Sum Latitudes
    });
    
    // Return [AvgLat, AvgLng]
    return [latSum / coords.length, lngSum / coords.length];
  };

  // Helper to convert the polygon path from [Lng, Lat] to [Lat, Lng] for Leaflet
  const getPolygonPath = (coords: number[][]): LatLngExpression[] => {
    // @ts-ignore - Leaflet types can be strict about tuples vs arrays
    return coords.map(coord => [coord[1], coord[0]]);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Water Bodies Map</h1>
        <p className="mt-2 text-gray-600">Interactive map showing all monitored water bodies in Delhi</p>
      </div>

      <div className="h-[600px] bg-white rounded-lg shadow-md overflow-hidden">
        <MapContainer
          center={[28.6139, 77.2090]}
          zoom={11}
          className="h-full w-full"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          
          {waterBodies.map(body => {
            const center = getCenter(body.coordinates);
            const polygonPath = getPolygonPath(body.coordinates);

            return (
              <React.Fragment key={body.id}>
                {/* Draw the shape of the water body */}
                <Polygon 
                  positions={polygonPath}
                  pathOptions={{ color: '#3b82f6', fillColor: '#3b82f6', fillOpacity: 0.4 }} 
                />
                
                {/* Place a marker at the center */}
                <Marker
                  position={center}
                  icon={icon}
                >
                  <Popup>
                    <div className="p-2 min-w-[150px]">
                      <h3 className="font-semibold text-lg">{body.name}</h3>
                      <p className="text-sm text-gray-600 mb-2">{body.location}</p>
                      <div className="text-xs text-gray-500 mb-3">
                        Area: {body.area} hectares
                      </div>
                      <button
                        onClick={() => navigate(`/dashboard/lake/${encodeURIComponent(body.name.toLowerCase())}`)}
                        className="block w-full text-center py-1.5 px-3 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 transition-colors"
                      >
                        View Details
                      </button>
                    </div>
                  </Popup>
                </Marker>
              </React.Fragment>
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
}