import React from "react";
import { MapContainer, TileLayer, Marker, Popup, Circle } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "../style/Trends.css";

// fix default icon path for many bundlers
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require("leaflet/dist/images/marker-icon-2x.png"),
  iconUrl: require("leaflet/dist/images/marker-icon.png"),
  shadowUrl: require("leaflet/dist/images/marker-shadow.png"),
});

function MapView({ items = [] }) {
  const center = items.length
    ? [items[0].lat, items[0].lon]
    : [12.9716, 77.5946];

  return (
    <MapContainer
      center={center}
      zoom={13}
      style={{ height: "100%", width: "100%" }}
    >
      <TileLayer
        attribution="&copy; OpenStreetMap contributors"
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {items.map((it) => (
        <Marker key={it.id} position={[it.lat, it.lon]}>
          <Popup>
            <div style={{ minWidth: 160 }}>
              <div style={{ fontWeight: 700 }}>{it.type}</div>
              <div style={{ fontSize: 12, color: "#6b7280" }}>
                {new Date(it.timestamp).toLocaleString()}
              </div>
              <div>Confidence: {it.confidence.toFixed(2)}</div>
            </div>
          </Popup>
        </Marker>
      ))}
      {items
        .filter((i) => !i.resolved)
        .map((it) => (
          <Circle
            key={"c" + it.id}
            center={[it.lat, it.lon]}
            radius={60}
            pathOptions={{ color: "red", opacity: 0.14 }}
          />
        ))}
    </MapContainer>
  );
}

export default MapView;
