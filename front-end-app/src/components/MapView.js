import React, { useState, useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup, Tooltip } from "react-leaflet";
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

// small helper to create colored emoji icons per type
const makeDivIcon = (emoji, bg = "#fff") =>
  new L.DivIcon({
    html: `<div style="
      width:34px;height:34px;border-radius:50%;
      display:flex;align-items:center;justify-content:center;
      background:${bg};box-shadow:0 1px 3px rgba(0,0,0,0.25);
      font-size:18px;">${emoji}</div>`,
    className: "",
    iconSize: [34, 34],
    iconAnchor: [17, 34],
  });

const ICONS = {
  human: makeDivIcon("👤", "#60a5fa"),
  animal: makeDivIcon("🐾", "#f87171"),
  vehicle: makeDivIcon("🚗", "#facc15"),
  default: makeDivIcon("📍", "#9ca3af"),
};

function MapView() {
  const [items, setItems] = useState([]);
  const [range, setRange] = useState("12h");
  const [loading, setLoading] = useState(true);

  // fetch detections (normalized)
  useEffect(() => {
    let mounted = true;
    async function fetchDetections() {
      try {
        const res = await fetch("http://localhost:5000/api/detections");
        const data = await res.json();
        if (!mounted) return;
        const arr = Array.isArray(data) ? data : data.items || [];
        const normalized = arr
          .map((it, idx) => {
            const lat =
              it.latitude ??
              it.lat ??
              it.latitute ??
              it.metadata_json?.latitude ??
              it.metadata_json?.lat ??
              null;
            const lon =
              it.longitude ??
              it.lon ??
              it.lng ??
              it.metadata_json?.longitude ??
              it.metadata_json?.lon ??
              null;
            return {
              id: it.id ?? idx,
              type: (it.type || it.metadata_json?.found?.type || "") + "",
              timestamp:
                it.timestamp ||
                it.created_at ||
                it.metadata_json?.timestamp ||
                Date.now(),
              latitude: lat !== null ? Number(lat) : null,
              longitude: lon !== null ? Number(lon) : null,
              confidence: it.confidence ?? it.metadata_json?.accuracy ?? 0,
              raw: it,
            };
          })
          .filter(
            (x) => Number.isFinite(x.latitude) && Number.isFinite(x.longitude)
          );
        setItems(normalized);
      } catch (e) {
        console.error("fetch detections error", e);
        setItems([]);
      } finally {
        if (mounted) setLoading(false);
      }
    }
    fetchDetections();
    return () => (mounted = false);
  }, []);

  // filter by selected time range
  const now = Date.now();
  const filtered = items.filter((it) => {
    const diff = now - new Date(it.timestamp).getTime();
    switch (range) {
      case "12h":
        return diff <= 12 * 60 * 60 * 1000;
      case "24h":
        return diff <= 24 * 60 * 60 * 1000;
      case "7d":
        return diff <= 7 * 24 * 60 * 60 * 1000;
      case "30d":
        return diff <= 30 * 24 * 60 * 60 * 1000;
      default:
        return true;
    }
  });

  const center = filtered.length
    ? [filtered[0].latitude, filtered[0].longitude]
    : [20.5937, 78.9629];

  const pickIcon = (type) => {
    const t = (type || "").toLowerCase();
    if (t.includes("human") || t.includes("person")) return ICONS.human;
    if (
      t.includes("animal") ||
      t.includes("dog") ||
      t.includes("cat") ||
      t.includes("deer")
    )
      return ICONS.animal;
    if (
      t.includes("veh") ||
      t.includes("car") ||
      t.includes("truck") ||
      t.includes("bus")
    )
      return ICONS.vehicle;
    return ICONS.default;
  };

  if (loading) return <div className="map-loading">Loading map...</div>;

  return (
    <div className="mapview-root">
      <div
        className="map-controls"
        style={{
          marginBottom: 8,
          display: "flex",
          gap: 8,
        }}
      >
        <button
          className={`btn ${range === "12h" ? "active" : ""}`}
          onClick={() => setRange("12h")}
        >
          12 hrs
        </button>
        <button
          className={`btn ${range === "24h" ? "active" : ""}`}
          onClick={() => setRange("24h")}
        >
          24 hrs
        </button>
        <button
          className={`btn ${range === "7d" ? "active" : ""}`}
          onClick={() => setRange("7d")}
        >
          1 week
        </button>
        <button
          className={`btn ${range === "30d" ? "active" : ""}`}
          onClick={() => setRange("30d")}
        >
          1 month
        </button>
      </div>

      <div style={{ height: 480 }}>
        <MapContainer
          center={center}
          zoom={6}
          style={{ height: "100%", width: "100%" }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution="&copy; OpenStreetMap contributors"
          />
          {filtered.map((it) => (
            <Marker
              key={it.id}
              position={[it.latitude, it.longitude]}
              icon={pickIcon(it.type)}
            >
              <Tooltip direction="top" offset={[0, -10]}>
                <div style={{ fontSize: 12 }}>
                  Lat: {it.latitude.toFixed(5)} • Lon: {it.longitude.toFixed(5)}
                  <br />
                  {new Date(it.timestamp).toLocaleString()}
                </div>
              </Tooltip>
              <Popup>
                <div style={{ minWidth: 180 }}>
                  <div style={{ fontWeight: 700 }}>{it.type || "Unknown"}</div>
                  <div style={{ fontSize: 12, color: "#6b7280" }}>
                    {new Date(it.timestamp).toLocaleString()}
                  </div>
                  <div>Lat: {it.latitude.toFixed(6)}</div>
                  <div>Lon: {it.longitude.toFixed(6)}</div>
                  <div>Confidence: {Number(it.confidence || 0).toFixed(3)}</div>
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>
    </div>
  );
}

export default MapView;
