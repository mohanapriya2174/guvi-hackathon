// small helper to provide mock detections for the UI
export function loadMockDetections() {
  const now = Date.now();
  const pts = [
    { lat: 12.9721, lon: 77.5937 },
    { lat: 12.9753, lon: 77.5992 },
    { lat: 12.9692, lon: 77.5921 },
    { lat: 12.9719, lon: 77.59 },
    { lat: 12.9737, lon: 77.5955 },
  ];
  const types = ["Human", "Animal", "Vehicle", "Human", "Animal"];
  return pts.map((p, i) => ({
    id: 1000 + i,
    type: types[i],
    confidence: 0.7 + Math.random() * 0.28,
    lat: p.lat + (Math.random() - 0.5) * 0.0012,
    lon: p.lon + (Math.random() - 0.5) * 0.0012,
    timestamp: now - i * 1000 * 60 * 30, // spaced 30min apart
    resolved: false,
  }));
}
