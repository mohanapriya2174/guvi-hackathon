import React, { useState, useMemo, useEffect } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
} from "recharts";
import "../style/Trends.css";

function Trends() {
  const [range, setRange] = useState("12h");
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  // new: choose between time-series (historical) and current counts
  const [mode, setMode] = useState("historical"); // 'historical' | 'current'
  const [currentCounts, setCurrentCounts] = useState(null);

  // ✅ Fetch detections (historical) from backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("http://localhost:5000/api/detections");
        if (!res.ok) throw new Error("Failed to fetch data");
        const data = await res.json();
        setItems(data);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  // ✅ Fetch latest metadata counts to be used as "current data"
  useEffect(() => {
    const fetchCurrent = async () => {
      try {
        const res = await fetch("http://localhost:5000/api/recent_detections");
        if (!res.ok) throw new Error("Failed to fetch recent metadata");
        const data = await res.json();
        const counts =
          (data.metadata_json && data.metadata_json.counts) ||
          data.counts ||
          null;
        if (counts) {
          setCurrentCounts({
            Human: Number(counts.human || 0),
            Animal: Number(counts.animal || 0),
            Vehicle: Number(counts.vehicle || 0),
            Deforestation: Number(counts.deforestation || 0),
            LivingBean: Number(counts.living_bean || 0),
          });
        } else {
          setCurrentCounts(null);
        }
      } catch (err) {
        // don't fail whole component for current data errors
        console.warn("Failed to fetch current counts:", err);
        setCurrentCounts(null);
      }
    };
    fetchCurrent();
  }, []);

  // Helper for formatting
  const formatDate = (date) =>
    `${date.getDate()}-${date.toLocaleString("default", { month: "short" })}`;

  // --- Generate time slots based on selected range (historical time-series) ---
  const now = Date.now();
  let slots = [];
  let labels = [];

  if (range === "12h" || range === "24h") {
    const hoursToShow = range === "24h" ? 24 : 12;
    slots = Array.from({ length: hoursToShow }).map((_, i) => {
      return new Date(now - (hoursToShow - 1 - i) * 60 * 60 * 1000);
    });
    labels = slots.map((h) => `${h.getHours().toString().padStart(2, "0")}:00`);
  } else if (range === "7d" || range === "30d") {
    const daysToShow = range === "30d" ? 30 : 7;
    slots = Array.from({ length: daysToShow }).map((_, i) => {
      return new Date(now - (daysToShow - 1 - i) * 24 * 60 * 60 * 1000);
    });
    labels = slots.map((d) => formatDate(d));
  }

  // --- Compute counts per slot for historical series ---
  const types = ["Human", "Animal", "Vehicle"];
  const series = types.map((t) =>
    slots.map((slot) => {
      return items.filter((it) => {
        const d = new Date(it.timestamp || it.created_at);
        if (range === "12h" || range === "24h") {
          return (
            d.getHours() === slot.getHours() &&
            d.getDate() === slot.getDate() &&
            it.type?.toLowerCase() === t.toLowerCase()
          );
        } else {
          return (
            d.getDate() === slot.getDate() &&
            d.getMonth() === slot.getMonth() &&
            it.type?.toLowerCase() === t.toLowerCase()
          );
        }
      }).length;
    })
  );

  const data = labels.map((lab, idx) => ({
    time: lab,
    Human: series[0][idx],
    Animal: series[1][idx],
    Vehicle: series[2][idx],
  }));

  // --- Insights for historical mode ---
  const insights = useMemo(() => {
    if (mode === "current") {
      // when current mode, produce simple insights from currentCounts
      if (!currentCounts) return {};
      const totals = {
        Human: currentCounts.Human || 0,
        Animal: currentCounts.Animal || 0,
        Vehicle: currentCounts.Vehicle || 0,
      };
      const totalCount = totals.Human + totals.Animal + totals.Vehicle;
      const topCategory =
        Object.entries(totals).sort((a, b) => b[1] - a[1])[0]?.[0] || "-";
      return {
        topCategory,
        peakTime: "-", // not applicable
        totalCount,
      };
    }

    if (data.length === 0) return {};
    const totals = { Human: 0, Animal: 0, Vehicle: 0 };
    let peak = { time: "", value: 0 };

    data.forEach((d) => {
      const total = d.Human + d.Animal + d.Vehicle;
      if (total > peak.value) peak = { time: d.time, value: total };
      totals.Human += d.Human;
      totals.Animal += d.Animal;
      totals.Vehicle += d.Vehicle;
    });

    const topCategory = Object.entries(totals).sort(
      (a, b) => b[1] - a[1]
    )[0][0];
    const totalCount = totals.Human + totals.Animal + totals.Vehicle;

    return {
      topCategory,
      peakTime: peak.time,
      totalCount,
    };
  }, [data, mode, currentCounts]);

  // --- UI states ---
  if (loading) return <div className="trends-container">Loading data...</div>;
  if (error)
    return <div className="trends-container error">Error: {error}</div>;

  return (
    <div className="trends-container">
      <div className="trends-header">
        <h3>Detection Trends</h3>

        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <select
            className="range-selector"
            value={range}
            onChange={(e) => setRange(e.target.value)}
            disabled={mode === "current"}
            title={
              mode === "current" ? "Range disabled in Current Data mode" : ""
            }
          >
            <option value="12h">Last 12 Hours</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>

          <select
            className="range-selector"
            value={mode}
            onChange={(e) => setMode(e.target.value)}
          >
            <option value="historical">Historical (time-series)</option>
            <option value="current">Current Data (counts)</option>
          </select>
        </div>
      </div>

      <div className="insight-cards">
        <div className="insight-card">
          <p className="insight-label">Total Detections</p>
          <h4>{insights.totalCount || 0}</h4>
        </div>
        <div className="insight-card">
          <p className="insight-label">
            {mode === "historical"
              ? range.includes("h")
                ? "Peak Hour"
                : "Peak Day"
              : "Peak"}
          </p>
          <h4>{insights.peakTime || "-"}</h4>
        </div>
        <div className="insight-card">
          <p className="insight-label">Most Detected Type</p>
          <h4>{insights.topCategory || "-"}</h4>
        </div>
      </div>

      <div className="chart-wrapper" style={{ height: 360 }}>
        <ResponsiveContainer width="100%" height="100%">
          {mode === "current" ? (
            // Bar chart for current counts
            currentCounts ? (
              <BarChart
                data={[
                  { key: "Human", value: currentCounts.Human },
                  { key: "Animal", value: currentCounts.Animal },
                  { key: "Vehicle", value: currentCounts.Vehicle },
                ]}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="key" tick={{ fill: "#6b7280", fontSize: 12 }} />
                <YAxis tick={{ fill: "#6b7280", fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1f2937",
                    color: "#fff",
                    borderRadius: "8px",
                    border: "none",
                  }}
                />
                <Bar dataKey="value" fill="#3b82f6" />
              </BarChart>
            ) : (
              <div style={{ padding: 24 }}>No current counts available</div>
            )
          ) : (
            // historical line chart (existing)
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="time" tick={{ fill: "#6b7280", fontSize: 12 }} />
              <YAxis tick={{ fill: "#6b7280", fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  color: "#fff",
                  borderRadius: "8px",
                  border: "none",
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="Human"
                stroke="#ef4444"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="Animal"
                stroke="#10b981"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="Vehicle"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default Trends;
