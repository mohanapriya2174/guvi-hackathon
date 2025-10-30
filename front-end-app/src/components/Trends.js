import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend,
} from "recharts";

import "../style/Trends.css";

/*
  Trends: aggregate counts over the last N hours and render a line chart.
  Works with items array: { timestamp, type }
*/

function Trends({ items = [] }) {
  const now = Date.now();
  const hours = Array.from({ length: 12 }).map((_, i) => {
    const d = new Date(now - (11 - i) * 60 * 60 * 1000);
    return d;
  });

  const labels = hours.map((h) => `${h.getHours()}:00`);

  const types = ["Human", "Animal", "Vehicle"];
  const series = types.map((t) =>
    hours.map((h) => {
      const hour = h.getHours();
      return items.filter((it) => {
        const d = new Date(it.timestamp);
        return (
          d.getHours() === hour && it.type.toLowerCase() === t.toLowerCase()
        );
      }).length;
    })
  );

  const data = labels.map((lab, idx) => ({
    hour: lab,
    Human: series[0][idx],
    Animal: series[1][idx],
    Vehicle: series[2][idx],
  }));

  return (
    <div style={{ height: 260 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid stroke="#eee" />
          <XAxis dataKey="hour" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="Human" stroke="#ef4444" />
          <Line type="monotone" dataKey="Animal" stroke="#16a34a" />
          <Line type="monotone" dataKey="Vehicle" stroke="#0f62fe" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default Trends;
