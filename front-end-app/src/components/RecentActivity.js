import React from "react";
import "../style/RecentActivity.css";

function RecentActivity() {
  const detections = [
    { type: "Human", confidence: 0.92, location: "GPS(12.34, 78.45)" },
    { type: "Animal", confidence: 0.88, location: "GPS(12.36, 78.42)" },
    { type: "Vehicle", confidence: 0.81, location: "GPS(12.30, 78.48)" },
  ];

  return (
    <div className="recent-activity">
      <h3>Recent Activity</h3>
      <ul>
        {detections.map((d, i) => (
          <li key={i}>
            {d.type} detected (Confidence: {d.confidence}) at {d.location}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default RecentActivity;
