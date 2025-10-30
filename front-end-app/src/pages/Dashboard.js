import React from "react";
import Navbar from "../components/Navbar";
import RecentActivity from "../components/RecentActivity";
import Trends from "../components/Trends";
import MapView from "../components/MapView";
import "../style/Dashboard.css";

function Dashboard() {
  // Sample dummy data
  const items = [
    {
      id: 1,
      type: "Human",
      confidence: 0.92,
      lat: 12.9716,
      lon: 77.5946,
      timestamp: Date.now() - 1000 * 60 * 30,
    },
    {
      id: 2,
      type: "Animal",
      confidence: 0.85,
      lat: 12.975,
      lon: 77.59,
      timestamp: Date.now() - 1000 * 60 * 90,
    },
    {
      id: 3,
      type: "Vehicle",
      confidence: 0.81,
      lat: 12.965,
      lon: 77.6,
      timestamp: Date.now() - 1000 * 60 * 150,
    },
  ];

  return (
    <div className="dashboard">
      <Navbar />
      <div className="dashboard-content">
        <div className="left-panel">
          <RecentActivity />
          <Trends items={items} />
        </div>
        <div className="right-panel">
          <h3 className="map-title">Live Detection Map</h3>
          <div className="map-container">
            <MapView items={items} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
