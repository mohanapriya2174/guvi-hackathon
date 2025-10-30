import React from "react";
import { Link, useNavigate } from "react-router-dom";
import "../style/Navbar.css";

function Navbar() {
  const navigate = useNavigate();
  const user = JSON.parse(localStorage.getItem("fd_user") || "null");

  function logout() {
    localStorage.removeItem("fd_user");
    navigate("/login");
  }

  return (
    <div className="navbar">
      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <div className="brand">ForestWatch</div>
        <div className="nav-links">
          <Link to="/dashboard">Dashboard</Link>
          <Link to="/patrol">Patrol</Link>
        </div>
      </div>

      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <div style={{ textAlign: "right", marginRight: 6 }}>
          <div style={{ fontWeight: 700 }}>{user?.name || "Officer"}</div>
          <div style={{ fontSize: 12, color: "#6b7280" }}>
            {user?.role || "Forest Official"}
          </div>
        </div>
        <button className="action-btn" onClick={logout}>
          Logout
        </button>
      </div>
    </div>
  );
}

export default Navbar;
