import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "../style/Navbar.css";

function Navbar() {
  const navigate = useNavigate();
  const [showUpload, setShowUpload] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const user = JSON.parse(localStorage.getItem("fd_user") || "null");

  function logout() {
    localStorage.removeItem("fd_user");
    navigate("/login");
  }

  function handleFileSelect(e) {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  }

  async function handleFileUpload() {
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      // ✅ Backend FastAPI endpoint (change URL if needed)
      const response = await fetch("http://127.0.0.1:8000/upload_video/", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        alert(`✅ Video "${data.filename}" uploaded successfully!`);
        setShowUpload(false);
        setFile(null);
      } else {
        alert("❌ Upload failed. Please try again!");
      }
    } catch (error) {
      console.error("Upload error:", error);
      alert("⚠️ Error connecting to server!");
    } finally {
      setUploading(false);
    }
  }

  return (
    <>
      <div className="navbar">
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <div className="brand">ForestWatch</div>
          <div className="nav-links">
            <Link to="/dashboard">Dashboard</Link>
            <Link to="/patrol">Patrol</Link>
          </div>
        </div>

        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <button className="upload-btn" onClick={() => setShowUpload(true)}>
            Upload Video
          </button>

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

      {/* Upload Popup Modal */}
      {showUpload && (
        <div className="upload-popup">
          <div className="upload-modal">
            <h2>Upload Surveillance Video</h2>
            <p>Please upload drone or aerial footage (mp4, mov, avi).</p>
            <input
              type="file"
              accept="video/*"
              onChange={handleFileSelect}
              className="file-input"
            />
            <div className="upload-actions">
              <button
                className="cancel-btn"
                onClick={() => setShowUpload(false)}
              >
                Cancel
              </button>
              <button
                className="submit-btn"
                onClick={handleFileUpload}
                disabled={uploading}
              >
                {uploading ? "Uploading..." : "Upload"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default Navbar;
