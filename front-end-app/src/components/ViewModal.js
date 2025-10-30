import React from "react";

function ViewModal({ isOpen, onClose, image }) {
  if (!isOpen) return null;

  return (
    <div
      className="modal-overlay"
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0,0,0,0.6)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        zIndex: 1000,
      }}
      onClick={onClose}
    >
      <div
        className="modal-content"
        style={{
          background: "#fff",
          padding: "20px",
          borderRadius: "8px",
          maxWidth: "80%",
          maxHeight: "80%",
          overflow: "auto",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <img
          src={image}
          alt="Detected"
          style={{ width: "100%", borderRadius: "8px" }}
        />
        <button
          onClick={onClose}
          style={{
            marginTop: "10px",
            background: "#2e7d32",
            color: "white",
            border: "none",
            padding: "8px 12px",
            borderRadius: "5px",
            cursor: "pointer",
          }}
        >
          Close
        </button>
      </div>
    </div>
  );
}

export default ViewModal;
