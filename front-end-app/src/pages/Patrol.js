import React, { useState } from "react";
import ViewModal from "../components/ViewModal";
import { loadMockDetections } from "../utils/mockData";

function Patrol() {
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);

  const handleView = (imageUrl) => {
    setSelectedImage(imageUrl);
    setModalOpen(true);
  };

  const handleClose = () => {
    setModalOpen(false);
    setSelectedImage(null);
  };

  const patrols = [
    {
      datetime: "Oct 25, 11:43 PM",
      type: "Human",
      confidence: "0.92",
      location: "GPS(12.34, 78.45)",
      image: "https://placekitten.com/400/300",
      action: "Unresolved",
    },
    {
      datetime: "Oct 26, 01:15 AM",
      type: "Vehicle",
      confidence: "0.88",
      location: "GPS(11.90, 77.60)",
      image: "https://placekitten.com/401/300",
      action: "Resolved",
    },
  ];

  return (
    <div className="patrol-page">
      <h2>Patrol Records</h2>
      <table>
        <thead>
          <tr>
            <th>Date/Time</th>
            <th>Type</th>
            <th>Confidence</th>
            <th>Location</th>
            <th>Image/Video</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {patrols.map((p, index) => (
            <tr key={index}>
              <td>{p.datetime}</td>
              <td>{p.type}</td>
              <td>{p.confidence}</td>
              <td>{p.location}</td>
              <td>
                <button onClick={() => handleView(p.image)}>View</button>
              </td>
              <td>
                <button>
                  {p.action === "Resolved" ? "Resolved" : "Mark Resolved"}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <ViewModal
        isOpen={modalOpen}
        onClose={handleClose}
        image={selectedImage}
      />
    </div>
  );
}

export default Patrol;
