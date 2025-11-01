import React, { useEffect, useState } from "react";
import "../style/RecentActivity.css";

function RecentActivity() {
  const [detections, setDetections] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [fullscreenImage, setFullscreenImage] = useState(null);
  const [cache, setCache] = useState({});
  const [accuracy, setAccuracy] = useState(null);
  const [found, setFound] = useState(null);

  useEffect(() => {
    async function fetchDetections() {
      try {
        const response = await fetch(
          "http://localhost:5000/api/recent_detections"
        );
        const data = await response.json();

        // set accuracy & found directly from top-level response (from jsonfile.json)
        setAccuracy(
          data.accuracy ||
            (data.metadata_json && data.metadata_json.accuracy) ||
            null
        );
        setFound(
          data.found || (data.metadata_json && data.metadata_json.found) || null
        );

        const images = Array.isArray(data.images) ? data.images : [];
        const imageItems = images.map((item, idx) => {
          const isForest = item.fileName?.toLowerCase().includes("forest");
          const conf = isForest
            ? data.accuracy?.deforestation ??
              data.metadata_json?.accuracy?.deforestation ??
              0
            : Math.max(
                data.accuracy?.human ?? 0,
                data.accuracy?.animal ?? 0,
                data.accuracy?.vehicle ?? 0,
                data.metadata_json?.accuracy?.human ?? 0,
                data.metadata_json?.accuracy?.animal ?? 0,
                data.metadata_json?.accuracy?.vehicle ?? 0
              );
          return {
            id: `img-${idx}`,
            image_url: item.url,
            fileName: item.fileName,
            type: isForest ? "Forest Segmentation" : "Detection",
            confidence: conf,
            timestamp: Date.now() - idx * 1000,
          };
        });

        // include outputs from metadata_json.outputs (if present) and map proper confidence
        const outputItems = [];
        const outputs = data.metadata_json?.outputs || {};
        if (outputs && typeof outputs === "object") {
          Object.entries(outputs).forEach(([key, val], i) => {
            // val may be { url, fileName, path } or a string
            const url =
              val?.url ||
              (typeof val === "string" ? `${data.baseUrl || ""}/${val}` : null);
            const fileName =
              val?.fileName || (typeof val === "string" ? val : key);
            const isForest =
              key.toLowerCase().includes("forest") ||
              fileName.toLowerCase().includes("forest");
            const conf = isForest
              ? data.accuracy?.deforestation ??
                data.metadata_json?.accuracy?.deforestation ??
                0
              : Math.max(
                  data.accuracy?.human ?? 0,
                  data.accuracy?.animal ?? 0,
                  data.accuracy?.vehicle ?? 0,
                  data.metadata_json?.accuracy?.human ?? 0,
                  data.metadata_json?.accuracy?.animal ?? 0,
                  data.metadata_json?.accuracy?.vehicle ?? 0
                );

            if (url) {
              outputItems.push({
                id: `out-${i}`,
                image_url: url,
                fileName,
                type: isForest ? "Forest Segmentation" : key,
                confidence: conf,
                timestamp: Date.now() - i * 1000,
              });
            }
          });
        }

        // merge images and outputs (avoid duplicates by fileName)
        const byName = new Map();
        [...imageItems, ...outputItems].forEach((it) => {
          if (!byName.has(it.fileName)) byName.set(it.fileName, it);
        });

        setDetections(Array.from(byName.values()));
      } catch (error) {
        console.error("Error fetching detections:", error);
        setDetections([]);
        setAccuracy(null);
        setFound(null);
      }
    }

    fetchDetections();
    const handler = () => fetchDetections();
    window.addEventListener("detections:updated", handler);
    return () => window.removeEventListener("detections:updated", handler);
  }, []);

  // preload images
  useEffect(() => {
    if (!Array.isArray(detections) || detections.length === 0) return;
    detections.forEach((d) => {
      if (!d || !d.image_url) return;
      if (!cache[d.image_url]) {
        const img = new Image();
        img.src = d.image_url;
        img.onload = () =>
          setCache((prev) => ({ ...prev, [d.image_url]: true }));
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detections]);

  // ensure currentIndex stays within a sensible slide range when detections change
  useEffect(() => {
    const slidesCount = Math.max(3, detections.length);
    if (currentIndex >= slidesCount) setCurrentIndex(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detections]);

  const slidesCount = Math.max(3, detections.length);
  const handleNext = () => setCurrentIndex((prev) => (prev + 1) % slidesCount);
  const handlePrev = () =>
    setCurrentIndex((prev) => (prev - 1 + slidesCount) % slidesCount);

  const displayIndex = currentIndex % slidesCount;
  const showingImage = displayIndex < detections.length;
  const current = showingImage ? detections[displayIndex] : null;

  return (
    <div className="recent-activity">
      <h3>Recent Activity</h3>

      {!showingImage && !current ? (
        // aggregated slide when no image present at this index
        <div className="activity-card">
          <div className="activity-info">
            <h4>Aggregated Results</h4>
            {accuracy ? (
              <div className="accuracy-block">
                <h5>Accuracy</h5>
                <div>Human: {accuracy.human ?? "-"}</div>
                <div>Animal: {accuracy.animal ?? "-"}</div>
                <div>Vehicle: {accuracy.vehicle ?? "-"}</div>
                <div>Deforestation: {accuracy.deforestation ?? "-"}</div>
              </div>
            ) : (
              <div>Accuracy: N/A</div>
            )}

            {found ? (
              <div className="found-block">
                <h5>Found</h5>
                <div>Human: {found.human ? "Yes" : "No"}</div>
                <div>Animal: {found.animal ? "Yes" : "No"}</div>
                <div>Cars: {found.cars ? "Yes" : "No"}</div>
                <div>Deforestation: {found.deforestation ? "Yes" : "No"}</div>
                <div>Living Bean: {found.living_bean ? "Yes" : "No"}</div>
              </div>
            ) : (
              <div>Found flags: N/A</div>
            )}
          </div>

          <div className="image-viewer placeholder">
            <div className="aggregated-placeholder">
              Aggregated data (slide 3)
            </div>
            <button className="nav-btn left" onClick={handlePrev}>
              ◀
            </button>
            <button className="nav-btn right" onClick={handleNext}>
              ▶
            </button>
          </div>
        </div>
      ) : !current ? (
        <p>Loading detections...</p>
      ) : (
        <div className="activity-card">
          <div className="activity-info">
            {displayIndex === 2 ? (
              <>
                <h4>Aggregated Results</h4>
                {accuracy ? (
                  <div className="accuracy-block">
                    <h5>Accuracy</h5>
                    <div>Human: {accuracy.human ?? "-"}</div>
                    <div>Animal: {accuracy.animal ?? "-"}</div>
                    <div>Vehicle: {accuracy.vehicle ?? "-"}</div>
                    <div>Deforestation: {accuracy.deforestation ?? "-"}</div>
                  </div>
                ) : (
                  <div>Accuracy: N/A</div>
                )}

                {found ? (
                  <div className="found-block">
                    <h5>Found</h5>
                    <div>Human: {found.human ? "Yes" : "No"}</div>
                    <div>Animal: {found.animal ? "Yes" : "No"}</div>
                    <div>Cars: {found.cars ? "Yes" : "No"}</div>
                    <div>
                      Deforestation: {found.deforestation ? "Yes" : "No"}
                    </div>
                    <div>Living Bean: {found.living_bean ? "Yes" : "No"}</div>
                  </div>
                ) : (
                  <div>Found flags: N/A</div>
                )}
              </>
            ) : (
              <>
                <p>
                  <strong>{current.type}</strong> detected
                </p>
                <p>
                  Confidence: <b>{(current.confidence || 0).toFixed(3)}</b>
                </p>
                {/* <p>Location: {current.location || "Unknown"}</p> */}
              </>
            )}
          </div>

          <div className="image-viewer">
            <img
              src={current.image_url}
              alt={current.type || "detection"}
              onClick={() => setFullscreenImage(current.image_url)}
              className="activity-image"
            />
            <button className="nav-btn left" onClick={handlePrev}>
              ◀
            </button>
            <button className="nav-btn right" onClick={handleNext}>
              ▶
            </button>
          </div>
        </div>
      )}

      {fullscreenImage && (
        <div
          className="fullscreen-popup"
          onClick={() => setFullscreenImage(null)}
        >
          <img
            src={fullscreenImage}
            alt="Detection Full View"
            className="fullscreen-image"
          />
        </div>
      )}
    </div>
  );
}

export default RecentActivity;
