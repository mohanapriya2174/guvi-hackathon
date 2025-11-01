import { getDetections } from "./detection.service.js";

export const fetchDetections = async (req, res) => {
  try {
    const detections = await getDetections();
    res.json(detections);
  } catch (error) {
    console.error("Error fetching detections:", error);
    res.status(500).json({ message: "Error fetching data" });
  }
};
