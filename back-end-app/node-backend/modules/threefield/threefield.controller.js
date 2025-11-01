import { saveVideoMetadata } from "./threefield.service.js";

export const uploadfields = async (req, res) => {
  try {
    const file = req.file;
    const { timestamp, latitude, longitude } = req.body;

    if (!file) {
      return res.status(400).json({ message: "No file uploaded" });
    }

    // ✅ Store metadata in DB
    await saveVideoMetadata(timestamp, latitude, longitude);

    res.json({
      message: "Video uploaded successfully",
      //   filename: file.filename,
    });
  } catch (error) {
    console.error("Upload error:", error);
    res.status(500).json({ message: "Server error" });
  }
};
