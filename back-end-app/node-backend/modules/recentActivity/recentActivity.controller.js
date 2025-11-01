// import { processDetection } from "./detection.service.js";

// export const handleDetectionUpload = async (req, res) => {
//   try {
//     const file = req.file;

//     if (!file) {
//       return res.status(400).json({ message: "No file uploaded" });
//     }

//     const result = await processDetection(file);
//     return res.status(200).json(result);
//   } catch (err) {
//     console.error("❌ Detection upload error:", err.message);
//     return res
//       .status(500)
//       .json({ message: "Internal Server Error", error: err.message });
//   }
// };
