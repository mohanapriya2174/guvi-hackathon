// import express from "express";
// import fs from "fs";
// import path from "path";

// const router = express.Router();

// // Function to get latest folder from results directory
// function getLatestFolder(resultsPath) {
//   const folders = fs.readdirSync(resultsPath)
//     .map(name => ({
//       name,
//       time: fs.statSync(path.join(resultsPath, name)).mtime.getTime()
//     }))
//     .sort((a, b) => b.time - a.time);

//   return folders.length > 0 ? folders[0].name : null;
// }

// // GET /api/recent_detections
// router.get("/recent_detections", (req, res) => {
//   try {
//     const resultsPath = path.join(process.cwd(), "results");
//     const latestFolder = getLatestFolder(resultsPath);

//     if (!latestFolder) {
//       return res.json({ message: "No result folders found" });
//     }

//     const folderPath = path.join(resultsPath, latestFolder);
//     const files = fs.readdirSync(folderPath)
//       .filter(file => /\.(jpg|jpeg|png)$/i.test(file));

//     const baseUrl = `${req.protocol}://${req.get("host")}`;

//     const detections = files.map(filename => ({
//       image_url: `${baseUrl}/results/${latestFolder}/${filename}`,
//       folder: latestFolder,
//       timestamp: new Date().toISOString(),
//     }));

//     res.json(detections);
//   } catch (error) {
//     console.error("❌ Error reading results folder:", error);
//     res.status(500).json({ error: "Failed to fetch images" });
//   }
// });

// export default router;
