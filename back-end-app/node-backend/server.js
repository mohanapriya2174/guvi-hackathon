import express from "express";
import cors from "cors";
import path from "path";
import fs from "fs";
import dotenv from "dotenv";
import detectionRoutes from "./modules/detection/detection.routers.js"; // ✅ ensure filename matches
import threefieldRoutes from "./modules/threefield/threefield.routes.js";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// ✅ Mount detection routes under /api
app.use("/api", detectionRoutes);
app.use("/api", threefieldRoutes);

// ✅ Serve static images from results folder
const resultsDir = "D:/guvi-hackathon/guvi-hackathon-1/backend/results";
console.log(resultsDir);
app.use("/results", express.static(resultsDir));

// ✅ API route to get the most recent folder’s files (UUID-based)
app.get("/api/recent_detections", (req, res) => {
  try {
    if (!fs.existsSync(resultsDir)) {
      return res.status(404).json({ message: "Results folder not found" });
    }

    const folders = fs
      .readdirSync(resultsDir)
      .filter((folder) =>
        fs.lstatSync(path.join(resultsDir, folder)).isDirectory()
      );

    if (folders.length === 0) {
      return res.json({ message: "No folders found" });
    }

    const latestFolder = folders
      .map((folder) => ({
        name: folder,
        time: fs.statSync(path.join(resultsDir, folder)).mtime.getTime(),
      }))
      .sort((a, b) => b.time - a.time)[0].name;

    const folderPath = path.join(resultsDir, latestFolder);
    const files = fs.readdirSync(folderPath);

    const baseUrl = `${req.protocol}://${req.get("host")}`;
    const fileUrls = files.map((file) => ({
      fileName: file,
      url: `${baseUrl}/results/${latestFolder}/${encodeURIComponent(file)}`,
      extension: path.extname(file).slice(1).toLowerCase(),
    }));

    const images = fileUrls.filter((f) =>
      /\.(png|jpg|jpeg|mp4|mov|avi|mkv)$/i.test(f.fileName)
    );

    // Prefer jsonfile.json if present, otherwise any .json
    const metadataFiles = fileUrls.filter((f) => /\.json$/i.test(f.fileName));
    const preferredMeta = metadataFiles.find(
      (f) => f.fileName.toLowerCase() === "jsonfile.json"
    );
    const metaFile = preferredMeta || metadataFiles[0] || null;

    let metadataJson = null;
    if (metaFile) {
      try {
        const raw = fs.readFileSync(
          path.join(folderPath, metaFile.fileName),
          "utf8"
        );
        metadataJson = JSON.parse(raw);

        // convert outputs filenames to URLs if outputs exist
        if (metadataJson.outputs && typeof metadataJson.outputs === "object") {
          const out = {};
          Object.entries(metadataJson.outputs).forEach(([k, v]) => {
            if (typeof v === "string") {
              const basename = path.basename(v);
              out[k] = {
                path: v,
                url: `${baseUrl}/results/${latestFolder}/${encodeURIComponent(
                  basename
                )}`,
                fileName: basename,
              };
            } else {
              out[k] = v;
            }
          });
          metadataJson.outputs = out;
        }
      } catch (err) {
        console.warn("Failed to parse metadata JSON:", err);
        metadataJson = null;
      }
    }

    const responsePayload = {
      folder: latestFolder,
      files: fileUrls,
      images,
      metadata_file: metaFile || null,
      metadata_json: metadataJson,
      // expose accuracy and found at top-level for convenience
      accuracy:
        metadataJson && metadataJson.accuracy ? metadataJson.accuracy : null,
      found: metadataJson && metadataJson.found ? metadataJson.found : null,
    };

    return res.json(responsePayload);
  } catch (err) {
    console.error("Error while reading results folder:", err);
    return res.status(500).json({ error: "Something went wrong" });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () =>
  console.log(`🚀 Server running at http://localhost:${PORT}`)
);
