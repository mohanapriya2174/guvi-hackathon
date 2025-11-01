import express from "express";
import { fetchDetections } from "./detection.controller.js";

const router = express.Router();

// ✅ Full path: /api/detections
router.get("/detections", fetchDetections);

export default router;
