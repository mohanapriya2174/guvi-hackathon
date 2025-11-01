import express from "express";
import { uploadfields } from "./threefield.controller.js";

const router = express.Router();

// ✅ Full path: /api/upload_video
router.post("/upload_fields", uploadfields);

export default router;
