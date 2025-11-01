import pool from "../../config/db.js";

export const getDetections = async () => {
  const result = await pool.query("SELECT * FROM detections");
  return result.rows;
};
