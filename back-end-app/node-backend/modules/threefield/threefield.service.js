import pool from "../../config/db.js";

export async function saveVideoMetadata(timestamp, latitude, longitude) {
  const query = `
    INSERT INTO detections (timestamp, latitude, longitude)
    VALUES ($1, $2, $3)
  `;
  const values = [timestamp, type, count, camera_id, latitude, longitude];
  await pool.query(query, values);
}
