// import fs from "fs";
// import path from "path";
// import FormData from "form-data";
// import apiClient from "../../utils/apiClient.js";

// export const processDetection = async (file) => {
//   const formData = new FormData();
//   const fileStream = fs.createReadStream(path.resolve(file.path));
//   formData.append("file", fileStream, file.originalname);

//   // Python backend URL (FastAPI)
//   const PYTHON_URL = "http://127.0.0.1:8000/upload_video/";

//   const response = await apiClient.post(PYTHON_URL, formData, {
//     headers: formData.getHeaders(),
//     maxBodyLength: Infinity,
//   });

//   // Delete temp upload after sending
//   fs.unlinkSync(file.path);

//   return response.data;
// };
