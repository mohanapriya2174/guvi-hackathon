import axios from "axios";

const apiClient = axios.create({
  timeout: 100000, // allow long processing times for videos
});

export default apiClient;
