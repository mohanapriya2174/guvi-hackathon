import React from "react";
import { useNavigate } from "react-router-dom";

function Login() {
  const nav = useNavigate();
  const [email, setEmail] = React.useState("");
  const [pass, setPass] = React.useState("");

  function submit(e) {
    e.preventDefault();
    // demo auth: any non-empty values allowed
    if (!email || !pass) {
      alert("Enter email and password");
      return;
    }
    const user = { name: "Officer A", email, role: "Forest Official" };
    localStorage.setItem("fd_user", JSON.stringify(user));
    // seed mock detections if not present
    if (!localStorage.getItem("fd_detections")) {
      // lazy-load mock if needed
    }
    nav("/dashboard");
  }

  return (
    <div className="login-wrap">
      <h2 style={{ marginTop: 0 }}>Officer Login</h2>
      <form onSubmit={submit}>
        <div className="field">
          <label>Email</label>
          <input
            className="input"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@forest.gov"
          />
        </div>
        <div className="field">
          <label>Password</label>
          <input
            className="input"
            type="password"
            value={pass}
            onChange={(e) => setPass(e.target.value)}
            placeholder="••••••"
          />
        </div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <button className="action-btn" type="submit">
            Login
          </button>
          <div style={{ fontSize: 12, color: "#6b7280" }}>
            Demo: any creds works
          </div>
        </div>
      </form>
    </div>
  );
}

export default Login;
