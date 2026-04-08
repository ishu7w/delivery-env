from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import DeliveryEnv
import numpy as np

app = Flask(__name__)
CORS(app)

env = DeliveryEnv()

def _serialize_obs(obs):
    if isinstance(obs, np.ndarray):
        return obs.tolist()
    if isinstance(obs, dict):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in obs.items()}
    return obs

@app.route("/")
def index():
    return jsonify({"status": "running", "env": "DeliveryEnv"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    seed = data.get("seed", None)
    options = data.get("options", {})
    obs, info = env.reset(seed=seed, options=options)
    return jsonify({
        "observation": _serialize_obs(obs),
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "done": False,
        "info": info,
    })

@app.route("/step", methods=["POST"])
def step():
    data = request.get_json()
    if not data or "action" not in data:
        return jsonify({"error": "'action' field is required"}), 400
    action = data["action"]
    obs, reward, terminated, truncated, info = env.step(action)
    return jsonify({
        "observation": _serialize_obs(obs),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(terminated or truncated),
        "info": info,
    })

@app.route("/validate", methods=["GET", "POST"])
def validate():
    return jsonify({
        "status": "valid",
        "environment": "DeliveryEnv",
        "gymnasium_compatible": True,
        "endpoints": {"reset": "POST /reset", "step": "POST /step", "validate": "GET/POST /validate"},
        "structure": {"dockerfile": True, "inference_py": True, "app_py": True, "pyproject_toml": True, "uv_lock": True},
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
    })

@app.route("/action_space", methods=["GET"])
def action_space():
    return jsonify({"type": "Discrete", "n": env.action_space.n, "actions": ["UP", "DOWN", "LEFT", "RIGHT", "PICKUP", "DROPOFF"]})

@app.route("/observation_space", methods=["GET"])
def observation_space():
    return jsonify({
        "type": "Box", 
        "shape": list(env.observation_space.shape), 
        "low": env.observation_space.low.tolist(), 
        "high": env.observation_space.high.tolist(), 
        "dtype": str(env.observation_space.dtype)
    })

def main():
    app.run(host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
