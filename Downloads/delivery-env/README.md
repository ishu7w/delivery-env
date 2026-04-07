---
title: Smart Last-Mile Delivery Env
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🚚 Smart Last-Mile Delivery Optimization Environment

An **OpenEnv-compatible** reinforcement learning environment that simulates real-world last-mile delivery logistics. An AI agent must optimise delivery routing under constraints including fuel limits, time deadlines, traffic delays, package priorities, and multi-vehicle coordination.

---

## 📋 Problem Description

A fleet of delivery vehicles must complete a set of package deliveries across a 2D grid city. Each delivery has:

* A **pickup** and **drop-off** location
* A **deadline** by which it should be delivered
* A **priority** level (higher = more important)
* A **weight** affecting fuel consumption
* A **traffic delay** factor slowing movement

The agent must decide, at each time step, **which delivery to pursue next** (and with which vehicle in multi-vehicle mode) to maximise total reward while managing fuel and time budgets.

---

## 🧠 State Space

| Field | Type | Description |
| --- | --- | --- |
| `vehicle_location` | `(float, float)` | Active vehicle's (x, y) coordinates |
| `vehicles` | `List[Vehicle]` | All vehicles with id, position, fuel, speed |
| `fuel` | `float` | Active vehicle's remaining fuel |
| `deliveries` | `List[Delivery]` | All deliveries with locations, status, deadlines |
| `time` | `float` | Current simulation time |
| `max_time` | `float` | Maximum allowed time |
| `grid_size` | `float` | Size of the city grid |
| `total_reward` | `float` | Accumulated reward so far |
| `done` | `bool` | Whether the episode has ended |

### Delivery Object

| Field | Type | Description |
| --- | --- | --- |
| `delivery_id` | `int` | Unique identifier |
| `pickup_x/y` | `float` | Pickup coordinates |
| `dropoff_x/y` | `float` | Drop-off coordinates |
| `deadline` | `float` | Time deadline |
| `priority` | `float` | Priority multiplier (0.5–2.0) |
| `status` | `str` | `pending` / `in_transit` / `delivered` / `failed` |
| `picked_up` | `bool` | Whether package has been picked up |
| `traffic_delay` | `float` | Traffic penalty factor |
| `weight` | `float` | Package weight (affects fuel) |

---

## 🎮 Action Space

| Field | Type | Description |
| --- | --- | --- |
| `delivery_id` | `int` | ID of the delivery to pursue |
| `vehicle_id` | `int` | Vehicle to use (default: 0) |

---

## 🏆 Reward Function

The reward provides **continuous feedback** (not binary) and includes:

| Component | Effect | Description |
| --- | --- | --- |
| ✅ Delivery success | **+20 × priority** | Reward for completing a delivery |
| 📦 Pickup bonus | **+2.0** | Small reward for picking up a package |
| 📍 Proximity reward | **+0.3–0.6 × distance_reduced** | Continuous shaping for moving closer |
| ⏰ Late penalty | **−5–8 × lateness** | Penalty proportional to how late |
| ⛽ Fuel cost | **−fuel_used × 0.3** | Penalty for fuel consumption |
| 🕐 Time penalty | **−0.02–0.08** | Small per-step time cost |
| 🚫 Out of fuel | **−10.0** | Heavy penalty for running out |
| ❌ Invalid action | **−1.0** | Penalty for targeting completed deliveries |

---

## 🧪 Task Difficulty Levels

### Easy
* 1 vehicle, 3 deliveries
* No traffic delays, no deadlines
* Large fuel tank, fast vehicle
* Grid: 15×15

### Medium
* 1 vehicle, 6 deliveries
* Traffic delays enabled
* Tight deadlines enforced
* Grid: 20×20

### Hard
* 2 vehicles, 8+ deliveries
* Heavy traffic, strict deadlines
* Dynamic orders appear mid-episode
* Limited fuel, larger grid (25×25)

---

## 🚀 Setup & Installation

### Local Setup

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t delivery-env .
docker run -p 7860:7860 delivery-env
```

---

## ▶️ Running Inference

### With Heuristic Agent (no API key needed)

```bash
python inference.py
```

### With LLM Agent

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export HF_TOKEN="hf_your_token_here"
python inference.py
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/reset` | Reset environment, return initial state |
| `GET` | `/state` | Return current state |
| `POST` | `/step` | Execute action `{"delivery_id": int, "vehicle_id": int}` |
| `GET` | `/health` | Health check |

### Example

```bash
# Reset
curl http://localhost:7860/reset

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"delivery_id": 0}'

# State
curl http://localhost:7860/state
```

---

## 📊 Sample Output

```
[START] task=easy env=delivery_env model=mistralai/Mistral-Small-24B-Instruct-2501
[STEP] step=1 action=deliver(0) reward=0.52 done=false error=null
[STEP] step=2 action=deliver(1) reward=10.30 done=false error=null
[STEP] step=3 action=deliver(2) reward=9.85 done=true error=null
[END] success=true steps=3 score=0.823 rewards=0.52,10.30,9.85

[START] task=medium env=delivery_env model=mistralai/Mistral-Small-24B-Instruct-2501
[STEP] step=1 action=deliver(0) reward=0.41 done=false error=null
[END] success=true steps=12 score=0.689 rewards=0.41,...

[START] task=hard env=delivery_env model=mistralai/Mistral-Small-24B-Instruct-2501
[STEP] step=1 action=deliver(2) reward=0.38 done=false error=null
[END] success=false steps=25 score=0.543 rewards=0.38,...
```

---

## 📁 Project Structure

```
delivery-env/
├── env/
│   ├── __init__.py
│   └── delivery_env.py      # Core environment + Pydantic models
├── tasks/
│   ├── __init__.py
│   ├── easy.py
│   ├── medium.py
│   └── hard.py
├── graders/
│   ├── __init__.py
│   ├── easy_grader.py
│   ├── medium_grader.py
│   └── hard_grader.py
├── inference.py              # Main inference script
├── app.py                    # FastAPI server
├── openenv.yaml              # OpenEnv configuration
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Variables

| Variable | Description |
| --- | --- |
| `API_BASE_URL` | The API endpoint for the LLM |
| `MODEL_NAME` | The model identifier to use for inference |
| `HF_TOKEN` | Your Hugging Face / API key |

---

## ⚙️ Technical Constraints

* Inference runtime: < 20 minutes
* Resource requirements: 2 vCPU, 8 GB RAM
* Container port: 7860 (Hugging Face Spaces compatible)

---

## 📄 License

MIT License