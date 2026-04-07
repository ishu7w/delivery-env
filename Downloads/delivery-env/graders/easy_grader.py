"""
Easy Grader – runs an agent on the easy task and returns a normalised [0, 1] score.

Normalization offset: -50.0  (must match inference.py TASK_OFFSETS["easy"])
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Callable, Dict, Optional

from env.delivery_env import Action, DeliveryEnv, DeliveryStatus, State


# ---------------------------------------------------------------------------
# Deadline-aware heuristic
# Matches the heuristic_action() used in inference.py so that grader baseline
# scores and inference baseline scores are directly comparable.
# ---------------------------------------------------------------------------

def _deadline_aware_action(state: State) -> Optional[Action]:
    """
    Deadline-aware + nearest-pending heuristic.
    Prioritises urgent deliveries and prefers finishing in-transit packages.
    Identical logic to heuristic_action() in inference.py.
    """
    vx, vy = state.vehicle_location
    best_id: Optional[int] = None
    best_score = float("inf")

    for d in state.deliveries:
        if d.status in (DeliveryStatus.DELIVERED, DeliveryStatus.FAILED):
            continue
        if d.picked_up:
            tx, ty = d.dropoff_x, d.dropoff_y
        else:
            tx, ty = d.pickup_x, d.pickup_y

        dist = math.hypot(tx - vx, ty - vy)
        time_left = max(0.1, d.deadline - state.time)
        # Lower score = preferred
        score = dist / (d.priority + 0.1) + time_left * 0.3
        if d.picked_up:
            score -= 5.0  # prefer finishing in-transit deliveries

        if score < best_score:
            best_score = score
            best_id = d.delivery_id

    if best_id is not None:
        return Action(delivery_id=best_id)
    return None


# ---------------------------------------------------------------------------
# LLM-based agent
# ---------------------------------------------------------------------------

def _build_llm_agent(client: Any, model: str) -> Callable[[State], Optional[Action]]:
    """Return a function that queries an LLM to choose the next action."""

    def _agent(state: State) -> Optional[Action]:
        # Build a concise prompt
        deliveries_info = []
        for d in state.deliveries:
            if d.status in (DeliveryStatus.DELIVERED, DeliveryStatus.FAILED):
                continue
            deliveries_info.append({
                "id": d.delivery_id,
                "pickup": [round(d.pickup_x, 1), round(d.pickup_y, 1)],
                "dropoff": [round(d.dropoff_x, 1), round(d.dropoff_y, 1)],
                "deadline": d.deadline,
                "priority": d.priority,
                "picked_up": d.picked_up,
                "traffic_delay": d.traffic_delay,
                "weight": d.weight,
            })

        if not deliveries_info:
            return None

        prompt = (
            "You are a delivery route optimizer. Given the current state, choose the best "
            "delivery_id to pursue next. Respond with ONLY a JSON object: {\"delivery_id\": <int>}\n\n"
            f"Vehicle location: ({round(state.vehicle_location[0],1)}, {round(state.vehicle_location[1],1)})\n"
            f"Fuel remaining: {round(state.fuel, 1)}\n"
            f"Current time: {round(state.time, 1)} / {round(state.max_time, 1)}\n"
            f"Pending deliveries:\n{json.dumps(deliveries_info, indent=2)}\n\n"
            "Reply ONLY with the JSON object."
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
                temperature=0.1,
            )
            text = response.choices[0].message.content.strip()
            # Extract JSON from response
            if "{" in text:
                json_str = text[text.index("{"):text.rindex("}") + 1]
                data = json.loads(json_str)
                return Action(delivery_id=int(data["delivery_id"]))
        except Exception:
            pass
        # Fallback to deadline-aware heuristic
        return _deadline_aware_action(state)

    return _agent


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grade(
    env: Optional[DeliveryEnv] = None,
    agent_fn: Optional[Callable[[State], Optional[Action]]] = None,
    client: Any = None,
    model: str = "",
    max_steps: int = 200,
) -> float:
    """
    Run the agent on the easy task and return a score in [0.0, 1.0].

    Score normalisation:
    - theoretical_max = delivery_reward * sum(priorities) + pickup bonuses
    - theoretical_min = -(late_penalty * max_time * num_deliveries) - 50.0
    - offset 50.0 matches TASK_OFFSETS["easy"] in inference.py
    - score = clamp((reward - min) / (max - min), 0, 1)
    """
    if env is None:
        from tasks.easy import create_task
        env = create_task()

    if agent_fn is None:
        if client is not None and model:
            agent_fn = _build_llm_agent(client, model)
        else:
            agent_fn = _deadline_aware_action

    state = env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        action = agent_fn(state)
        if action is None:
            break
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    # Normalisation bounds — offset MUST stay in sync with inference.py TASK_OFFSETS["easy"]
    cfg = env.config
    total_priority = sum(d.priority for d in env.state().deliveries)
    theoretical_max = cfg.delivery_reward * total_priority + 2.0 * cfg.num_deliveries
    theoretical_min = -(cfg.late_penalty_factor * cfg.max_time * cfg.num_deliveries) - 50.0

    score = (total_reward - theoretical_min) / (theoretical_max - theoretical_min + 1e-8)
    return round(max(0.0, min(1.0, score)), 4)