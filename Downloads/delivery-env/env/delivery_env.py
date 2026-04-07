"""
Smart Last-Mile Delivery Optimization Environment.

A reinforcement learning environment simulating real-world delivery logistics
with fuel constraints, deadlines, traffic delays, and multi-vehicle routing.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Typed Pydantic Models
# ---------------------------------------------------------------------------

class DeliveryStatus(str, Enum):
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    FAILED = "failed"


class Delivery(BaseModel):
    """Represents a single delivery package."""
    delivery_id: int
    pickup_x: float = Field(..., description="Pickup X coordinate")
    pickup_y: float = Field(..., description="Pickup Y coordinate")
    dropoff_x: float = Field(..., description="Drop-off X coordinate")
    dropoff_y: float = Field(..., description="Drop-off Y coordinate")
    deadline: float = Field(..., description="Time-step deadline for delivery")
    priority: float = Field(default=1.0, ge=0.0, le=3.0, description="Priority multiplier")
    status: DeliveryStatus = DeliveryStatus.PENDING
    picked_up: bool = False
    traffic_delay: float = Field(default=0.0, ge=0.0, description="Traffic delay factor")
    weight: float = Field(default=1.0, ge=0.1, description="Package weight (affects fuel)")


class Vehicle(BaseModel):
    """Represents a delivery vehicle."""
    vehicle_id: int = 0
    x: float = 0.0
    y: float = 0.0
    fuel: float = 100.0
    max_fuel: float = 100.0
    speed: float = 1.0
    capacity: int = 10
    current_load: int = 0


class State(BaseModel):
    """Complete environment state."""
    vehicle_location: Tuple[float, float] = (0.0, 0.0)
    vehicles: List[Vehicle] = Field(default_factory=list)
    fuel: float = 100.0
    deliveries: List[Delivery] = Field(default_factory=list)
    time: float = 0.0
    max_time: float = 100.0
    grid_size: float = 20.0
    total_reward: float = 0.0
    done: bool = False
    active_vehicle_id: int = 0


class Action(BaseModel):
    """Agent action – select a delivery to pursue."""
    delivery_id: int = Field(..., description="ID of the delivery to pick up / drop off")
    vehicle_id: int = Field(default=0, description="Vehicle to use (multi-vehicle mode)")


# ---------------------------------------------------------------------------
# Environment Configuration
# ---------------------------------------------------------------------------

class EnvConfig(BaseModel):
    """Configuration for initialising DeliveryEnv."""
    num_deliveries: int = 3
    grid_size: float = 20.0
    max_time: float = 100.0
    initial_fuel: float = 100.0
    fuel_cost_per_unit: float = 0.5
    time_penalty_factor: float = 0.05
    late_penalty_factor: float = 5.0
    delivery_reward: float = 20.0
    proximity_reward_factor: float = 0.5
    traffic_enabled: bool = False
    traffic_max_delay: float = 3.0
    deadlines_enabled: bool = True
    num_vehicles: int = 1
    vehicle_speed: float = 1.0
    dynamic_orders: bool = False
    dynamic_order_prob: float = 0.15
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# DeliveryEnv – Core Environment
# ---------------------------------------------------------------------------

class DeliveryEnv:
    """
    OpenEnv-compatible delivery environment.

    API:
        reset()  -> State
        state()  -> State
        step(action: Action) -> Tuple[State, float, bool, dict]
    """

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self._rng = random.Random(self.config.seed)
        self._state: Optional[State] = None
        self._step_count: int = 0

    # ----- public API -----

    def reset(self) -> State:
        """Reset environment to initial conditions and return starting state."""
        cfg = self.config
        self._rng = random.Random(cfg.seed)
        self._step_count = 0

        # Create vehicles
        vehicles: List[Vehicle] = []
        for vid in range(cfg.num_vehicles):
            vehicles.append(
                Vehicle(
                    vehicle_id=vid,
                    x=self._rng.uniform(0, cfg.grid_size),
                    y=self._rng.uniform(0, cfg.grid_size),
                    fuel=cfg.initial_fuel,
                    max_fuel=cfg.initial_fuel,
                    speed=cfg.vehicle_speed,
                )
            )

        # Create deliveries
        deliveries = self._generate_deliveries(cfg.num_deliveries)

        primary = vehicles[0]
        self._state = State(
            vehicle_location=(primary.x, primary.y),
            vehicles=vehicles,
            fuel=primary.fuel,
            deliveries=deliveries,
            time=0.0,
            max_time=cfg.max_time,
            grid_size=cfg.grid_size,
            total_reward=0.0,
            done=False,
            active_vehicle_id=0,
        )
        return deepcopy(self._state)

    def state(self) -> State:
        """Return the current state (read-only copy)."""
        if self._state is None:
            return self.reset()
        return deepcopy(self._state)

    def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        """
        Execute one step.

        Returns:
            (next_state, reward, done, info)
        """
        if self._state is None:
            self.reset()
        assert self._state is not None

        s = self._state
        cfg = self.config
        info: Dict[str, Any] = {"events": []}

        # Resolve vehicle
        vid = action.vehicle_id
        if vid < 0 or vid >= len(s.vehicles):
            vid = 0
        vehicle = s.vehicles[vid]

        # Find target delivery
        target: Optional[Delivery] = None
        for d in s.deliveries:
            if d.delivery_id == action.delivery_id:
                target = d
                break

        if target is None or target.status in (DeliveryStatus.DELIVERED, DeliveryStatus.FAILED):
            # Invalid action – small penalty
            reward = -1.0
            s.time += 1.0
            info["events"].append("invalid_action")
            done = self._check_done()
            s.done = done
            s.total_reward += reward
            self._maybe_inject_dynamic_orders()
            return deepcopy(s), reward, done, info

        # ---- Determine destination ----
        if not target.picked_up:
            dest_x, dest_y = target.pickup_x, target.pickup_y
        else:
            dest_x, dest_y = target.dropoff_x, target.dropoff_y

        # ---- Movement ----
        dx = dest_x - vehicle.x
        dy = dest_y - vehicle.y
        dist = math.hypot(dx, dy)

        # Traffic delay
        effective_speed = vehicle.speed
        if cfg.traffic_enabled and target.traffic_delay > 0:
            effective_speed = max(0.2, vehicle.speed / (1.0 + target.traffic_delay))

        move_dist = min(dist, effective_speed)
        if dist > 0:
            vehicle.x += (dx / dist) * move_dist
            vehicle.y += (dy / dist) * move_dist

        # Fuel consumption
        fuel_used = move_dist * cfg.fuel_cost_per_unit * (1.0 + 0.1 * target.weight)
        vehicle.fuel -= fuel_used
        vehicle.fuel = max(0.0, vehicle.fuel)

        # Time
        time_step = 1.0 + (target.traffic_delay if cfg.traffic_enabled else 0.0)
        s.time += time_step
        self._step_count += 1

        # ---- Reward computation ----
        reward = 0.0

        # 1) Proximity reward (continuous shaping)
        remaining_dist = math.hypot(dest_x - vehicle.x, dest_y - vehicle.y)
        if remaining_dist < dist:
            reward += cfg.proximity_reward_factor * (dist - remaining_dist)

        # 2) Check arrival at destination
        arrived = remaining_dist < 0.5

        if arrived:
            if not target.picked_up:
                target.picked_up = True
                target.status = DeliveryStatus.IN_TRANSIT
                vehicle.current_load += 1
                info["events"].append(f"picked_up_{target.delivery_id}")
                reward += 2.0  # small pick-up bonus
            else:
                # Drop-off
                target.status = DeliveryStatus.DELIVERED
                vehicle.current_load = max(0, vehicle.current_load - 1)
                info["events"].append(f"delivered_{target.delivery_id}")

                # Delivery success reward
                reward += cfg.delivery_reward * target.priority

                # Late penalty
                if cfg.deadlines_enabled and s.time > target.deadline:
                    lateness = s.time - target.deadline
                    reward -= cfg.late_penalty_factor * lateness
                    info["events"].append(f"late_{target.delivery_id}")

        # 3) Fuel cost penalty
        reward -= fuel_used * 0.3

        # 4) Time penalty
        reward -= cfg.time_penalty_factor

        # 5) Out-of-fuel penalty
        if vehicle.fuel <= 0:
            reward -= 10.0
            info["events"].append("out_of_fuel")
            # Mark remaining carried deliveries as failed
            for d in s.deliveries:
                if d.status == DeliveryStatus.IN_TRANSIT:
                    d.status = DeliveryStatus.FAILED
                    info["events"].append(f"failed_{d.delivery_id}")

        # Sync primary state fields
        s.vehicle_location = (vehicle.x, vehicle.y)
        s.fuel = vehicle.fuel
        s.total_reward += reward

        # Dynamic orders injection
        self._maybe_inject_dynamic_orders()

        # Done check
        done = self._check_done()
        s.done = done

        return deepcopy(s), reward, done, info

    # ----- helpers -----

    def _generate_deliveries(self, n: int) -> List[Delivery]:
        cfg = self.config
        deliveries: List[Delivery] = []
        for i in range(n):
            traffic_delay = 0.0
            if cfg.traffic_enabled:
                traffic_delay = self._rng.uniform(0, cfg.traffic_max_delay)

            deadline = self._rng.uniform(cfg.max_time * 0.3, cfg.max_time * 0.9)
            if not cfg.deadlines_enabled:
                deadline = cfg.max_time * 10  # effectively infinite

            deliveries.append(
                Delivery(
                    delivery_id=i,
                    pickup_x=self._rng.uniform(0, cfg.grid_size),
                    pickup_y=self._rng.uniform(0, cfg.grid_size),
                    dropoff_x=self._rng.uniform(0, cfg.grid_size),
                    dropoff_y=self._rng.uniform(0, cfg.grid_size),
                    deadline=round(deadline, 1),
                    priority=round(self._rng.uniform(0.5, 2.0), 2),
                    traffic_delay=round(traffic_delay, 2),
                    weight=round(self._rng.uniform(0.5, 3.0), 2),
                )
            )
        return deliveries

    def _maybe_inject_dynamic_orders(self) -> None:
        if not self.config.dynamic_orders or self._state is None:
            return
        if self._rng.random() < self.config.dynamic_order_prob:
            new_id = max((d.delivery_id for d in self._state.deliveries), default=-1) + 1
            new_deliveries = self._generate_deliveries(1)
            new_deliveries[0].delivery_id = new_id
            self._state.deliveries.append(new_deliveries[0])

    def _check_done(self) -> bool:
        if self._state is None:
            return True
        s = self._state

        # All deliveries resolved
        all_resolved = all(
            d.status in (DeliveryStatus.DELIVERED, DeliveryStatus.FAILED)
            for d in s.deliveries
        )
        if all_resolved:
            return True

        # Time exceeded
        if s.time >= s.max_time:
            return True

        # All vehicles out of fuel
        if all(v.fuel <= 0 for v in s.vehicles):
            return True

        return False
