"""
Easy Task: 1 vehicle, 3 deliveries, no traffic, generous deadlines.
"""

from env.delivery_env import DeliveryEnv, EnvConfig


def create_task() -> DeliveryEnv:
    """Return a configured DeliveryEnv for the easy difficulty."""
    config = EnvConfig(
        num_deliveries=3,
        grid_size=15.0,
        max_time=80.0,
        initial_fuel=100.0,
        fuel_cost_per_unit=0.3,
        time_penalty_factor=0.02,
        late_penalty_factor=2.0,
        delivery_reward=20.0,
        proximity_reward_factor=0.6,
        traffic_enabled=False,
        deadlines_enabled=False,       # No deadline pressure
        num_vehicles=1,
        vehicle_speed=2.0,             # Faster vehicle
        dynamic_orders=False,
        seed=42,
    )
    return DeliveryEnv(config)


# Alias for OpenEnv spec compatibility
get_env = create_task
