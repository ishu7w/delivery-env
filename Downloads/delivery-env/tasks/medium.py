"""
Medium Task: 1 vehicle, 6 deliveries, traffic delays, tight deadlines.
"""

from env.delivery_env import DeliveryEnv, EnvConfig


def create_task() -> DeliveryEnv:
    """Return a configured DeliveryEnv for medium difficulty."""
    config = EnvConfig(
        num_deliveries=6,
        grid_size=20.0,
        max_time=100.0,
        initial_fuel=90.0,
        fuel_cost_per_unit=0.5,
        time_penalty_factor=0.05,
        late_penalty_factor=5.0,
        delivery_reward=20.0,
        proximity_reward_factor=0.4,
        traffic_enabled=True,
        traffic_max_delay=2.5,
        deadlines_enabled=True,
        num_vehicles=1,
        vehicle_speed=1.5,
        dynamic_orders=False,
        seed=123,
    )
    return DeliveryEnv(config)


# Alias for OpenEnv spec compatibility
get_env = create_task
