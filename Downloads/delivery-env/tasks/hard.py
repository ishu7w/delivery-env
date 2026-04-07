"""
Hard Task: 2 vehicles, 8 deliveries, traffic, fuel constraints,
dynamic incoming orders, randomness.
"""

from env.delivery_env import DeliveryEnv, EnvConfig


def create_task() -> DeliveryEnv:
    """Return a configured DeliveryEnv for hard difficulty."""
    config = EnvConfig(
        num_deliveries=8,
        grid_size=25.0,
        max_time=120.0,
        initial_fuel=70.0,
        fuel_cost_per_unit=0.7,
        time_penalty_factor=0.08,
        late_penalty_factor=8.0,
        delivery_reward=25.0,
        proximity_reward_factor=0.3,
        traffic_enabled=True,
        traffic_max_delay=4.0,
        deadlines_enabled=True,
        num_vehicles=2,
        vehicle_speed=1.2,
        dynamic_orders=True,
        dynamic_order_prob=0.15,
        seed=None,  # Random each run for true difficulty
    )
    return DeliveryEnv(config)


# Alias for OpenEnv spec compatibility
get_env = create_task
