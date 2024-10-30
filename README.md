# RL for Predictive Maintenance

## Implementation Notes
### Fixed parameters
1. Sample rate: 25 i.e. about 40 records
2. Gym Env. termination: ```max_episode_steps = MILLING_OPERATIONS_MAX```
3. Rewards: ```R1: +1.0, R2: -4.0, R3: -0.5```
4. Episodes: 30k
