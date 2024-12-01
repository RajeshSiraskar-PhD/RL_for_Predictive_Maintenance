# RL for Predictive Maintenance

## Implementation Notes
```
1. ALGO = 'PPO'
2. EARLY_DETECT_FACTOR = -0.125
3. r1 = 1; r2 = -4; r3 = -0.5
4. SAMPLING_RATE = 25
5. EPISODES = 300 k for C01, 200 k for NUAA W1
6. ADD_NOISE = 5*1e2
7. MAX_EPISODE_STEPS_FACTOR = 10 # MAX_EPISODE_STEPS = MAX_EPISODE_STEPS_FACTOR*records
8. BATCH_SIZE = 16
```

## What we have
1. Trained PdM agent: "Agent_PHM_C01" implies trained on C01
2. **THREE** training run results and
3. **THREE** trained agent 
4. TensorBoard plots
5. Trainining results as saved images and .csv results

## What we can demonstrate
1. Three runs -- so show REPEATABILITY
2. Trained PdM agent -- so show ROBUSTNESS or TRANSFERABILITY by testing on another set

## Presentation of Results
1. Trained model agents: C01
2. Show C01 Tool wear data - normal
3. Show with noise - Mention for robustness
4. Evaluate on C04 and C08
5. **Refresh untill reasonable REPLACEMENT**
6. Attempt an evaluation on NUAA W1
7. Repeat with C04 on rest i.e. C01 and C08 etc.
**Visualizations**:
9. Tool wear plot normal and with noise 
10. Tensorboard reward learning curve - self explainable
11. Tool replacement time reduction - self explainable
12. RUL improvement - will need explaining so show last 