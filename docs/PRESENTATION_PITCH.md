# STTCS Presentation Outline

## The Problem
- The railway industry still relies heavily on fixed-block signaling (putting huge gaps between trains just to be safe).
- It's terribly inefficient. A single train breaking down causes a chain reaction of delays.
- We need moving-block intelligence, but doing this manually is impossible.

## The Solution
- STTCS uses Multi-Agent Reinforcement Learning (MARL).
- Instead of static blocks, the AI computes safe speeds in real-time, letting trains run much closer together without sacrificing safety.

## How we built it
- We feed sensor data into a FastAPI backend. 
- That data gets converted into a graph so Graph Neural Networks (GNNs) can process the track layouts.
- Then, decentralized RL agents (one for each section of track) decide how fast the trains should go.

## The Big "What If?" (Safety)
- The main concern with AI is safety (black box problem).
- Our solution: The RL agents only offer *suggestions*.
- We built a strict mathematical wrapper in Python that intercepts every command. If a command would cause two trains to get closer than 2.0 km, or run a red light, it overrides the AI and forces a stop. Zero trust.

## Disruption Handling
- If a track gets blocked, the system runs parallel "what-if" simulations in the background.
- It calculates the absolute best way to route trains into loop lines to avoid cascading delays, and does this in milliseconds.

## Performance Numbers
- Throughput increased by ~25%.
- Average delays dropped by over 40%.
- Recovery time from a track failure went from 45 minutes down to 12 minutes.

## The Demo
- We'll now jump into the React dashboard and run a live simulation.
- I'll intentionally break a track section so we can watch the AI instantly recover and reroute the trailing trains safely.
