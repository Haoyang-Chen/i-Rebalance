# i-Rebalance: Personalized Vehicle Repositioning with Deep Reinforcement Learning

Ride-hailing platforms have long grappled with the challenge of effectively balancing the demand and supply of vehicles. Traditional vehicle repositioning techniques often assume uniform behavior among drivers and deterministic relocation strategies. However, in this paper, we propose a more realistic and driver-centric approach to vehicle repositioning.

We present **i-Rebalance**, a novel personalized vehicle repositioning technique powered by deep reinforcement learning (DRL). Unlike conventional methods, i-Rebalance acknowledges that drivers possess unique cruising preferences and individual decision-making autonomy. By leveraging a **sequential vehicle repositioning framework with dual DRL agents**, i-Rebalance optimizes both supply-demand equilibrium and driver preference satisfaction.

## Key Features

- Incorporates unique driver cruising preferences into the repositioning strategy.
- Using a light-weight LSTM to model driver cruising preference.
- A driver decision making module based on survey data.
- Utilizes sequential vehicle repositioning framework with dual DRL agents to solve the problem of sequence and reposition simultaneously.
- Enhanced supply-demand balance and driver satisfaction achieved simultaneously.

## Methodology

i-Rebalance employs a dual-agent DRL framework consisting of Grid Agent and Vehicle Agent. Grid Agent determines the optimal repositioning order of idle vehicles within the grid, while Vehicle Agent provides personalized recommendations to each vehicle based on their preferences and real-time context.

## Installation 

```
cd I-Rebalance
conda env create -f environment.yaml  
```

