# i-Rebalance: Personalized Vehicle Repositioning with Deep Reinforcement Learning

![GitHub repo size](https://img.shields.io/github/repo-size/Haoyang-Chen/I-Rebalance)
![GitHub stars](https://img.shields.io/github/stars/Haoyang-Chen/I-Rebalance?style=social)
![GitHub forks](https://img.shields.io/github/forks/Haoyang-Chen/I-Rebalance?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/Haoyang-Chen/I-Rebalance?style=social)
![License](https://img.shields.io/github/license/Haoyang-Chen/I-Rebalance)

## Introduction

Ride-hailing platforms have long grappled with the challenge of effectively balancing the demand and supply of vehicles. Traditional vehicle repositioning techniques often assume uniform behavior among drivers and deterministic relocation strategies. However, in this paper, we propose a more realistic and driver-centric approach to vehicle repositioning.

We present **i-Rebalance**, a novel personalized vehicle repositioning technique powered by deep reinforcement learning (DRL). Unlike conventional methods, i-Rebalance acknowledges that drivers possess unique cruising preferences and individual decision-making autonomy. By leveraging a deep reinforcement learning framework, i-Rebalance optimizes both supply-demand equilibrium and driver preference satisfaction.

## Key Features

- Incorporates unique driver cruising preferences into the repositioning strategy.
- Utilizes deep reinforcement learning to model and optimize driver decisions.
- Sequential repositioning approach with Grid Agent and Vehicle Agent for improved policy training.
- Enhanced supply-demand balance and driver satisfaction achieved simultaneously.

## Methodology

i-Rebalance employs a dual-agent DRL framework consisting of Grid Agent and Vehicle Agent. Grid Agent determines the optimal repositioning order of idle vehicles within the grid, while Vehicle Agent provides personalized recommendations to each vehicle based on their preferences and real-time context.

## Results

Our approach was validated through an on-field user study involving 99 real drivers. The evaluation of real-world trajectory data demonstrated significant improvements:
- Driver acceptance rate increased by 38.07%
- Total driver income improved by 9.97%

