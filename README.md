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

to train a model:

```
cd simulator/train
python Disp_Seq_A2C.py --cuda
```

## Data Structure

```
data							
│
├── AccurateMap.csv				
├── Node.csv
├── NodeIDList.txt
├── questionaire.csv
│
├── driver_clustering
│   ├── center0.csv
│   ├── center1.csv
│   ├── center2.csv
│   ├── center3.csv
│   ├── center4.csv
│   └── clustering_outcome.csv
│
├── Order_List
│   ├── Order_08_03.csv
│   ├── Order_08_04.csv
│   ├── Order_08_05.csv
│   ├── Order_08_06.csv
│   ├── Order_08_09.csv
│   └── Order_08_10.csv
│
├── PreData
│   ├── clustering_outcome.csv
│   ├── DriverFamilarity.csv
│   ├── home_info.csv
│   ├── poi_location.pkl
│   ├── Pre_Class0_o.pt
│   ├── Pre_Class0.pt
│   ├── Pre_Class1_o.pt
│   ├── Pre_Class1.pt
│   ├── Pre_Class2_o.pt
│   ├── Pre_Class2.pt
│   ├── Pre_Class3_o.pt
│   ├── Pre_Class3.pt
│   ├── Pre_Class4_o.pt
│   ├── Pre_Class4.pt
│   └── RidePreference.csv
│
└── Vehicle_List
    ├── Vehicle_List8_03.csv
    ├── Vehicle_List8_04.csv
    ├── Vehicle_List8_05.csv
    ├── Vehicle_List8_06.csv
    ├── Vehicle_List8_09.csv
    └── Vehicle_List8_10.csv

```

