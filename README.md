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

<span style="color:red;">Due to file size limitation, some of the data are not available, we define their structures here</span>

```
data							
│
├── AccurateMap.csv								min distance between each node
├── Node.csv											location of each node
├── NodeIDList.txt								ID list of valid nodes
├── questionaire.csv							survey outcome
│
├── Order_List										Order info of one day
│   ├── Order_08_03.csv					
│   ├── Order_08_04.csv
│   ├── Order_08_05.csv
│   ├── Order_08_06.csv
│   ├── Order_08_09.csv
│   └── Order_08_10.csv
│
├── PreData											
│   ├── clustering_outcome.csv		driver ID and cluster mapping
│   ├── DriverFamilarity.csv			driver's familarity to grid
│   ├── home_info.csv							driver's home location
│   ├── poi_location.pkl					grid's POI info		
│   ├── Pre_Class0.pt							pre_trained preference model
│   ├── Pre_Class1.pt
│   ├── Pre_Class2.pt
│   ├── Pre_Class3.pt
│   ├── Pre_Class4.pt
│   └── RidePreference.csv				collective preference
│
└── Vehicle_List									vehicles' ID & first location
    ├── Vehicle_List8_03.csv
    ├── Vehicle_List8_04.csv
    ├── Vehicle_List8_05.csv
    ├── Vehicle_List8_06.csv
    ├── Vehicle_List8_09.csv
    └── Vehicle_List8_10.csv
```

### AccurateMap.csv:

Scalar at **Column i Row j** indicates the distance (km) between node ID **i** and node ID **j**

### Node.csv:

**id**: id of node

**lat**: latitude of node

**lon**: longitude of node

| id   | lat        | lon         |
| ---- | ---------- | ----------- |
| 49   | 30.6971113 | 104.0789935 |

### NodeIDList.txt

Valid nodes' ID list, only when a node is reachable is it valid. 

```
49
50
83
84
85
86
87
```

### Order_XX_XX.csv

**ID**: id of order

**Start_time**: start time of order, time follows the definition in simulator

**End_time**: end time of order

**PointS_Longitude**: longitude of order start point

**PointS_Latitude**: latitude of order end point

**PointE_Longitude**: longitude of order end point

**PointE_Latitude**: latitude of order end point

**NodeS**: order start node id

**NodeE**: order end node id

| ID   | Start_time | End_time   | PointS_Longitude | PointS_Latitude | PointE_Longitude | PointE_Latitude | NodeS | NodeE |
| ---- | ---------- | ---------- | ---------------- | --------------- | ---------------- | --------------- | ----- | ----- |
| 4768 | 1407020491 | 1407020982 | 104.106222       | 30.65969        | 104.096488       | 30.68178        | 52579 | 78075 |

### Vehicle_ListX_XX.csv

**DriverID**: id of vehicle

**Start_time**: start working time of driver on that day

**NodeS**: start working node id of driver on that day

| DriverID | Start_time          | NodeS |
| -------- | ------------------- | ----- |
| 0        | 2014/08/03 07:00:53 | 52579 |

### clustering_outcome.csv

each row's first column is driver's id, second column is driver's cluster

| 0    | 1    |
| ---- | ---- |
| 1    | 0    |
| 2    | 4    |
| 3    | 4    |

### DriverFamilarity.csv & RidePreference.csv

Each row's index is driver's id, scalar at column **i** is driver's familarity to grid **i**

| 0.03 | 0.04 | 0.07 | 0.14 | 0.18 | 0.13 | 0.08 | 0.22 | 0.11 | 0.07 | 0.16 | 0.08 | 0.2  | 0.13 | 0.26 | 0.21 | 0.2  | 0.14 | 0.12 | 0.18 | 0.14 | 0.22 | 0.15 | 0.32 | 0.48 | 0.54 | 0.32 | 0.05 | 0.05 | 0.03 | 0.16 | 0.34 | 0.53 | 0.55 | 0.57 | 0.76 | 0.1  | 0.16 | 0.15 | 0.35 | 0.24 | 0.55 | 0.43 | 0.4  | 0.65 | 0.17 | 0.27 | 0.14 | 0.35 | 0.28 | 0.31 | 0.24 | 0.31 | 0.36 | 0.09 | 0.11 | 0.1  | 0.12 | 0.1  | 0.15 | 0.29 | 0.27 | 0.26 | 0.06 | 0.09 | 0.15 | 0.07 | 0.07 | 0.11 | 0.2  | 0.12 | 0.14 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0.07 | 0.14 | 0.52 | 0.24 | 0.33 | 0.2  | 0.05 | 0.15 | 0.16 | 0.26 | 0.41 | 0.37 | 0.62 | 0.31 | 0.23 | 0.23 | 0.13 | 0.04 | 0.2  | 0.32 | 0.35 | 0.48 | 0.28 | 0.34 | 0.47 | 0.13 | 0.3  | 0.23 | 0.22 | 0.08 | 0.56 | 0.76 | 0.75 | 0.44 | 0.85 | 0.23 | 0.56 | 0.35 | 0.25 | 0.7  | 0.52 | 0.6  | 0.39 | 0.23 | 0.25 | 0.39 | 0.34 | 0.57 | 0.85 | 0.65 | 0.35 | 0.25 | 0.13 | 0.26 | 0.53 | 1.2  | 1.37 | 0.62 | 0.4  | 0.59 | 0.35 | 0.23 | 0.47 | 1.09 | 1.86 | 0.67 | 0.22 | 0.41 | 0.85 | 0.47 | 0.14 | 0.07 |

### Home_info.csv

**ID**: driver's id

**Home_Longitude**: longitude of driver's home

**Home_Latitude**: latitude of driver's home

| ID   | Home_Longitude | Home_Latitude |
| ---- | -------------- | ------------- |
| 0    | 104.09934      | 30.65652      |

