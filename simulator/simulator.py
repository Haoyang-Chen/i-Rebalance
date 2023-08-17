import os
import sys
import random
import re
import copy
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import datetime as dt
import torch
import statistics
from math import radians, cos, sin, asin, sqrt
from datetime import datetime, timedelta

from tqdm import tqdm
from objects.objects import Cluster, Order, Vehicle
from config.setting import *
from preprocessing.readfiles import *
from haversine import haversine
from sklearn.preprocessing import minmax_scale

NEW_PRE = True
RANGE = 9
HIDDEN_SIZE = 64

if torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')
np.set_printoptions(threshold=np.inf)


class Simulation(object):
    def __init__(self, ClusterMode, DemandPredictionMode,
                 DispatchMode, VehiclesNumber, TimePeriods, LocalRegionBound,
                 SideLengthMeter, VehiclesServiceMeter,
                 NeighborCanServer, FocusOnLocalRegion):

        self.step = 0
        self.reject = 0
        self.StayNum = 0
        self.DispatchModule = None
        self.DemandPredictorModule = None

        self.OrderNum = 0
        self.RejectNum = 0
        self.DispatchNum = 0
        self.TotallyDispatchCost = 0
        self.SumOrderValue = 0
        self.AutoOrderValue = 0
        self.AVGreward = 0
        self.TotallyWaitTime = 0
        self.TotallyUpdateTime = dt.timedelta()
        self.TotallyRewardTime = dt.timedelta()
        self.TotallyNextStateTime = dt.timedelta()
        self.TotallyLearningTime = dt.timedelta()
        self.TotallyDispatchTime = dt.timedelta()
        self.TotallyMatchTime = dt.timedelta()
        self.TotallyDemandPredictTime = dt.timedelta()

        self.Clusters = None
        self.Orders = None
        self.OrdersbyTime = None
        self.RawOrders = None
        self.Vehicles = None
        self.RawVehicles = None
        self.Map = None
        self.Node = None
        self.NodeIDList = None

        self.DriverClusteringInst = None
        self.DriverClusteringData = None

        self.NodeID2Cluseter = {}
        self.NodeID2NodesLocation = {}
        self.TransitionTempPool = []

        self.MapWestBound = LocalRegionBound[0]
        self.MapEastBound = LocalRegionBound[1]
        self.MapSouthBound = LocalRegionBound[2]
        self.MapNorthBound = LocalRegionBound[3]

        self.ClusterMode = ClusterMode
        self.DispatchMode = DispatchMode
        self.VehiclesNumber = VehiclesNumber
        self.TimePeriods = TimePeriods
        self.LocalRegionBound = LocalRegionBound
        self.SideLengthMeter = SideLengthMeter
        self.VehiclesServiceMeter = VehiclesServiceMeter
        self.ClustersNumber = None
        self.NumGrideWidth = None
        self.NumGrideHeight = None
        self.NeighborServerDeepLimit = None

        self.NeighborCanServer = NeighborCanServer
        self.FocusOnLocalRegion = FocusOnLocalRegion

        self.RealExpTime = None
        self.NowOrder = None
        self.step = None
        self.Episode = 10000

        self.CalculateTheScaleOfDivision()

        self.DemandPredictionMode = DemandPredictionMode
        self.SupplyExpect = None
        self.Supply_obs = None
        self.SE_mat = None
        self.SE_mat_new = None
        self.StayExpect = None
        self.DemandExpect = None
        self.DE_mat = None
        self.DE_mat_new = None
        self.IdleTimeCounter = None
        self.IdleCarsCounter = None
        self.Waiting_Time = None

        self.PreList = np.ones((3000, 9))
        self.ClassNetList = None
        self.ClassDict = None
        self.HomeLocation = None
        self.PoILocation = None
        self.VisFreDict = None
        self.PrePredict = np.zeros((1, 9))
        self.NaivePre=None
        return

    def Reset(self):
        print("Reset the experimental environment")

        self.StayNum = 0
        self.OrderNum = 0
        self.RejectNum = 0
        self.DispatchNum = 0
        self.TotallyDispatchCost = 0
        self.SumOrderValue = 0
        self.AutoOrderValue = 0
        self.AVGreward = 0
        self.TotallyWaitTime = 0
        self.TotallyUpdateTime = dt.timedelta()
        self.TotallyNextStateTime = dt.timedelta()
        self.TotallyLearningTime = dt.timedelta()
        self.TotallyDispatchTime = dt.timedelta()
        self.TotallyMatchTime = dt.timedelta()
        self.TotallyDemandPredictTime = dt.timedelta()

        self.TransitionTempPool.clear()
        self.RealExpTime = None
        self.NowOrder = None
        self.step = 0
        self.reject = 0
        self.PrePredict = np.zeros((1, 9))

        # Reset the Orders and Clusters and Vehicles
        # -------------------------------
        for i in self.Orders:
            i.Reset()

        for i in self.Clusters:
            i.Reset()

        for i in self.Vehicles:
            i.Reset()

        self.InitVehiclesIntoCluster()
        # -------------------------------
        return

    def InitVehiclesIntoCluster(self):
        print("Initialization Vehicles into Clusters or Grids")
        for i in self.Vehicles:
            while True:
                RandomNode = random.choice(range(len(self.Node)))
                if RandomNode in self.NodeID2Cluseter:
                    i.LocationNode = RandomNode
                    i.Cluster = self.NodeID2Cluseter[i.LocationNode]
                    i.Cluster.VehiclesArrivetime[i] = i.StartTime
                    i.DeliveryPoint = i.LocationNode
                    break

    def RoadCost(self, start, end):
        dist = self.Map[start][end]
        res = 30 * dist / 11
        return int(res // 1 + (res % 1 > 0))

    def RoadDistance(self, start, end):
        dist = self.Map[start][end]
        return dist

    def GetValue(self, dist):
        if dist <= 2:
            return 8.5 * 0.89
        else:
            if dist <= 10:
                return (8.5 + (dist - 2) * 2) * 0.89
        return (24.5 + (dist - 10) * 3) * 0.89

    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r * 1000

    def CalculateTheScaleOfDivision(self):
        EastWestSpan = self.LocalRegionBound[1] - self.LocalRegionBound[0]
        NorthSouthSpan = self.LocalRegionBound[3] - self.LocalRegionBound[2]

        AverageLongitude = (self.MapEastBound - self.MapWestBound) / 2
        AverageLatitude = (self.MapNorthBound - self.MapSouthBound) / 2

        self.NumGrideWidth = int(self.haversine(self.MapWestBound, AverageLatitude, self.MapEastBound,
                                                AverageLatitude) / self.SideLengthMeter + 1)
        self.NumGrideHeight = int(self.haversine(AverageLongitude, self.MapSouthBound, AverageLongitude,
                                                 self.MapNorthBound) / self.SideLengthMeter + 1)

        self.NeighborServerDeepLimit = int(
            (self.VehiclesServiceMeter - (0.5 * self.SideLengthMeter)) // self.SideLengthMeter)
        self.ClustersNumber = self.NumGrideWidth * self.NumGrideHeight

        print("----------------------------")
        print("Map extent", self.LocalRegionBound)
        print("The width of each grid", self.SideLengthMeter, "meters")
        print("Vehicle service range", self.VehiclesServiceMeter, "meters")
        print("Number of grids in east-west direction", self.NumGrideWidth)
        print("Number of grids in north-south direction", self.NumGrideHeight)
        print("Number of grids", self.ClustersNumber)
        print("----------------------------")
        return

    def CreateAllInstantiate(self):
        print("Read all files")
        if NEW_PRE:
            self.Node, self.NodeIDList, Orders, Vehicles, self.Map, \
            self.ClassNetList, self.ClassDict, self.HomeLocation, \
            self.PoILocation, self.VisFreDict, self.NaivePre = ReadAllFiles()
        else:
            self.Node, self.NodeIDList, Orders, Vehicles, self.Map, self.NaivePre = ReadAllFiles_OLD()

        print("Create Grids")
        self.Clusters = self.CreateGrid()

        # Construct NodeID to Cluseter map for Fast calculation
        NodeID = self.Node['id'].values
        for i in range(len(NodeID)):
            NodeID[i] = self.NodeIDList.index(NodeID[i])
        for i in NodeID:
            for j in self.Clusters:
                for k in j.Nodes:
                    if i == k[0]:
                        self.NodeID2Cluseter[i] = j

        print("Create Orders set")
        self.Orders = [
            Order(i[0], i[1], self.NodeIDList.index(i[2]), self.NodeIDList.index(i[3]), i[1] + PICKUPTIMEWINDOW, None,
                  None, None) for i in Orders]

        # Limit order generation area
        # -------------------------------
        if self.FocusOnLocalRegion == True:
            print("Remove out-of-bounds Orders")
            for i in self.Orders[:]:
                if self.IsOrderInLimitRegion(i) == False:
                    self.Orders.remove(i)
            for i in range(len(self.Orders)):
                self.Orders[i].ID = i
        # -------------------------------

        # Calculate the value of all orders in advance
        # -------------------------------
        print("Pre-calculated order value")
        for EachOrder in self.Orders:
            # EachOrder.OrderValue = self.RoadCost(EachOrder.PickupPoint, EachOrder.DeliveryPoint)
            EachOrder.OrderValue = self.Map[EachOrder.PickupPoint][EachOrder.DeliveryPoint]

        # -------------------------------

        # Select number of vehicles
        # -------------------------------
        Vehicles = Vehicles[:self.VehiclesNumber]
        # -------------------------------

        print("Create Vehicles set")
        self.Vehicles = [Vehicle(i[0], i[1], i[3], self.NodeIDList.index(i[2]), None, [], None) for i in Vehicles]
        self.InitVehiclesIntoCluster()

        self.OrdersbyTime = [[] for _ in range(109)]
        start_time = self.Orders[0].ReleasTime
        for order in self.Orders:
            step = (order.ReleasTime - start_time) // self.TimePeriods
            self.OrdersbyTime[step].append(order)
        return

    def CreateAllInstantiate_TEST(self, idx):
        print("Read all files")
        if NEW_PRE:
            self.Node, self.NodeIDList, Orders, Vehicles, self.Map, \
            self.ClassNetList, self.ClassDict, self.HomeLocation, \
            self.PoILocation, self.VisFreDict, self.NaivePre = ReadAllFiles_TEST(idx)
        else:
            self.Node, self.NodeIDList, Orders, Vehicles, self.Map, self.NaivePre = ReadAllFiles_OLD()

        print("Create Grids")
        self.Clusters = self.CreateGrid()

        # Construct NodeID to Cluseter map for Fast calculation
        NodeID = self.Node['id'].values
        for i in range(len(NodeID)):
            NodeID[i] = self.NodeIDList.index(NodeID[i])
        for i in NodeID:
            for j in self.Clusters:
                for k in j.Nodes:
                    if i == k[0]:
                        self.NodeID2Cluseter[i] = j

        print("Create Orders set")
        self.Orders = [
            Order(i[0], i[1], self.NodeIDList.index(i[2]), self.NodeIDList.index(i[3]), i[1] + PICKUPTIMEWINDOW, None,
                  None, None) for i in Orders]

        # Limit order generation area
        # -------------------------------
        if self.FocusOnLocalRegion == True:
            print("Remove out-of-bounds Orders")
            for i in self.Orders[:]:
                if self.IsOrderInLimitRegion(i) == False:
                    self.Orders.remove(i)
            for i in range(len(self.Orders)):
                self.Orders[i].ID = i
        # -------------------------------

        # Calculate the value of all orders in advance
        # -------------------------------
        print("Pre-calculated order value")
        for EachOrder in self.Orders:
            # EachOrder.OrderValue = self.RoadCost(EachOrder.PickupPoint, EachOrder.DeliveryPoint)
            EachOrder.OrderValue = self.Map[EachOrder.PickupPoint][EachOrder.DeliveryPoint]

        # -------------------------------

        # Select number of vehicles
        # -------------------------------
        Vehicles = Vehicles[:self.VehiclesNumber]
        # -------------------------------

        print("Create Vehicles set")
        self.Vehicles = [Vehicle(i[0], i[1], i[3], self.NodeIDList.index(i[2]), None, [], None) for i in Vehicles]
        self.InitVehiclesIntoCluster()

        self.OrdersbyTime = [[] for _ in range(109)]
        start_time = self.Orders[0].ReleasTime
        for order in self.Orders:
            step = (order.ReleasTime - start_time) // self.TimePeriods
            self.OrdersbyTime[step].append(order)
        return

    def IsOrderInLimitRegion(self, Order):
        if not Order.PickupPoint in self.NodeID2NodesLocation:
            return False
        if not Order.DeliveryPoint in self.NodeID2NodesLocation:
            return False

        return True

    def IsNodeInLimitRegion(self, TempNodeList):
        if TempNodeList[0][0] < self.LocalRegionBound[0] or TempNodeList[0][0] > self.LocalRegionBound[1]:
            return False
        elif TempNodeList[0][1] < self.LocalRegionBound[2] or TempNodeList[0][1] > self.LocalRegionBound[3]:
            return False

        return True

    def CreateGrid(self):
        NumGrideHeight = self.NumGrideHeight
        NumGride = self.NumGrideWidth * self.NumGrideHeight

        NodeLocation = self.Node[['lon', 'lat']].values.round(7)
        NodeID = self.Node['id'].values.astype('int64')

        # Select small area simulation
        # ----------------------------------------------------
        if self.FocusOnLocalRegion == True:
            NodeLocation = NodeLocation.tolist()
            NodeID = NodeID.tolist()

            TempNodeList = []
            for i in range(len(NodeLocation)):
                TempNodeList.append((NodeLocation[i], NodeID[i]))

            for i in TempNodeList[:]:
                if self.IsNodeInLimitRegion(i) == False:
                    TempNodeList.remove(i)

            NodeLocation.clear()
            NodeID.clear()

            for i in TempNodeList:
                NodeLocation.append(i[0])
                NodeID.append(i[1])

            NodeLocation = np.array(NodeLocation)
        # --------------------------------------------------

        NodeSet = {}
        for i in range(len(NodeID)):
            NodeSet[(NodeLocation[i][0], NodeLocation[i][1])] = self.NodeIDList.index(NodeID[i])

        # Build each grid
        # ------------------------------------------------------
        if self.FocusOnLocalRegion == True:
            TotalWidth = self.LocalRegionBound[1] - self.LocalRegionBound[0]
            TotalHeight = self.LocalRegionBound[3] - self.LocalRegionBound[2]
        else:
            TotalWidth = self.MapEastBound - self.MapWestBound
            TotalHeight = self.MapNorthBound - self.MapSouthBound

        IntervalWidth = TotalWidth / self.NumGrideWidth
        IntervalHeight = TotalHeight / self.NumGrideHeight

        AllGrid = [Cluster(i, [], [], 0, [], {}, []) for i in range(NumGride)]

        for key, value in NodeSet.items():
            NowGridWidthNum = None
            NowGridHeightNum = None

            for i in range(self.NumGrideWidth):
                if self.FocusOnLocalRegion == True:
                    LeftBound = (self.LocalRegionBound[0] + i * IntervalWidth)
                    RightBound = (self.LocalRegionBound[0] + (i + 1) * IntervalWidth)
                else:
                    LeftBound = (self.MapWestBound + i * IntervalWidth)
                    RightBound = (self.MapWestBound + (i + 1) * IntervalWidth)

                if key[0] > LeftBound and key[0] < RightBound:
                    NowGridWidthNum = i
                    break

            for i in range(self.NumGrideHeight):
                if self.FocusOnLocalRegion == True:
                    DownBound = (self.LocalRegionBound[2] + i * IntervalHeight)
                    UpBound = (self.LocalRegionBound[2] + (i + 1) * IntervalHeight)
                else:
                    DownBound = (self.MapSouthBound + i * IntervalHeight)
                    UpBound = (self.MapSouthBound + (i + 1) * IntervalHeight)

                if key[1] > DownBound and key[1] < UpBound:
                    NowGridHeightNum = i
                    break

            if NowGridWidthNum == None or NowGridHeightNum == None:
                print(key[0], key[1])
                raise Exception('error')
            else:
                AllGrid[self.NumGrideWidth * NowGridHeightNum + NowGridWidthNum].Nodes.append((value, (key[0], key[1])))
        # ------------------------------------------------------

        for i in AllGrid:
            for j in i.Nodes:
                self.NodeID2NodesLocation[j[0]] = j[1]

        # Add neighbors to each grid
        # ------------------------------------------------------
        for i in AllGrid:

            # Bound Check
            # ----------------------------
            UpNeighbor = True
            DownNeighbor = True
            LeftNeighbor = True
            RightNeighbor = True
            LeftUpNeighbor = True
            LeftDownNeighbor = True
            RightUpNeighbor = True
            RightDownNeighbor = True

            if i.ID >= self.NumGrideWidth * (self.NumGrideHeight - 1):
                UpNeighbor = False
                LeftUpNeighbor = False
                RightUpNeighbor = False
            if i.ID < self.NumGrideWidth:
                DownNeighbor = False
                LeftDownNeighbor = False
                RightDownNeighbor = False
            if i.ID % self.NumGrideWidth == 0:
                LeftNeighbor = False
                LeftUpNeighbor = False
                LeftDownNeighbor = False
            if (i.ID + 1) % self.NumGrideWidth == 0:
                RightNeighbor = False
                RightUpNeighbor = False
                RightDownNeighbor = False
            # ----------------------------

            # Add all neighbors
            # ----------------------------
            if UpNeighbor:
                i.Neighbor.append(AllGrid[i.ID + self.NumGrideWidth])
            if DownNeighbor:
                i.Neighbor.append(AllGrid[i.ID - self.NumGrideWidth])
            if LeftNeighbor:
                i.Neighbor.append(AllGrid[i.ID - 1])
            if RightNeighbor:
                i.Neighbor.append(AllGrid[i.ID + 1])
            if LeftUpNeighbor:
                i.Neighbor.append(AllGrid[i.ID + self.NumGrideWidth - 1])
            if LeftDownNeighbor:
                i.Neighbor.append(AllGrid[i.ID - self.NumGrideWidth - 1])
            if RightUpNeighbor:
                i.Neighbor.append(AllGrid[i.ID + self.NumGrideWidth + 1])
            if RightDownNeighbor:
                i.Neighbor.append(AllGrid[i.ID - self.NumGrideWidth + 1])
            # ----------------------------
        return AllGrid

    def Normaliztion_1D(self, arr):
        arrmax = arr.max()
        arrmin = arr.min()
        arrmaxmin = arrmax - arrmin
        result = []
        for x in arr:
            x = float(x - arrmin) / arrmaxmin
            result.append(x)

        return np.array(result)

    def WorkdayOrWeekend(self, day):
        if type(day) != type(0) or day < 0 or day > 6:
            raise Exception('input format error')
        elif day == 5 or day == 6:
            return "Weekend"
        else:
            return "Workday"

    def DemandPredictFunction(self,step):
        """
        Here you can implement your own order forecasting method
        to provide efficient and accurate help for Dispatch method
        """
        self.DemandExpect = torch.zeros(self.ClustersNumber)
        DE = torch.zeros(self.ClustersNumber)
        for order in self.OrdersbyTime[step+1]:
            if self.RealExpTime + self.TimePeriods <= order.ReleasTime < self.RealExpTime + 2 * self.TimePeriods:
                self.DemandExpect[self.NodeID2Cluseter[order.PickupPoint].ID] += 1
                cluster = self.NodeID2Cluseter[order.PickupPoint]
                rodedist = self.RoadDistance(order.PickupPoint, order.DeliveryPoint)
                value = self.GetValue(rodedist)
                cluster.potential += value
        for order in self.OrdersbyTime[step]:
            if self.RealExpTime <= order.ReleasTime < self.RealExpTime + self.TimePeriods:
                DE[self.NodeID2Cluseter[order.PickupPoint].ID] += 1

        self.DE_mat = DE.reshape((1, 1, self.NumGrideHeight, self.NumGrideWidth))
        trans = torch.nn.Conv2d(1, 1, (5, 5), stride=1, padding=2, bias=False)
        trans.weight.data = torch.Tensor([[[[1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1]]]])
        self.DE_mat = trans(self.DE_mat)
        self.DE_mat = self.DE_mat.view(-1, self.DE_mat.shape[2] * self.DE_mat.shape[3]).squeeze()
        return self.DE_mat

    def SupplyExpectFunction(self):
        """
        Calculate the number of idle Vehicles in the next time slot
        of each cluster due to the completion of the order
        """
        self.SupplyExpect = torch.zeros(self.ClustersNumber)
        SE = torch.zeros(self.ClustersNumber)
        for i in self.Clusters:
            for key, value in list(i.VehiclesArrivetime.items()):
                if value <= self.RealExpTime + self.TimePeriods:
                    self.SupplyExpect[i.ID] += 1
            SE[i.ID] += len(i.IdleVehicles)

        ##########################################################
        self.SE_mat = SE.reshape((1, 1, self.NumGrideHeight, self.NumGrideWidth))
        self.SE_mat_new = self.SupplyExpect.reshape((1, 1, self.NumGrideHeight, self.NumGrideWidth))
        trans = torch.nn.Conv2d(1, 1, (5, 5), stride=1, padding=2, bias=False)
        trans.weight.data = torch.tensor([[[[1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1]]]], dtype=torch.float32)
        self.SE_mat = trans(self.SE_mat)
        self.SE_mat = self.SE_mat.view(-1, self.SE_mat.shape[2] * self.SE_mat.shape[3]).squeeze()
        ##########################################################

        return

    def IdleTimeCounterFunction(self):
        self.IdleTimeCounter = torch.zeros(self.ClustersNumber)
        self.IdleCarsCounter = torch.zeros(self.ClustersNumber)
        for cluster in self.Clusters:
            for vehicle in cluster.IdleVehicles:
                self.IdleCarsCounter[cluster.ID] += 1
                self.IdleTimeCounter[cluster.ID] += vehicle.idle_start_time
        self.ITC_mat = self.IdleTimeCounter.reshape((1, 1, self.NumGrideHeight, self.NumGrideWidth))
        self.ICC_mat = self.IdleCarsCounter.reshape((1, 1, self.NumGrideHeight, self.NumGrideWidth))
        trans = torch.nn.Conv2d(1, 1, (5, 5), stride=1, padding=2, bias=False)
        trans.weight.data = torch.Tensor([[[[1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1],
                                            [1, 1, 1, 1, 1]]]])
        self.ITC_mat = trans(self.ITC_mat)
        self.ICC_mat = trans(self.ICC_mat)
        self.ITC_mat = self.ITC_mat.view(-1, self.ITC_mat.shape[2] * self.ITC_mat.shape[3]).squeeze()
        self.ICC_mat = self.ICC_mat.view(-1, self.ICC_mat.shape[2] * self.ICC_mat.shape[3]).squeeze()
        self.Waiting_Time = self.step - self.ITC_mat / (self.ICC_mat + 0.01)
        return self.Waiting_Time

    def MatchFunction(self):
        """
        Each matching module will match the orders that will occur within the current time slot. 
        The matching module will find the nearest idle vehicles for each order. It can also enable 
        the neighbor car search system to determine the search range according to the set search distance 
        and the size of the grid. It use dfs to find the nearest idle vehicles in the area.
        """

        # Count the number of idle vehicles before matching
        for i in self.Clusters:
            i.PerMatchIdleVehicles = len(i.IdleVehicles)
        while self.NowOrder.ReleasTime < self.RealExpTime + self.TimePeriods:
            if self.NowOrder.ID == self.Orders[-1].ID:
                break

            self.OrderNum += 1
            NowCluster = self.NodeID2Cluseter[self.NowOrder.PickupPoint]
            NowCluster.Orders.append(self.NowOrder)

            if len(NowCluster.IdleVehicles) or len(NowCluster.Neighbor):
                TempMin = None

                if len(NowCluster.IdleVehicles):

                    # Find a nearest car to match the current order
                    # --------------------------------------

                    for i in NowCluster.IdleVehicles:

                        TempRoadCost = self.RoadCost(i.LocationNode, self.NowOrder.PickupPoint)
                        if TempMin == None:
                            TempMin = (i, TempRoadCost, NowCluster)
                        elif TempRoadCost < TempMin[1]:
                            TempMin = (i, TempRoadCost, NowCluster)
                    # --------------------------------------
                # Neighbor car search system to increase search range
                elif self.NeighborCanServer and len(NowCluster.Neighbor):
                    TempMin = self.FindServerVehicleFunction(
                        NeighborServerDeepLimit=self.NeighborServerDeepLimit,
                        Visitlist={}, Cluster=NowCluster, TempMin=None, deep=0
                    )

                # When all Neighbor Cluster without any idle Vehicles
                if TempMin == None or TempMin[1] > PICKUPTIMEWINDOW:
                    self.RejectNum += 1
                    self.NowOrder.ArriveInfo = "Reject"
                else:
                    NowVehicle = TempMin[0]
                    self.NowOrder.PickupWaitTime = TempMin[1]
                    NowVehicle.Orders.append(self.NowOrder)

                    self.TotallyWaitTime += self.RoadCost(NowVehicle.LocationNode, self.NowOrder.PickupPoint)

                    ScheduleCost = self.RoadCost(NowVehicle.LocationNode, self.NowOrder.PickupPoint) + self.RoadCost(
                        self.NowOrder.PickupPoint, self.NowOrder.DeliveryPoint)

                    # Add a destination to the current vehicle
                    NowVehicle.DeliveryPoint = self.NowOrder.DeliveryPoint

                    # Delivery Cluster {Vehicle:ArriveTime}
                    self.Clusters[self.NodeID2Cluseter[self.NowOrder.DeliveryPoint].ID].VehiclesArrivetime[
                        NowVehicle] = self.RealExpTime + np.timedelta64(ScheduleCost * MINUTES)

                    # refresh idle_start_time
                    Date = self.Orders[0].ReleasTime.day
                    TimeStamp = (self.RealExpTime + np.timedelta64(ScheduleCost * MINUTES)).value / (1000000000) - 28800
                    NowVehicle.idle_start_time = self.Cal_TimePeriod(Date, TimeStamp) / 2  # 以step为单位

                    # delete now Cluster's recode about now Vehicle
                    TempMin[2].IdleVehicles.remove(NowVehicle)

                    self.NowOrder.ArriveInfo = "Success"

                    if NowVehicle.Auto:
                        self.AutoOrderValue += self.GetValue(self.NowOrder.OrderValue)
                        NowVehicle.Auto = False
            else:
                # None available idle Vehicles
                self.RejectNum += 1
                self.NowOrder.ArriveInfo = "Reject"

            # The current order has been processed and start processing the next order
            # ------------------------------
            self.NowOrder = self.Orders[self.NowOrder.ID + 1]
        return

    def FindServerVehicleFunction(self, NeighborServerDeepLimit, Visitlist, Cluster, TempMin, deep):
        """
        Use dfs visit neighbors and find nearest idle Vehicle
        """
        if deep > NeighborServerDeepLimit or Cluster.ID in Visitlist:
            return TempMin

        Visitlist[Cluster.ID] = True
        for i in Cluster.IdleVehicles:
            TempRoadCost = self.RoadCost(i.LocationNode, self.NowOrder.PickupPoint)
            if TempMin == None:
                TempMin = (i, TempRoadCost, Cluster)
            elif TempRoadCost < TempMin[1]:
                TempMin = (i, TempRoadCost, Cluster)

        if self.NeighborCanServer:
            for j in Cluster.Neighbor:
                TempMin = self.FindServerVehicleFunction(NeighborServerDeepLimit, Visitlist, j, TempMin, deep + 1)
        return TempMin

    def UpdateFunction(self):
        """
        Each time slot update Function will update each cluster
        in the simulator, processing orders and vehicles
        """
        self.StayExpect = np.zeros(self.ClustersNumber)
        for i in self.Clusters:
            # Records array of orders cleared for the last time slot
            i.Orders.clear()
            i.potential = 0
            for key, value in list(i.VehiclesArrivetime.items()):
                if value <= self.RealExpTime:
                    # update Order
                    if len(key.Orders):
                        key.Orders[0].ArriveOrderTimeRecord(self.RealExpTime)
                        # update Vehicle info
                        key.ArriveVehicleUpDate(self.step, key.Orders[0].OrderValue, i)
                    else:
                        key.ArriveVehicleUpDate(self.step, 0, i)
                    # update Cluster record
                    i.ArriveClusterUpDate(key)

        return

    def CalcOrderNum(self):
        order_num = np.zeros((18, 72))
        for order in self.Orders:
            time = order.ReleasTime
            step = time // self.TimePeriods
            cluster_id = self.NodeID2Cluseter[order.PickupPoint].ID
            order_num[step][cluster_id] += 1
        return order_num

    def Act2Cluster(self, action, cluster):
        """
        return a cluster given an action(0-8) and current cluster
        """
        rand_cluster = cluster.Neighbor[random.randrange(0, len(cluster.Neighbor))]
        try:
            if action <= 2:
                dest = self.Clusters[cluster.ID + self.NumGrideWidth - 1 + action]
            elif 2 < action <= 5:
                dest = self.Clusters[cluster.ID - 4 + action]
            elif 5 < action <= 8:
                dest = self.Clusters[cluster.ID - self.NumGrideWidth - 7 + action]
            else:
                raise 'wrong action'
        except:
            dest = rand_cluster

        if cluster.ID < self.NumGrideWidth and 5 < action <= 8:
            dest = rand_cluster
        if cluster.ID >= self.NumGrideWidth * (self.NumGrideHeight - 1) and action <= 2:
            dest = rand_cluster
        if cluster.ID % self.NumGrideWidth == 0 and action % 3 == 0:
            dest = rand_cluster
        if cluster.ID % self.NumGrideWidth == self.NumGrideWidth - 1 and action % 3 == 2:
            dest = rand_cluster

        return dest

    def Getidx(self, Cluster_ID, Goal_ID):
        index = Cluster_ID - Goal_ID
        if abs(index) > 10:
            return 4
        C2ADict = {0: 4, -1: 5, 1: 3, 9: 7, -9: 1, 10: 6, 8: 8, -8: 0, -10: 2}

        return C2ADict[index]

    def move(self, vehicle, action, cluster, pre_idxs, idxs):
        vehicle.Cluster = self.Act2Cluster(action, vehicle.Cluster)

        num_supply = self.SupplyExpect[vehicle.Cluster.ID]
        num_demand = self.DemandExpect[vehicle.Cluster.ID]
        if num_supply > num_demand:
            expected_income = vehicle.Cluster.potential / num_supply
        else:
            expected_income = vehicle.Cluster.potential / num_demand

        idx = self.Getidx(cluster.ID, vehicle.Cluster.ID)
        ranking = minmax_scale(pre_idxs, (1, 9))
        ranking = [10 - x for x in ranking]
        ranking = ranking[idx]

        result = -0.3892 * ranking + 0.2932 * expected_income - 1.3959 + 1.8379 * vehicle.Obey
        vehicle.Auto = False
        incentive = 0
        if result < 0:
            vehicle.Auto = True
            incentive += (0 - result) / 0.2932
            idxs = idxs[0:int(len(idxs) / 2):1]
            self.reject += 1
            np.random.shuffle(idxs)
            vehicle.Cluster = self.Act2Cluster(idxs[0], cluster)

        if vehicle.Cluster == cluster:
            self.StayExpect[cluster.ID] += 1

        RandomNode = random.choice(vehicle.Cluster.Nodes)
        RandomNode = RandomNode[0]
        vehicle.DeliveryPoint = RandomNode

        ScheduleCost = self.RoadCost(vehicle.LocationNode, RandomNode)
        self.TotallyDispatchCost += self.RoadDistance(vehicle.LocationNode, RandomNode)

        vehicle.Cluster.VehiclesArrivetime[vehicle] = min(self.RealExpTime + np.timedelta64(
            MINUTES * ScheduleCost), self.RealExpTime + self.TimePeriods)

        self.SupplyExpect[vehicle.Cluster.ID] += 1
        return incentive

    def Cal_TimePeriod(self, Date, Timestamp):
        base_time = 1407016800 + (Date - 3) * 86400
        t = int((Timestamp - base_time) // 300)
        return t

    def Cal_HomeDis(self, Cluster_ID, Vehicle_ID):
        row = self.NumGrideHeight - 1 - int(Cluster_ID) // self.NumGrideWidth
        col = int(Cluster_ID) % self.NumGrideWidth

        lo_div = (104.1128 - 104.0178) / self.NumGrideWidth
        la_div = (30.6974 - 30.6218) / self.NumGrideHeight

        now_la = 30.6974 - la_div * row
        now_lo = 104.0178 + lo_div * col

        Home_Dis = haversine((now_la, now_lo), (self.HomeLocation[Vehicle_ID, 1], self.HomeLocation[Vehicle_ID, 0]))
        return Home_Dis

    def PrepareInput(self, Idle_vehicle_list, Date, Day, TimePeriod, traffic_input):
        Input_matrix = []
        for vehicle in Idle_vehicle_list:
            Vehicle_ID = vehicle.ID
            Cluster_ID = vehicle.Cluster.ID
            HomeDis = self.Cal_HomeDis(Cluster_ID, Vehicle_ID)
            try:
                Familarity = self.VisFreDict[Vehicle_ID, Cluster_ID]
            except:
                Familarity = 0

            if vehicle.idle_start_time == 0:
                IdleTime = 1
            else:
                IdleTime = (self.step - vehicle.idle_start_time) * 2

            SumOrderNum = vehicle.Sum_order_num / 5
            SumOrderValue = vehicle.Sum_order_value / 5
            PoIDisList = self.PoILocation[Cluster_ID]

            current = np.zeros((1, 131))
            current[0, 0] = TimePeriod
            current[0, 1] = Day
            current[0, 2:25] = PoIDisList

            current[0, 25:125] = traffic_input[Cluster_ID]

            current[0, 125] = HomeDis
            current[0, 126] = Familarity
            current[0, 127] = Cluster_ID
            current[0, 128] = IdleTime
            current[0, 129] = SumOrderNum
            current[0, 130] = SumOrderValue

            Temp = vehicle.UpdateLog(current)
            Input_matrix.append(Temp)

        return Input_matrix

    def Update_Pre_List_batch(self, id_list, data, num):
        for index in range(num):
            id = id_list[index]
            self.PreList[id] = data[index].cpu().detach().numpy()

    def Update_Pre_List(self, classified_id, classified_input, tag):
        class_num = len(classified_id[tag])
        id_list = classified_id[tag]
        data = classified_input[tag]
        data = torch.tensor(np.array(data), dtype=torch.float)
        data = data.view(-1, 131 * 3)
        Network = self.ClassNetList[tag]
        batch_num = class_num // 64
        surplus_num = class_num % 64
        for iter_batch in range(batch_num):
            NetworkInput = torch.zeros([64, 131 * 3])
            NetworkInput = data[0 + iter_batch * 64:64 + iter_batch * 64, :]
            output = Network(NetworkInput)
            self.Update_Pre_List_batch(id_list[0 + iter_batch * 64:64 + iter_batch * 64], output, 64)
        NetworkInput = torch.zeros([64, 131 * 3])
        NetworkInput[0:surplus_num, :] = data[class_num - surplus_num:class_num, :]
        output = Network(NetworkInput)
        self.Update_Pre_List_batch(id_list[class_num - surplus_num:class_num], output, surplus_num)

    def min_max(self, input):
        min = torch.min(input)
        max = torch.max(input)
        output = (input - min) / (max - min + 1)
        return output / 3

    def Refresh_Pre(self):
        Idle_vehicle_list = []
        Idle_vehicle_ID_list = []
        for cluster in self.Clusters:
            for Idle_vehicle in cluster.IdleVehicles:
                Idle_vehicle_list.append(Idle_vehicle)
                Idle_vehicle_ID_list.append(Idle_vehicle.ID)

        Date = self.Orders[0].ReleasTime.day
        Day = (Date % 7 + 3) % 7 + 1
        TimePeriod = self.step * 2

        DE_mat = self.min_max(self.DE_mat)
        SE_mat = self.min_max(self.SE_mat)
        Waiting_Time = self.min_max(self.Waiting_Time)
        speed = 0.5

        traffic_input = []
        for Cluster_ID in range(0, self.NumGrideHeight * self.NumGrideWidth):
            ai = self.NumGrideHeight - 1 - int(Cluster_ID) // self.NumGrideWidth
            aj = int(Cluster_ID) % self.NumGrideWidth
            tmp = []
            for i in range(-2, 3):
                for j in range(-2, 3):
                    xi = ai + i
                    xj = aj + j
                    if xi < 0 or xi > self.NumGrideHeight - 1 or xj < 0 or xj > self.NumGrideWidth - 1:
                        vain = [0] * 4
                        tmp.extend(vain)
                    else:
                        clu = Cluster_ID + j - self.NumGrideWidth * i
                        tmp.append(DE_mat[clu].item())
                        tmp.append(SE_mat[clu].item())
                        tmp.append(speed)
                        tmp.append(Waiting_Time[clu].item())
            traffic_input.append(tmp)

        Input_matrix = self.PrepareInput(Idle_vehicle_list, Date, Day, TimePeriod, traffic_input)

        classified_input = [[], [], [], [], []]
        classified_id = [[], [], [], [], []]
        for id in Idle_vehicle_ID_list:
            classID = int(self.ClassDict[id])
            classified_id[classID].append(id)
            classified_input[classID].append(Input_matrix[Idle_vehicle_ID_list.index(id)])

        self.Update_Pre_List(classified_id, classified_input, 0)
        self.Update_Pre_List(classified_id, classified_input, 1)
        self.Update_Pre_List(classified_id, classified_input, 2)
        self.Update_Pre_List(classified_id, classified_input, 3)
        self.Update_Pre_List(classified_id, classified_input, 4)
