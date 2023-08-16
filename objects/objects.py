from config.setting import *

class Cluster(object):

    def __init__(self, ID, Nodes, Neighbor, RebalanceNumber, IdleVehicles, VehiclesArrivetime, Orders):
        self.ID = ID  # int        编号
        self.Nodes = Nodes  # list       节点列表
        self.Neighbor = Neighbor  # list       相邻cluster列表
        self.RebalanceNumber = RebalanceNumber  # int
        self.IdleVehicles = IdleVehicles  # list       空车列表
        self.VehiclesArrivetime = VehiclesArrivetime  # dict       即将到达(sorted)
        self.Orders = Orders  # list       订单列表
        self.PerRebalanceIdleVehicles = 0
        self.LaterRebalanceIdleVehicles = 0
        self.PerMatchIdleVehicles = 0  # 在match（）后重制，表示当前match（）时有多少量空车
        self.RebalanceFrequency = 0
        self.potential = 0

        self.minibatch = None
        self.trans = []
        self.actor = None
        self.critic = None
        self.actor_opt=None
        self.critic_opt=None

    def Reset(self):
        self.RebalanceNumber = 0
        self.IdleVehicles.clear()
        self.VehiclesArrivetime.clear()
        self.Orders.clear()
        self.PerRebalanceIdleVehicles = 0
        self.PerMatchIdleVehicles = 0
        self.potential = 0

    def ArriveClusterUpDate(self, vehicle):  # 添加车辆
        self.IdleVehicles.append(vehicle)
        self.VehiclesArrivetime.pop(vehicle)  # 推出即将到达

    def Example(self):
        print("Order Example output")
        print("ID:", self.ID)
        print("Nodes:", self.Nodes)
        print("Neighbor:", self.Neighbor)
        print("RebalanceNumber:", self.RebalanceNumber)
        print("IdleVehicles:", self.IdleVehicles)
        print("VehiclesArrivetime:", self.VehiclesArrivetime)
        print("Orders:", self.Orders)


class Order(object):

    def __init__(self, ID, ReleasTime, PickupPoint, DeliveryPoint, PickupTimeWindow, PickupWaitTime, ArriveInfo,
                 OrderValue):
        self.ID = ID  # This order's ID
        self.ReleasTime = ReleasTime  # Start time of this order
        self.PickupPoint = PickupPoint  # The starting position of this order
        self.DeliveryPoint = DeliveryPoint  # Destination of this order
        self.PickupTimeWindow = PickupTimeWindow  # Limit of waiting time for this order
        self.PickupWaitTime = PickupWaitTime  # This order's real waiting time from running in the simulator
        self.ArriveInfo = ArriveInfo  # Processing information for this order
        self.OrderValue = OrderValue  # The value of this order

    def ArriveOrderTimeRecord(self, ArriveTime):
        self.ArriveInfo = "ArriveTime:" + str(ArriveTime)

    def Example(self):
        print("Order Example output")
        print("ID:", self.ID)
        print("ReleasTime:", self.ReleasTime)
        print("PickupPoint:", self.PickupPoint)
        print("DeliveryPoint:", self.DeliveryPoint)
        print("PickupTimeWindow:", self.PickupTimeWindow)
        print("PickupWaitTime:", self.PickupWaitTime)
        print("ArriveInfo:", self.ArriveInfo)
        print()

    def Reset(self):
        self.PickupWaitTime = None
        self.ArriveInfo = None


class Vehicle(object):

    def __init__(self, ID, time, obey, LocationNode, Cluster, Orders, DeliveryPoint):
        self.ID = ID  # This vehicle's ID
        self.idle_start_time = 0
        self.Sum_order_num = 0
        self.Sum_order_value = 0
        self.LocationNode = LocationNode  # Current vehicle's location
        self.Cluster = Cluster  # Which cluster the current vehicle belongs to
        self.Orders = Orders  # Orders currently on board
        self.DeliveryPoint = DeliveryPoint  # Next destination of current vehicle
        self.LstmLog = np.zeros((3, 131))
        self.LogPointer = 3  # 初始化
        self.Auto = False
        self.Obey = obey
        self.StartTime = time

    def ArriveVehicleUpDate(self, time, value, DeliveryCluster):  # 移动车辆
        # self.idle_start_time = time
        self.Sum_order_num += 1
        self.Sum_order_value += value
        self.LocationNode = self.DeliveryPoint
        self.DeliveryPoint = None
        self.Cluster = DeliveryCluster
        if len(self.Orders):
            self.Orders.clear()

    def Reset(self):  # 清空所有操作
        self.Orders.clear()
        self.idle_start_time = 0
        self.DeliveryPoint = None
        self.Sum_order_value = 0
        self.Sum_order_num = 0
        self.Auto = False

    def UpdateLog(self, NewLog):
        if self.LogPointer == 3:
            now_time = last_time = 0
            self.LogPointer = 2
        else:
            # 判断是否连续巡航
            now_time = NewLog[0, 0]
            last_time = self.LstmLog[2, 0]
        if abs(last_time - now_time) < 10:  # 连续巡航
            if self.LogPointer == 2:
                self.LstmLog = np.zeros((3, 131))
                self.LstmLog[self.LogPointer, :] = NewLog
                self.LogPointer += -1
            elif -1 < self.LogPointer:
                self.LstmLog[self.LogPointer, :] = NewLog
                self.LogPointer += -1
            else:
                self.LstmLog[0:2, :] = self.LstmLog[1:3, :]
                self.LstmLog[2, :] = NewLog
        else:
            self.LogPointer = 2
            self.LstmLog = np.zeros((3, 131))
            self.LstmLog[self.LogPointer, :] = NewLog
            self.LogPointer += -1

        input = np.zeros((1, 131 * 3))
        fs = self.LstmLog[:, 0:125].reshape((1, -1), order='C')
        fc = self.LstmLog[:, 125:131].reshape((1, -1), order='C')
        input[0, 0:125 * 3] = fs
        input[0, 125 * 3:131 * 3] = fc
        return input

    def Example(self):
        print("Vehicle Example output")
        print("ID:", self.ID)
        print("IdleStartTime:", self.idle_start_time)
        print("LocationNode:", self.LocationNode)
        print("Cluster:", self.Cluster)
        print("Orders:", self.Orders)
        print("DeliveryPoint:", self.DeliveryPoint)

