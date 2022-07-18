from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
import spatialmath as sm
from roboticstoolbox import *
from zmqRemoteApi import RemoteAPIClient
import time

l1 = 0.475
l2 = 0.4

# Intervalo de integração

deltaTime = 0.05

# Robô utilizando os parâmetro de Denavit-Hartenberg

robot = DHRobot([
    RevoluteDH(a = l1, d = 0, alpha = 0),
    RevoluteDH(a = l2, d = 0, alpha = np.pi),
    PrismaticDH(a = 0, theta = 0, alpha = 0, qlim = [0, 0.1]),
    RevoluteDH(a = 0, d = 0, alpha = 0)
])

client = RemoteAPIClient()
sim = client.getObject('sim')

client.setStepping(True)
sim.startSimulation()

# Obtendo Handles das juntas e do dummy

JR1 = sim.getObject('/MTB/axis')
JR2 = sim.getObject('/MTB/link/axis')
JP1 = sim.getObject('/MTB/link/axis/link/axis')
suction_pad = sim.getObject('/MTB/suctionPad')
dummy = sim.getObject('/reference')

error_J1 = []
error_J2 = []
error_J3 = []
error_J4 = []
error_arr = []

while True:
    q0 = sim.getJointPosition(JR1)
    q1 = sim.getJointPosition(JR2)
    q2 = sim.getJointPosition(JP1)

    q = [q0, q1, q2, 0]
    dum_pos = sim.getObjectPosition(dummy, -1)
    pad_pos = sim.getObjectPosition(suction_pad, -1)

    pad_pose = robot.fkine(q)
    dummy_pose = sm.SE3(dum_pos[0], dum_pos[1], dum_pos[2])

    v, arrived = rtb.p_servo(pad_pose, dummy_pose, 1)

    Jacob = robot.jacobe(q)
    Jacob_inv = np.linalg.pinv(Jacob)

    q_dot = np.matmul(Jacob_inv, v)
    q += q_dot * deltaTime

    error_J1.append(q_dot[0])
    error_J2.append(q_dot[1])
    error_J3.append(q_dot[2])
    error_J4.append(q_dot[3])

    error = np.linalg.norm(np.array(dum_pos) - np.array(pad_pos))
    error_arr.append(error)

    sim.setJointPosition(JR1, q[0])
    time.sleep(deltaTime)
    sim.setJointPosition(JR2, q[1])
    time.sleep(deltaTime)
    sim.setJointPosition(JP1, q[2])
    time.sleep(deltaTime)

    if error < 0.03:
        sim.stopSimulation()
        break

plt.plot(error_arr, color='red', lw=2, label='Erro X Time')
plt.title('Distance Error')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Error")
#plt.savefig('D:\joaop\Desktop\Robotica\Cinematica_Scara\ErrorPlot.png')
plt.show()


plt.plot(error_J1, color='blue', label='Joint 1', lw=2)
plt.plot(error_J2, color='red', label='Joint 2', lw=2)
plt.plot(error_J3, color='green', label='Joint 3', lw=2)
plt.plot(error_J4, color='yellow', label='Joint 4', lw=2)
plt.title("Erro para cada junta")
#plt.savefig('D:\joaop\Desktop\Robotica\Cinematica_Scara\Juntas.png')
plt.legend()
plt.show()
