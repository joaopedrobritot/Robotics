# Projeto de Robô para disciplina Robótica
# Professor Glauber Rodrigues Leite
# Dupla:
#       João Pedro Tomé
#       Igor Hutson

import roboticstoolbox as rtb
import spatialmath as sm
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi
from roboticstoolbox import *
import sim

np.set_printoptions(suppress=True)

# DH parameters
a = [0, -0.425, -0.39225, 0, 0, 0]
d = [0.089, 0, 0, 0.10915, 0.09465, 0.0823]
alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]

robot = DHRobot([
    RevoluteDH(a = a[0], d = d[0], alpha = alpha[0], offset=-pi/2),
    RevoluteDH(a = a[1], d = d[1], alpha = alpha[1], offset=-pi/2),
    RevoluteDH(a = a[2], d = d[2], alpha = alpha[2]),
    RevoluteDH(a = a[3], d = d[3], alpha = alpha[3], offset=-pi/2),
    RevoluteDH(a = a[4], d = d[4], alpha = alpha[4], offset=pi),
    RevoluteDH(a = a[5], d = d[5], alpha = alpha[5], offset=-pi/2)
])

print("DH parameters (UR5 robot):", robot)

# Coppelia simulation

print ('Program started')
sim.simxFinish(-1) 
clientID = sim.simxStart('127.0.0.1',19999,True,True,5000,5)

if clientID != -1:
    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot_wait)
    print ('Connected to remote API server')
    
    sim.simxAddStatusbarMessage(clientID,'Connection working...',sim.simx_opmode_oneshot_wait)
    time.sleep(0.05)
    
    _, joint1 = sim.simxGetObjectHandle(clientID, '/UR5/joint', sim.simx_opmode_oneshot_wait)
    _, joint2 = sim.simxGetObjectHandle(clientID, '/UR5/link/joint', sim.simx_opmode_oneshot_wait)
    _, joint3 = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint', sim.simx_opmode_oneshot_wait)
    _, joint4 = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint/link/joint', sim.simx_opmode_oneshot_wait)
    _, joint5 = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint/link/joint/link/joint', sim.simx_opmode_oneshot_wait)
    _, joint6 = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint/link/joint/link/joint/link/joint', sim.simx_opmode_oneshot_wait)
    _, atuador = sim.simxGetObjectHandle(clientID, '/UR5/link7_visible', sim.simx_opmode_oneshot_wait)
    _, dummy = sim.simxGetObjectHandle(clientID, '/Dummy', sim.simx_opmode_oneshot_wait)

    val = sim.simxGetJointPosition(clientID, joint1, sim.simx_opmode_streaming)
    val = sim.simxGetJointPosition(clientID, joint2, sim.simx_opmode_streaming)
    val = sim.simxGetJointPosition(clientID, joint3, sim.simx_opmode_streaming)
    val = sim.simxGetJointPosition(clientID, joint4, sim.simx_opmode_streaming)
    val = sim.simxGetJointPosition(clientID, joint5, sim.simx_opmode_streaming)
    val = sim.simxGetJointPosition(clientID, joint6, sim.simx_opmode_streaming)
    err, d = sim.simxGetObjectPosition(clientID, dummy, -1, sim.simx_opmode_streaming)
    err, X = sim.simxGetObjectPosition(clientID, atuador, -1, sim.simx_opmode_streaming)

    sim.simxSetJointTargetPosition(clientID, joint1, 0, sim.simx_opmode_oneshot)
    time.sleep(0.05)
    sim.simxSetJointTargetPosition(clientID, joint2, 0, sim.simx_opmode_oneshot)
    time.sleep(0.05) 
    sim.simxSetJointTargetPosition(clientID, joint3, 0, sim.simx_opmode_oneshot)
    time.sleep(0.05) 
    sim.simxSetJointTargetPosition(clientID, joint4, 0, sim.simx_opmode_oneshot)
    time.sleep(0.05) 
    sim.simxSetJointTargetPosition(clientID, joint5, 0, sim.simx_opmode_oneshot)
    time.sleep(0.05) 
    sim.simxSetJointTargetPosition(clientID, joint6, 0, sim.simx_opmode_oneshot)
    time.sleep(0.05)    
    time.sleep(1)
    
    delta_t = 0.1
    e1, e2, e3, e4, e5, e6 = ([] for i in range(6))
    
    j = 100
    while j > 0:
        _, q1 = sim.simxGetJointPosition(clientID, joint1, sim.simx_opmode_buffer)
        _, q2 = sim.simxGetJointPosition(clientID, joint2, sim.simx_opmode_buffer)
        _, q3 = sim.simxGetJointPosition(clientID, joint3, sim.simx_opmode_buffer)
        _, q4 = sim.simxGetJointPosition(clientID, joint4, sim.simx_opmode_buffer)
        _, q5 = sim.simxGetJointPosition(clientID, joint5, sim.simx_opmode_buffer)
        _, q6 = sim.simxGetJointPosition(clientID, joint6, sim.simx_opmode_buffer)
        time.sleep(0.05)
        q = [q1, q2, q3, q4, q5, q6]
                
        _, d = sim.simxGetObjectPosition(clientID, dummy, -1, sim.simx_opmode_buffer)    
        _, X = sim.simxGetObjectPosition(clientID, atuador, sim.sim_handle_parent, sim.simx_opmode_buffer)
        time.sleep(0.05)
        
        atuador_pose = robot.fkine(q)
        dum_pose = sm.SE3(d[0], d[1], d[2])   
        v, arrived = rtb.p_servo(atuador_pose, dum_pose, 1)
        Jacob = robot.jacobe(q)
        Jacob_inv = np.linalg.pinv(Jacob)
        q_dot = np.matmul(Jacob_inv, v)                
        q += q_dot * delta_t

        error = np.linalg.norm(np.array(d) - np.array(X))
        if error < 0.05:
            break
        
        e1.append(q_dot[0])
        e2.append(q_dot[1])
        e3.append(q_dot[2])
        e4.append(q_dot[3])
        e5.append(q_dot[4])
        e6.append(q_dot[5])

        joints = [joint1, joint2, joint3, joint4, joint5, joint6]

        for i in range(0, 6):
            sim.simxSetJointTargetPosition(clientID, joints[i], q[i], sim.simx_opmode_oneshot)
            time.sleep(0.05)

        j -= 1

    plt.plot(e1, color='blue', label='Joint 1', lw=2)
    plt.plot(e2, color='red', label='Joint 2', lw=2)
    plt.plot(e3, color='green', label='Joint 3', lw=2)
    plt.plot(e4, color='yellow', label='Joint 4', lw=2)
    plt.plot(e5, color='magenta', label='Joint 5', lw=2)
    plt.plot(e6, color='turquoise', label='Joint 6', lw=2)
    plt.xlim([0,100])
    plt.ylim([-60, 60])
    plt.title("Erro para cada junta")
    plt.legend()
    plt.savefig('D:\joaop\Desktop\Robotica\Projeto_robo\ErroJuntas.png')
    plt.show()

    sim.simxPauseSimulation(clientID,sim.simx_opmode_oneshot_wait)
    sim.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
