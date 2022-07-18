import roboticstoolbox as rtb
import spatialmath as sm
import time
import numpy as np
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
    
    err, joint1 = sim.simxGetObjectHandle(clientID, '/UR5/joint', sim.simx_opmode_oneshot_wait)
    err, joint2 = sim.simxGetObjectHandle(clientID, '/UR5/link/joint', sim.simx_opmode_oneshot_wait)
    err, joint3 = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint', sim.simx_opmode_oneshot_wait)
    err, joint4 = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint/link/joint', sim.simx_opmode_oneshot_wait)
    err, joint5 = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint/link/joint/link/joint', sim.simx_opmode_oneshot_wait)
    err, joint6 = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint/link/joint/link/joint/link/joint', sim.simx_opmode_oneshot_wait)
    err, pad = sim.simxGetObjectHandle(clientID, '/UR5/link7_visible', sim.simx_opmode_oneshot_wait)
    err, dummy = sim.simxGetObjectHandle(clientID, '/Dummy', sim.simx_opmode_oneshot_wait)
    
    err, d = sim.simxGetObjectPosition(clientID, dummy, -1, sim.simx_opmode_streaming)
    err, X = sim.simxGetObjectPosition(clientID, pad, -1, sim.simx_opmode_streaming)
    
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
    error1 = []
    error2 = []
    error3 = []
    error4 = []
    error5 = []
    error6 = []
     
    while True:
        err, q1 = sim.simxGetJointPosition(clientID, joint1, sim.simx_opmode_buffer)
        err, q2 = sim.simxGetJointPosition(clientID, joint2, sim.simx_opmode_buffer)
        err, q3 = sim.simxGetJointPosition(clientID, joint3, sim.simx_opmode_buffer)
        err, q4 = sim.simxGetJointPosition(clientID, joint4, sim.simx_opmode_buffer)
        err, q5 = sim.simxGetJointPosition(clientID, joint5, sim.simx_opmode_buffer)
        err, q6 = sim.simxGetJointPosition(clientID, joint6, sim.simx_opmode_buffer)
        time.sleep(0.05)
        q = [q1, q2, q3, q4, q5, q6]
                
        err, d = sim.simxGetObjectPosition(clientID, dummy, -1, sim.simx_opmode_buffer)    
        err, X = sim.simxGetObjectPosition(clientID, pad, sim.sim_handle_parent, sim.simx_opmode_buffer)
        time.sleep(0.05)
        
        pad_pose = robot.fkine(q)
        dummy_pose = sm.SE3(d[0], d[1], d[2])
                
        v, arrived = rtb.p_servo(pad_pose, dummy_pose, 1)
        
        J = robot.jacobe(q)
        J_inv = np.linalg.pinv(J)
        
        q_dot = J_inv @ v                
        q += q_dot*(delta_t)
        
        error1.append(q_dot[0])
        error2.append(q_dot[1])
        error3.append(q_dot[2])
        error4.append(q_dot[3])
        error5.append(q_dot[4])
        error6.append(q_dot[5])
        
        np.save('error_plot_ur5.npy', np.array([error1, error2, error3, error4, error5, error6]))
        
        sim.simxSetJointTargetPosition(clientID, joint1, q[0], sim.simx_opmode_oneshot)
        time.sleep(0.05)
        
        sim.simxSetJointTargetPosition(clientID, joint2, q[1], sim.simx_opmode_oneshot)
        time.sleep(0.05)
        
        sim.simxSetJointTargetPosition(clientID, joint3, q[2], sim.simx_opmode_oneshot)
        time.sleep(0.05)
        
        sim.simxSetJointTargetPosition(clientID, joint4, q[3], sim.simx_opmode_oneshot)
        time.sleep(0.05)
        
        sim.simxSetJointTargetPosition(clientID, joint5, q[4], sim.simx_opmode_oneshot)
        time.sleep(0.05)
        
        sim.simxSetJointTargetPosition(clientID, joint6, q[5], sim.simx_opmode_oneshot)
        time.sleep(0.05)
        
        time.sleep(0.05)
        

    sim.simxPauseSimulation(clientID,sim.simx_opmode_oneshot_wait)

    sim.simxAddStatusbarMessage(clientID, 'Program paused', sim.simx_opmode_blocking )
    sim.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
