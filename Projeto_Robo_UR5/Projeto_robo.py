import time

import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
from numpy import cos, pi, sin
from roboticstoolbox import *

import sim

np.set_printoptions(suppress=True)

#######################
# CRIANDO O ROBÔ      #
#######################

# parametros DH

# revolução
a1 = 0
d1 = 0.089
# theta1 variável
alpha1 = pi/2

# revolução
a2 = -0.425
d2 = 0
# theta2 variável
alpha2 = 0

# revolução
a3 = -0.39225
d3 = 0
# theta3 variável
alpha3 = 0

# revolução
a4 = 0
d4 = 0.10915
# theta4 variável
alpha4 = pi/2

# revolução
a5 = 0
d5 = 0.09465
# theta5 variável
alpha5 = -pi/2

# revolução
a6 = 0
d6 = 0.0823
# theta6 variável
alpha6 = 0

# código abaixo monta o robô a partir dos parâmetros de DH
robot = DHRobot([
    RevoluteDH(a=a1, d=d1, alpha=alpha1, offset=-pi/2),
    RevoluteDH(a=a2, d=d2, alpha=alpha2, offset=-pi/2),
    RevoluteDH(a=a3, d=d3, alpha=alpha3),
    RevoluteDH(a=a4, d=d4, alpha=alpha4, offset=-pi/2),
    RevoluteDH(a=a5, d=d5, alpha=alpha5, offset=pi),
    RevoluteDH(a=a6, d=d6, alpha=alpha6, offset=-pi/2)
])

# robot.teach()
print("Tabela DH do robô:")
print(robot)

#######################
# LOOP PRINCIPAL      #
#######################

print ('Program started')
sim.simxFinish(-1) 
clientID = sim.simxStart('127.0.0.1',19999,True,True,5000,5)

if clientID != -1:
    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot_wait)
    print ('Connected to remote API server')
    
    sim.simxAddStatusbarMessage(clientID,'Connection working...',sim.simx_opmode_oneshot_wait)
    time.sleep(0.02)
    
    # Criando handlers
    erro, Joint1 = sim.simxGetObjectHandle(clientID, '/UR5/joint', sim.simx_opmode_oneshot_wait)
    erro, Joint2 = sim.simxGetObjectHandle(clientID, '/UR5/link/joint', sim.simx_opmode_oneshot_wait)
    erro, Joint3 = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint', sim.simx_opmode_oneshot_wait)
    erro, Joint4 = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint/link/joint', sim.simx_opmode_oneshot_wait)
    erro, Joint5 = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint/link/joint/link/joint', sim.simx_opmode_oneshot_wait)
    erro, Joint6 = sim.simxGetObjectHandle(clientID, '/UR5/link/joint/link/joint/link/joint/link/joint/link/joint', sim.simx_opmode_oneshot_wait)
    erro, pad = sim.simxGetObjectHandle(clientID, '/UR5/link7_visible', sim.simx_opmode_oneshot_wait)
    erro, dummy = sim.simxGetObjectHandle(clientID, '/Dummy', sim.simx_opmode_oneshot_wait)
    
    # Inicializando stream de dados do coppelia
    value = sim.simxGetJointPosition(clientID, Joint1, sim.simx_opmode_streaming)
    value = sim.simxGetJointPosition(clientID, Joint2, sim.simx_opmode_streaming)
    value = sim.simxGetJointPosition(clientID, Joint3, sim.simx_opmode_streaming)
    value = sim.simxGetJointPosition(clientID, Joint4, sim.simx_opmode_streaming)
    value = sim.simxGetJointPosition(clientID, Joint5, sim.simx_opmode_streaming)
    value = sim.simxGetJointPosition(clientID, Joint6, sim.simx_opmode_streaming)
    erro, d = sim.simxGetObjectPosition(clientID, dummy, -1, sim.simx_opmode_streaming)
    erro, X = sim.simxGetObjectPosition(clientID, pad, -1, sim.simx_opmode_streaming)
    # time.sleep(1)
    
    # Atribuindo valores iniciais às juntas do robô  
    sim.simxSetJointTargetPosition(clientID, Joint1, 0, sim.simx_opmode_oneshot)
    time.sleep(0.05)
    sim.simxSetJointTargetPosition(clientID, Joint2, 0, sim.simx_opmode_oneshot)
    time.sleep(0.05) 
    sim.simxSetJointTargetPosition(clientID, Joint3, 0, sim.simx_opmode_oneshot)
    time.sleep(0.05) 
    sim.simxSetJointTargetPosition(clientID, Joint4, 0, sim.simx_opmode_oneshot)
    time.sleep(0.05) 
    sim.simxSetJointTargetPosition(clientID, Joint5, 0, sim.simx_opmode_oneshot)
    time.sleep(0.05) 
    sim.simxSetJointTargetPosition(clientID, Joint6, 0, sim.simx_opmode_oneshot)
    time.sleep(0.05)    
    time.sleep(1)
    
    delta_t = 0.1 # intervalo de tempo usado para a integração numérica
    error1 = []
    error2 = []
    error3 = []
    error4 = []
    error5 = []
    error6 = []
     
    while True:
        # lendo valores das juntas
        erro, q1 = sim.simxGetJointPosition(clientID, Joint1, sim.simx_opmode_buffer)
        erro, q2 = sim.simxGetJointPosition(clientID, Joint2, sim.simx_opmode_buffer)
        erro, q3 = sim.simxGetJointPosition(clientID, Joint3, sim.simx_opmode_buffer)
        erro, q4 = sim.simxGetJointPosition(clientID, Joint4, sim.simx_opmode_buffer)
        erro, q5 = sim.simxGetJointPosition(clientID, Joint5, sim.simx_opmode_buffer)
        erro, q6 = sim.simxGetJointPosition(clientID, Joint6, sim.simx_opmode_buffer)
        time.sleep(0.05)
        q = [q1, q2, q3, q4, q5, q6]
                
        # lendo posição do atuador e do dummy
        erro, d = sim.simxGetObjectPosition(clientID, dummy, -1, sim.simx_opmode_buffer)    
        erro, X = sim.simxGetObjectPosition(clientID, pad, sim.sim_handle_parent, sim.simx_opmode_buffer)
        time.sleep(0.05)
        
        # aplicando resolved-rate
        
        pad_pose = robot.fkine(q)
        dummy_pose = sm.SE3(d[0], d[1], d[2])
                
        v, arrived = rtb.p_servo(pad_pose, dummy_pose, 1)
        
        J = robot.jacobe(q)
        J_inv = np.linalg.pinv(J)
        
        q_dot = J_inv @ v                
        q += q_dot*(delta_t)
        
        # salvando erros para o plot
        
        error1.append(q_dot[0])
        error2.append(q_dot[1])
        error3.append(q_dot[2])
        error4.append(q_dot[3])
        error5.append(q_dot[4])
        error6.append(q_dot[5])
        
        np.save('ur3_errors.npy', np.array([error1, error2, error3, error4, error5, error6]))
        
        sim.simxSetJointTargetPosition(clientID, Joint1, q[0], sim.simx_opmode_oneshot)
        time.sleep(0.05)
        
        sim.simxSetJointTargetPosition(clientID, Joint2, q[1], sim.simx_opmode_oneshot)
        time.sleep(0.05)
        
        sim.simxSetJointTargetPosition(clientID, Joint3, q[2], sim.simx_opmode_oneshot)
        time.sleep(0.05)
        
        sim.simxSetJointTargetPosition(clientID, Joint4, q[3], sim.simx_opmode_oneshot)
        time.sleep(0.05)
        
        sim.simxSetJointTargetPosition(clientID, Joint5, q[4], sim.simx_opmode_oneshot)
        time.sleep(0.05)
        
        sim.simxSetJointTargetPosition(clientID, Joint6, q[5], sim.simx_opmode_oneshot)
        time.sleep(0.05)
        
        time.sleep(0.05)
                
    #     break
        
        
    # Pause simulation
    sim.simxPauseSimulation(clientID,sim.simx_opmode_oneshot_wait)

    # Close connection:
    sim.simxAddStatusbarMessage(clientID, 'Program paused', sim.simx_opmode_blocking )
    sim.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
