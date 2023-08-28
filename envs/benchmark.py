import numpy as np
import subprocess
from sumolib import checkBinary
import time
import traci
import vectormath as vmath
import pandas as pd
from scipy import stats

episode = 0
Overall_reward = []
safety = []
avg = []
std = []

def ttc_function(dis, ego_speed, traffic_speed, veh_metric):
    if traffic_speed <= ego_speed:
        ttc_index = max_reward
    else:
        ttc_index = (dis - veh_metric) / (traffic_speed - ego_speed)
    ttc_index = np.clip(ttc_index, 0, max_reward)
    return ttc_index

while episode < 31:
    teleport_range= np.linspace(900,1500,7)
    teleport_time = np.random.choice(teleport_range)
    sumocfg_file = './envs/large_grid_data/exp_0.sumocfg'
    app = 'sumo'
    command = [checkBinary(app), '-c', sumocfg_file]
    command += ['--seed', '10']
    command += ['--no-step-log', 'True']
    command += ['--time-to-teleport', '%d'%teleport_time] # select the time for accident to be solved
    command += ['--no-warnings', 'True']
    command += ['--duration-log.disable', 'True']
    # wait 1s to establish the traci server
    time.sleep(1)
    traci.start(command)
    step = 0
    node_names = traci.trafficlight.getIDList()
    pre_queue = np.zeros(len(node_names))
    accident_vehs = []
    nodes = {}
    for node_name in node_names:
        nodes[node_name] = traci.trafficlight.getControlledLanes(node_name)
    num_accident = np.random.choice(stats.poisson.rvs(mu=4, size=720))
    accident_step = np.random.choice(720, num_accident)
    # print('accident step', accident_step, 'accident time', teleport_time)

    while step < 3601:
        traci.simulationStep()
        episode_reward = []
        episode_safety = []
        episode_avg = []
        episode_std = []
        if step/5 in accident_step:
            accident_veh = np.random.choice(traci.vehicle.getIDList())
            accident_vehs.append(accident_veh)
            traci.vehicle.setSpeed(vehID= accident_veh, speed=0)
        if step % 5 ==0:
            reward_safety_index = []
            avgreward_queue = []
            stdreward_queue = []
            for node_name in node_names:
                queues = []
                stdqueue = []
                waits = []
                for ild in nodes[node_name]:
                    cur_queue = traci.lane.getLastStepHaltingNumber(ild)
                    vehIDs = traci.lane.getLastStepVehicleIDs(ild)
                    for vehID in vehIDs:
                        if vehID in accident_vehs:
                            cur_queue = cur_queue - 1
                        else:
                            cur_queue = cur_queue
                    queues.append(cur_queue)
                avgqueue = np.sum(np.array(queues)) if len(queues) else 0
                stdqueue = np.abs(pre_queue - avgqueue)
                avgreward_queue.append(avgqueue)
                stdreward_queue.append(stdqueue)
            reward_avg_queue = np.array(avgreward_queue)
            reward_avg_queue = np.mean(reward_avg_queue)
            # reward_std_queue = np.array(stdreward_queue)
            reward_std_queue = np.array(stdreward_queue)
            reward_std_queue = np.mean(reward_std_queue)
            pre_queue = np.array(avgreward_queue)
            # risk_inices = np.array get_risk_index())
            # risk_inices = risk_inices.reshape(-1, 1)
            # scaler.fit(risk_inices)
            edges = traci.edge.getIDList()
            edge_veh = {}
            risk_indices = []
            veh_width = 1.8
            veh_length = 5
            # hard code upper limit of reward
            max_length = 200
            min_vel_gap = 0.1
            max_reward = max_length / min_vel_gap
            for edge in edges:
                edge_veh[edge] = traci.edge.getLastStepVehicleIDs(edge)
            for key in edge_veh:
                veh_pos = {}
                veh_forward = {}
                for veh in edge_veh[key]:
                    veh_pos[veh] = traci.vehicle.getPosition(veh)
                    veh_forward[veh] = traci.vehicle.getAngle(veh)
                    # veh_acc[veh] = traci.vehicle.getAcceleration(veh)
                for ego in edge_veh[key]:
                    ego_position = vmath.Vector2Array(veh_pos[ego])
                    ego_forward_angle = np.deg2rad(veh_forward[ego])
                    # convert into vector
                    ego_forward = vmath.Vector2Array(
                        (round(np.cos(ego_forward_angle), 5), round(np.sin(ego_forward_angle), 5)))
                    ego_right_angle = np.deg2rad(veh_forward[ego] + 90)
                    # ego_left_angle = np.deg2rad(veh_forward[ego] -90)
                    ego_left = vmath.Vector2Array((round(np.cos(ego_right_angle), 5), round(np.sin(ego_right_angle), 5)))
                    # ego_right = (round(np.cos(ego_left_angle), 5), round(np.sin(ego_left_angle), 5))
                    front = rear = left = right = None
                    # set up intial threshold
                    front_dis = left_dis = 10
                    rear_dis = right_dis = 0
                    for traffic in edge_veh[key]:
                        traffic_positon = vmath.Vector2Array(veh_pos[traffic])
                        veh_distance = ego_position - traffic_positon
                        longi_dis = veh_distance.dot(ego_forward)
                        lat_dis = veh_distance.dot(ego_left)
                        if longi_dis > 0 and np.abs(longi_dis) <= 1.5 * veh_width:
                            if front is None or longi_dis < front_dis:
                                front = traffic
                                front_dis = longi_dis
                        elif longi_dis < 0 and np.abs(longi_dis) <= 1.5 * veh_width:
                            if rear is None or longi_dis > rear_dis:
                                rear = traffic
                                rear_dis = longi_dis
                        if lat_dis > 0 and np.abs(lat_dis) <= 1.5 * veh_length:
                            if left is None or lat_dis < left_dis:
                                left = traffic
                                left_dis = lat_dis
                        elif lat_dis < 0 and np.abs(lat_dis) <= 1.5 * veh_length:
                            if right is None or lat_dis > right_dis:
                                right = traffic
                                right_dis = lat_dis
                    ego_speed = traci.vehicle.getSpeed(ego)
                    ttc_front = ttc_rear = ttc_left = ttc_right = max_reward
                    if front:
                        front_speed = traci.vehicle.getSpeed(front)
                        ttc_front = ttc_function(front_dis, front_speed, ego_speed, veh_length)
                    if rear:
                        rear_speed = traci.vehicle.getSpeed(rear)
                        ttc_rear = ttc_function(-rear_dis, ego_speed, rear_speed, veh_length)
                    if left:
                        left_speed = traci.vehicle.getSpeed(left)
                        ttc_left = ttc_function(left_dis, left_speed, ego_speed, veh_width)
                    if right:
                        right_speed = traci.vehicle.getSpeed(right)
                        ttc_right = ttc_function(-right_dis, ego_speed, right_speed, veh_width)
                    ttc = min(ttc_front, ttc_right, ttc_rear, ttc_left)
                    reward_safety_index.append(float(ttc))
            reward_safety_index = np.array(reward_safety_index)
            # reward_safety_index = np.mean(reward_safety_index)
            safe_veh = (reward_safety_index < 10).sum()
            reward_safety_index = -(safe_veh / reward_safety_index.shape[0]) * 100
            rewards = 10 * reward_safety_index - 10 * reward_std_queue - reward_avg_queue
            episode_reward.append(rewards)
            episode_safety.append(reward_safety_index)
            episode_avg.append(reward_avg_queue)
            episode_std.append(reward_std_queue)
        step +=1
    traci.close()
    Overall_reward.append(np.mean(episode_reward))
    print('episode %d'%episode, 'episode reward', np.mean(episode_reward))
    safety.append(np.mean(episode_safety))
    avg.append(np.mean(episode_avg))
    std.append(np.mean(episode_std))
    episode +=1

d = {'rewards':Overall_reward, 'safety index': safety, 'average queue': avg, "std queue":std }
df = pd.DataFrame(data = d)
print(df)
df.to_excel('benchmark.xlsx')


