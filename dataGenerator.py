import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import pickle
from ratSimulator import RatSimulator

class dataGenerator():
    def __init__(self, number_steps, num_features, place_units, head_units):
        #HYPERPARAMETERS
        self.number_steps = number_steps
        self.num_features = num_features
        self.place_units = place_units
        self.head_units = head_units

    def generate_data(self, num_traj):
        ratSimulator = RatSimulator(self.number_steps)

        input_data = np.zeros((num_traj, self.number_steps, 3))
        velocities = np.zeros((num_traj, self.number_steps))
        angVelocities = np.zeros((num_traj, self.number_steps))
        angles = np.zeros((num_traj, self.number_steps))
        positions = np.zeros((num_traj, self.number_steps, 2))

        print(">>Generating trajectories")
        for i in tqdm(range(num_traj)):
            vel, angVel, pos, angle = ratSimulator.generateTrajectory()    

            input_data[i,:,0] = vel
            input_data[i,:,1] = np.sin(angVel)
            input_data[i,:,2] = np.cos(angVel)
            angles[i] = angle
            positions[i] = pos

        return input_data, positions, angles
    
    def generate_lstm_init_state(self, num_traj, pos, angle, place_cell_centers, head_cell_centers):
        init_LSTM_state = np.zeros((num_traj, 8, self.place_units + self.head_units))

        print(">>Compute Distribution of Cells")
        for i in tqdm(range(num_traj)):
            init_LSTM_state[:, i, :self.place_units] = self.computePlaceCellsDistrib(pos[:,(i*100)], place_cell_centers)
            init_LSTM_state[:, i, self.place_units:] = self.computeHeadCellsDistrib(angle[:,(i*100)], head_cell_centers)

        return init_LSTM_state

    def computePlaceCellsDistrib(self, positions, cellCenters):
        num_cells=cellCenters.shape[0]
        num_traj=positions.shape[0]
        #Place Cell scale
        sigma=0.3#0.01 NOT 0.01 PAPER ERROR

        summs=np.zeros(num_traj)
        #Store [envs,256] elements. Every row stores the distribution for a trajectory
        distributions=np.zeros((num_traj,num_cells))
        #We have 256 elements in the Place Cell Distribution. For each of them
        for i in range(num_cells):
            #positions has shape [envs,2] and cellCenters[i] has shape [envs,2]
            l2Norms=np.sum((positions - cellCenters[i])**2, axis=1)

            #placeCells has shape [envs,1]
            placeCells=np.exp(-(l2Norms/(2*sigma**2)))

            distributions[:,i]=placeCells
            summs +=placeCells

        distributions=distributions/summs[:,None]
        return distributions

    def computeHeadCellsDistrib(self,facingAngles, cellCenters):
        num_cells=cellCenters.shape[0]
        num_traj=facingAngles.shape[0]
        #Concentration parameter
        k=20

        summs=np.zeros(num_traj)
        #Store [envs,12] elements. Every row stores the distribution for a trajectory
        distributions=np.zeros((num_traj,num_cells))
        #We have 12 elements in the Head Direction Cell Distribution. For each of them
        for i in range(num_cells):
            #facingAngles has shape [envs, 1] while cellCenters[i] has shape [envs,1]
            headDirects=np.squeeze(np.exp(k*np.cos(facingAngles - cellCenters[i])))
            distributions[:,i]=headDirects
            summs+=headDirects
        
        distributions=distributions/summs[:,None]

        return distributions
