import os, multiprocessing

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from rat_simulator import RatSimulator

class RatSimulatedDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self.length = self.X.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.X[idx, :, :], self.Y[idx, :, :]

class DataGenerator():
    def __init__(self, place_units, head_units, place_cell_centers, head_cell_centers, num_features, batch_timesteps, num_processes=None):
        
        if num_processes == None:
            self.num_processes = int(multiprocessing.cpu_count() / 2)
        else:
            self.num_processes = num_processes
        
        # HYPERPARAMETERS
        self.place_units = place_units
        self.head_units = head_units
        self.place_cell_centers = place_cell_centers
        self.head_cell_centers = head_cell_centers
        self.num_features = num_features
        self.batch_timesteps = batch_timesteps

    def generate_dataset(self, num_traj, num_batch_timesteps, train=True):

        if num_traj % 50 == 0 and (num_traj >= 50):
            sharding_size = 50
        elif num_traj % 40 == 0 and (num_traj >= 40):
            sharding_size = 40
        elif num_traj % 80 == 0 and (num_traj >= 80):
            sharding_size = 80
        elif num_traj % 100 == 0 and (num_traj >= 100):
            sharding_size = 100
        elif (num_traj % 200 == 0) and (num_traj >= 200):
            sharding_size = 200
        elif num_traj % 20 == 0 and (num_traj >= 20):
            sharding_size = 20
        elif num_traj % 10 == 0 and (num_traj >= 10):
            sharding_size = 10
        else:
            sharding_size = 1
        
        if (num_traj / sharding_size) <= self.num_processes:
            self.num_processes = int(num_traj / sharding_size)

        sharding_size = int(num_traj / self.num_processes)
        
        print(f">> Generating Dataset using Multiprocess. sharding_size: {sharding_size}, num_traj: {num_traj}, num_batch_timesteps: {num_batch_timesteps}, num_process: {self.num_processes}\n")
        with multiprocessing.Pool(self.num_processes) as p:
            results = p.starmap(self.generate_dataset_single_process, [(sharding_size, num_batch_timesteps, train, idx) for idx in range(self.num_processes)])

        print(f">> Gathering Dataset..")
        X_list = []
        Y_list = []
        positions_list = []
        angles_list = []

        for (X_piece, Y_piece, positions_piece, angles_piece) in results:
            X_list.append(X_piece)
            Y_list.append(Y_piece)
            positions_list.append(positions_piece)
            angles_list.append(angles_piece)

        X = np.concatenate(X_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)
        positions = np.concatenate(positions_list, axis=0)
        angles = np.concatenate(angles_list, axis=0)

        return X, Y, positions, angles


    def generate_dataset_single_process(self, num_traj, num_batch_timesteps, train=True, n=0):
        X, positions, angles = self.generate_X_data(num_traj, self.batch_timesteps*num_batch_timesteps, n)

        if train:
            Y = self.generate_Y_train_data(num_traj, self.batch_timesteps*num_batch_timesteps, positions, angles, n)
        else:
            Y = self.generate_Y_test_data(num_traj, num_batch_timesteps, positions, angles, n)

        return (X, Y, positions, angles) # (500, 800, 3) (500, 800, 268) (500, 800, 2) (500, 800)

    # def generate_dataset(self, num_traj, num_batch_timesteps, train=True):
    #     X, positions, angles = self.generate_X_data(num_traj, self.batch_timesteps*num_batch_timesteps)

    #     if train:
    #         Y = self.generate_Y_train_data(num_traj, self.batch_timesteps*num_batch_timesteps, positions, angles)
    #     else:
    #         Y = self.generate_Y_test_data(num_traj, num_batch_timesteps, positions, angles)

    #     return X, Y, positions, angles # (500, 800, 3) (500, 800, 268) (500, 800, 2) (500, 800)

    def generate_X_data(self, num_traj, num_timesteps, n=0):
        ratSimulator = RatSimulator(num_timesteps)

        X = np.zeros((num_traj, num_timesteps, self.num_features))
        angles = np.zeros((num_traj, num_timesteps))
        positions = np.zeros((num_traj, num_timesteps, 2))

        # for idx in tqdm(range(num_traj), desc=f">> Generating trajectories #{n}", position=n):
        for idx in range(num_traj):
            vel, angVel, pos, angle = ratSimulator.generateTrajectory()    

            X[idx,:,0] = vel
            X[idx,:,1] = np.sin(angVel)
            X[idx,:,2] = np.cos(angVel)
            angles[idx] = angle
            positions[idx] = pos

        return X, positions, angles
    
    def generate_Y_train_data(self, num_traj, num_timesteps, pos, angle, n=0):
        Y = np.zeros((num_traj, num_timesteps, self.place_units + self.head_units))

        for idx in tqdm(range(num_timesteps), desc=f">> Compute Distribution of Cells #{n}"):
        # for idx in range(num_timesteps):
            Y[:, idx, :self.place_units] = self.compute_place_distribution(pos[:,idx], self.place_cell_centers)
            Y[:, idx, self.place_units:] = self.compute_head_distribution(angle[:,idx], self.head_cell_centers)
        return Y

    def generate_Y_test_data(self, num_traj, num_batch_timesteps, pos, angle, n=0):
        Y = np.zeros((num_traj, num_batch_timesteps, self.place_units + self.head_units))

        for idx in tqdm(range(num_batch_timesteps), desc=f">> Compute Distribution of Cells #{n}"):
        # for idx in range(num_batch_timesteps):
            Y[:, idx, :self.place_units] = self.compute_place_distribution(pos[:,idx*self.batch_timesteps], self.place_cell_centers)
            Y[:, idx, self.place_units:] = self.compute_head_distribution(angle[:,idx*self.batch_timesteps], self.head_cell_centers)
        return Y

    def compute_place_distribution(self, positions, cellCenters):
        num_cells = cellCenters.shape[0]
        num_traj = positions.shape[0]
        # Place Cell scale
        sigma = 0.3 # 0.01 NOT 0.01 PAPER ERROR

        summs = np.zeros(num_traj)

        # Store [envs,256] elements. Every row stores the distribution for a trajectory
        distributions = np.zeros((num_traj,num_cells))

        # We have 256 elements in the Place Cell Distribution. For each of them
        for i in range(num_cells):
            # positions has shape [envs,2] and cellCenters[i] has shape [envs,2]
            l2Norms = np.sum((positions - cellCenters[i])**2, axis=1)

            # placeCells has shape [envs,1]
            placeCells = np.exp(-(l2Norms/(2*sigma**2)))

            distributions[:,i]=placeCells
            summs += placeCells

        distributions = distributions/summs[:,None]
        return distributions

    def compute_head_distribution(self,facingAngles, cellCenters):
        num_cells = cellCenters.shape[0]
        num_traj = facingAngles.shape[0]
        # Concentration parameter
        k = 20

        summs = np.zeros(num_traj)

        # Store [envs,12] elements. Every row stores the distribution for a trajectory
        distributions = np.zeros((num_traj,num_cells))

        # We have 12 elements in the Head Direction Cell Distribution. For each of them
        for i in range(num_cells):
            # facingAngles has shape [envs, 1] while cellCenters[i] has shape [envs,1]
            headDirects = np.squeeze(np.exp(k*np.cos(facingAngles - cellCenters[i])))
            distributions[:,i] = headDirects
            summs += headDirects
        
        distributions = distributions/summs[:,None]

        return distributions
