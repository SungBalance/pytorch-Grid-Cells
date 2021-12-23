import os

import numpy as np
import torch
from torch.nn import functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import utils

class Trainer():
    def __init__(self, model, num_timesteps, batch_timesteps, learning_rate, clipping, weight_decay):
        self.model = model

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        self.model.to(self.device)

        self.place_units = self.model.place_units
        self.head_units = self.model.head_units

        self.global_iteration = 0

        self.learning_rate = learning_rate
        self.clipping = clipping
        self.weight_decay = weight_decay

        self.num_timesteps = num_timesteps
        self.batch_timesteps = batch_timesteps

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

        self.writer = SummaryWriter()

    def train(self, X, Y):
        self.model.train()

        X = X.to(self.device, dtype=torch.float32)
        Y = Y.to(self.device, dtype=torch.float32)
        
        train_mean_loss = 0        

        for time_start_idx in range(0, self.num_timesteps, self.batch_timesteps):
            # set data batch
            time_end_idx = time_start_idx + self.batch_timesteps

            # Retrieve the inputs for the self.batch_timesteps timesteps
            x_batch = X[:,time_start_idx:time_end_idx, :]
            # Retrieve the labels for the self.batch_timesteps timesteps
            y_batch = Y[:,time_start_idx:time_end_idx, :]
            # Retrieve label at timestep 0 for the self.batch_timesteps timesteps
            init_LSTM = y_batch[:, 0, :]

            place_cell_ground = init_LSTM[:, :self.place_units].unsqueeze(0)
            head_cell_ground = init_LSTM[:, self.place_units:].unsqueeze(0)

            label_place = y_batch[:, :, :self.place_units].reshape((-1, self.place_units))
            label_head = y_batch[:, :, self.place_units:].reshape((-1, self.head_units))

            # init update
            self.optimizer.zero_grad()

            # forward
            outputs_place, outputs_head, weight_place, weight_cell = self.model(x_batch, place_cell_ground, head_cell_ground)

            # get loss
            loss_place = torch.mean(F.binary_cross_entropy_with_logits(outputs_place, label_place))
            loss_head = torch.mean(F.binary_cross_entropy_with_logits(outputs_head, label_head))
            l2_loss = self.weight_decay * torch.sum(weight_place.pow(2))/2 + self.weight_decay * torch.sum(weight_cell.pow(2))/2
            loss = loss_place + loss_head + l2_loss

            # get gradients
            loss.backward()

            # gradient clipping for linear_place_cell and linear_head_cell
            torch.nn.utils.clip_grad_value_(self.model.linear_place_cell.weight, self.clipping)
            torch.nn.utils.clip_grad_value_(self.model.linear_head_cell.weight, self.clipping)

            # update parameters
            self.optimizer.step()

            train_mean_loss += loss/(self.num_timesteps//self.batch_timesteps)

        self.writer.add_scalar("train_mean_loss", train_mean_loss, self.global_iteration)
        self.writer.flush()

    def inference(self, X, Y, positions_array, place_cell_centers, test_num_traj, num_timesteps, draw=True):
        self.model.eval()

        X = X.to(self.device, dtype=torch.float32)
        Y = Y.to(self.device, dtype=torch.float32)

        mean_distance = 0

        display_pred_trajectories = np.zeros((test_num_traj, num_timesteps, 2)) # [test_num_traj, num_timesteps, (x,y)]

        # Divide the sequence in self.batch_timesteps steps
        for batch_timesteps_idx, time_start_idx in enumerate(range(0, self.num_timesteps, self.batch_timesteps)):
            time_end_idx = time_start_idx + self.batch_timesteps

            # Retrieve the inputs for the timestep
            x_batch = X[:,time_start_idx:time_end_idx]
            init_LSTM_state = Y[:, batch_timesteps_idx, :]

            # When the timestep=0, initialize the hidden and cell state of LSTm using init_LSTM_state. if not timestep=0, the network will use cell_state and hidden_state
            place_cell_ground = init_LSTM_state[:, :self.place_units].unsqueeze(0)
            head_cell_ground = init_LSTM_state[:, self.place_units:].unsqueeze(0)

            # forward
            place_cell_layer, _, _, _ = self.model(x_batch, place_cell_ground, head_cell_ground)
            

            # retrieve the position in these self.batch_timesteps timesteps
            positions = positions_array[:, time_start_idx:time_end_idx]
            # Retrieve which cell has been activated. Placecell has shape self.batch_timesteps0,256. idx has shape self.batch_timesteps0,1
            idx = place_cell_layer.argmax(dim=1).cpu().numpy()
            
            # Retrieve the place cell center of the activated place cell
            pred_positions = place_cell_centers[idx]


            display_pred_trajectories[:,time_start_idx:time_end_idx] = np.reshape(pred_positions, (10,self.batch_timesteps,2))

            # Compute the distance between truth position and place cell center
            distances = np.sqrt(np.sum((pred_positions - np.reshape(positions, (-1,2)))**2, axis=1))
            mean_distance += np.mean(distances) / (self.batch_timesteps)
        
            self.writer.add_scalar("mean_distance", mean_distance, self.global_iteration)
        self.writer.flush()

        if draw:
            # Compare predicted trajectory with real trajectory
            rows=3
            cols=3
            fig = plt.figure(figsize=(40, 40))
            for i in range(rows*cols):
                ax=fig.add_subplot(rows, cols, i+1)
                #plot real trajectory
                plt.plot(positions_array[i,:,0], positions_array[i,:,1], 'b', label="Truth Path")
                #plot predicted trajectory
                plt.plot(display_pred_trajectories[i,:,0], display_pred_trajectories[i,:,1], 'go', label="Predicted Path")
                plt.legend()
                ax.set_xlim(0,2.2)
                ax.set_ylim(0,2.2)

            utils.save_fig(fig, f'./results/Trajectories/{self.global_iteration}_predicted_trajectory.png')

    def save_model(self, path=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'global_iteration': self.global_iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)

    def load_model(self, path=None):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.global_iteration = checkpoint['global_iteration']

            return True
        else:
            return False