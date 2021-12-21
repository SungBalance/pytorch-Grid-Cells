import numpy as np
import torch
from torch.nn import functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self, model, place_units, head_units, len_steps, learning_rate, clipping, weight_decay, batch_size):
        self.model = model
        self.place_units = place_units
        self.head_units = head_units
        self.len_steps = len_steps
        self.learning_rate = learning_rate
        self.clipping = clipping
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

        self.writer = SummaryWriter()

    def train(self, X, Y, global_step):

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        
        self.model.to(device)
        self.model.train()

        train_mean_loss = 0        

        for time_start_idx in range(0, self.len_steps, 100):
            # set data batch
            time_end_idx = time_start_idx + 100

            # Retrieve the inputs for the 100 timesteps
            x_batch = torch.tensor(X[:,time_start_idx:time_end_idx], dtype=torch.float32, device=device)
            # Retrieve the labels for the 100 timesteps
            y_batch = torch.tensor(Y[:,time_start_idx:time_end_idx], dtype=torch.float32, device=device)
            # Retrieve label at timestep 0 for the 100 timesteps
            init_LSTM = y_batch[:,0]

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

            train_mean_loss += loss/(self.len_steps//100)

        self.writer.add_scalar("train_mean_loss", train_mean_loss, global_step)
        self.writer.flush()

    def inference(self, X, init_LSTM_state, positions_array, pcc, global_step):

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        
        self.model.to(device)
        self.model.eval()

        mean_distance = 0

        display_pred_trajectories = np.zeros((10,800,2))

        # Divide the sequence in 100 steps
        for time_start_idx in range(0, self.len_steps, 100):
            time_end_idx = time_start_idx + 100

            # Retrieve the inputs for the timestep
            x_batch = torch.tensor(X[:,time_start_idx:time_end_idx], dtype=torch.float32, device=device)
            init_LSTM_state = torch.tensor(init_LSTM_state, dtype=torch.float32, device=device)

            # When the timestep=0, initialize the hidden and cell state of LSTm using init_LSTM_state. if not timestep=0, the network will use cell_state and hidden_state
            place_cell_ground = init_LSTM_state[:, (time_start_idx//100), :self.place_units].unsqueeze(0)
            head_cell_ground = init_LSTM_state[:, (time_start_idx//100), self.place_units:].unsqueeze(0)

            # forward
            place_cell_layer, _, _, _ = self.model(x_batch, place_cell_ground, head_cell_ground)
            

            # retrieve the position in these 100 timesteps
            positions = positions_array[:, time_start_idx:time_end_idx]
            # Retrieve which cell has been activated. Placecell has shape 1000,256. idx has shape 1000,1
            idx = place_cell_layer.argmax(dim=1).cpu().numpy()
            
            # Retrieve the place cell center of the activated place cell
            pred_positions = pcc[idx]

            # Update the predictedTrajectory.png
            if global_step%8000 == 0:
                display_pred_trajectories[:,time_start_idx:time_end_idx] = np.reshape(pred_positions, (10,100,2))

            # Compute the distance between truth position and place cell center
            distances = np.sqrt(np.sum((pred_positions - np.reshape(positions, (-1,2)))**2, axis=1))
            mean_distance += np.mean(distances) / (self.len_steps//100)
        
        self.writer.add_scalar("mean_distance", mean_distance, global_step)
        self.writer.flush()

        # Compare predicted trajectory with real trajectory
        if global_step%8000 == 0:
            rows=3
            cols=3
            fig=plt.figure(figsize=(40, 40))
            for i in range(rows*cols):
                ax=fig.add_subplot(rows, cols, i+1)
                #plot real trajectory
                plt.plot(positions_array[i,:,0], positions_array[i,:,1], 'b', label="Truth Path")
                #plot predicted trajectory
                plt.plot(display_pred_trajectories[i,:,0], display_pred_trajectories[i,:,1], 'go', label="Predicted Path")
                plt.legend()
                ax.set_xlim(0,2.2)
                ax.set_ylim(0,2.2)

            fig.savefig(f'./results/{global_step}_predictedTrajectory.png')

    def save_model(self, global_step=None, path=None):
        if path is not None:
            torch.save({
                'global_step': global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, path)

    def load_model(self, path=None):
        if path is not None:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['global_step']

    '''No support Tensorboard yet
    def buildTensorBoardStats(self):
        #Episode data
        self.mn_loss=tf.placeholder(tf.float32)
        self.mergeEpisodeData=tf.summary.merge([tf.summary.scalar("mean_loss", self.mn_loss)])

        self.avgD=tf.placeholder(tf.float32)
        self.mergeAccuracyData=tf.summary.merge([tf.summary.scalar("average_distance", self.avgD)])
    '''
