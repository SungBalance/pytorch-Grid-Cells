import numpy as np
import os

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate2d

def show_grid_cells(model, data_generator, num_traj, batch_traj, num_timesteps, num_batch_timesteps, batch_timesteps, place_units, linear_units, bins):
    print("\n\n-----------------------------------")
    print("Show Grid Cells..")

    # Ready model
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    model.eval()

    print("> Generate new data")
    X_data, Y_data, positions, _ = data_generator.generate_dataset(num_traj, num_batch_timesteps, train=False)
    X_data = torch.tensor(X_data, dtype=torch.float32, device=device)
    Y_data = torch.tensor(Y_data, dtype=torch.float32, device=device)


    # Feed 500 examples at time to avoid memory problems. Otherwise (10000*100=1million matrix)
    factor = 2.2 / bins
    activityMap = np.zeros((linear_units, bins, bins))
    counterActivityMap = np.zeros((linear_units, bins, bins))
    
    for batch_timesteps_idx, time_start_idx in enumerate(range(0, num_timesteps, batch_timesteps)):
        time_end_idx = time_start_idx + batch_timesteps

        # Retrieve the inputs for the timestep
        x_batch = X_data[:,time_start_idx:time_end_idx, :]

        # When the timestep=0, initialize the hidden and cell state of LSTm using init_LSTM_state. if not timestep=0, the network will use cell_state and hidden_state
        place_cell_ground = Y_data[:, batch_timesteps_idx, :place_units].unsqueeze(0)
        head_cell_ground = Y_data[:, batch_timesteps_idx, place_units:].unsqueeze(0)

        # forward
        linear_neurons = model.linear_output(x_batch, place_cell_ground, head_cell_ground).cpu().detach().numpy()
        
        # Convert 500, (batch_timesteps=100), 2 -> (50000,2)
        position_reshaped = np.reshape(positions[:,time_start_idx:time_end_idx, :],(-1,2))

        #save the value of the neurons in the linear layer at each timestep
        for t in range(linear_neurons.shape[0]): # 50000 iter
            # Compute which bins are for each position
            bin_x, bin_y = (position_reshaped[t]//factor).astype(int)

            if(bin_y == bins):
                bin_y = bins-1
            elif(bin_x == bins):
                bin_x = bins-1
            
            #Now there are the 512 values of the same location
            activityMap[:, bin_y, bin_x] += np.abs(linear_neurons[t]) # linear_neurons must be a vector of 512
            counterActivityMap[:, bin_y, bin_x] += np.ones((linear_units))

    # Compute average value
    result = activityMap #/ counterActivityMap

    # normalize
    norm_map=(result-np.min(result)) / (np.max(result)-np.min(result))

    # Save images
    os.makedirs("./results/grid_cells", exist_ok=True)

    cols = 16
    rows = 32
    
    # Save images
    print(">> Printing neurons of activityMaps")
    fig = plt.figure(figsize=(80, 80))
    for i in range(linear_units):
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(norm_map[i], cmap="jet", origin="lower")
        plt.axis('off')

    fig.savefig('./results/grid_cells/activityMaps.jpg')

    print(">> Printing neurons of corrMaps")
    fig = plt.figure(figsize=(80, 80))
    for i in range(linear_units):
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(correlate2d(norm_map[i], norm_map[i]), cmap="jet", origin="lower")
        plt.axis('off')

    fig.savefig('./results/grid_cells/corrMaps.jpg')





