import numpy as np
import os

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate2d

from ratSimulator import RatSimulator


def showGridCells(model, dataGenerator, num_traj, number_steps, place_units, head_units, llu, bins, place_cell_centers, head_cell_centers):
    
    #Feed 500 examples at time to avoid memory problems. Otherwise (10000*100=1million matrix)
    
    factor=2.2/bins
    activityMap = np.zeros((llu, bins, bins))
    counterActivityMap = np.zeros((llu, bins, bins))

    ratSimulator = RatSimulator(number_steps)

    input_data = np.zeros((num_traj, number_steps, 3))
    velocities = np.zeros((num_traj, number_steps))
    angVelocities = np.zeros((num_traj, number_steps))
    angles = np.zeros((num_traj, number_steps))
    positions = np.zeros((num_traj, number_steps, 2))

    print(">>Generating trajectories")
    for i in tqdm(range(num_traj)):
        vel, angVel, pos, angle = ratSimulator.generateTrajectory()    

        input_data[i,:,0] = vel
        input_data[i,:,1] = np.sin(angVel)
        input_data[i,:,2] = np.cos(angVel)
        angles[i] = angle
        positions[i] = pos

    init_LSTM_state=np.zeros((num_traj,8,place_units + head_units))
    for i in range(8):
        init_LSTM_state[:, i, :place_units]=dataGenerator.computePlaceCellsDistrib(positions[:,(i*100)], place_cell_centers)
        init_LSTM_state[:, i, place_units:]=dataGenerator.computeHeadCellsDistrib(angles[:,(i*100)], head_cell_centers)


    print(">>Computing Actvity maps")

    if torch.cuda.is_available():
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    model.eval()

    for batch_start_idx in range(0, num_traj, num_traj):
        batch_end_idx = batch_start_idx+num_traj

        #Divide the sequence in 100 steps.
        for time_start_idx in range(0, number_steps, 100):
            time_end_idx=time_start_idx+100

            #Retrieve the inputs for the timestep
            x_batch = torch.tensor(input_data[batch_start_idx:batch_end_idx, time_start_idx:time_end_idx], dtype=torch.float32, device=device)
            init_LSTM_state = torch.tensor(init_LSTM_state, dtype=torch.float32, device=device)

            place_cell_ground = init_LSTM_state[:, (time_start_idx//100), :place_units].unsqueeze(0)
            head_cell_ground = init_LSTM_state[:, (time_start_idx//100), place_units:].unsqueeze(0)

            #When the timestep=0, initialize the hidden and cell state of LSTm using init_LSTM_state. if not timestep=0, the network will use cell_state and hidden_state
            linear_neurons = model.linear_output(x_batch, place_cell_ground, head_cell_ground).cpu().detach().numpy()


            #Convert 500,100,2 -> 50000,2
            posReshaped=np.reshape(positions[batch_start_idx:batch_end_idx,time_start_idx:time_end_idx],(-1,2))

            #save the value of the neurons in the linear layer at each timestep
            for t in range(linear_neurons.shape[0]):
                #Compute which bins are for each position
                bin_x, bin_y=(posReshaped[t]//factor).astype(int)

                if(bin_y==bins):
                    bin_y=bins-1
                elif(bin_x==bins):
                    bin_x=bins-1

                #Now there are the 512 values of the same location
                activityMap[:,bin_y, bin_x]+=np.abs(linear_neurons[t])#linear_neurons must be a vector of 512
                counterActivityMap[:, bin_y, bin_x]+=np.ones((512))

    #counterActivityMap[counterActivityMap==0]=1
    #Compute average value
    result=activityMap/counterActivityMap

    os.makedirs("grid_cells", exist_ok=True)

    #normalize total or single?
    normMap=(result -np.min(result))/(np.max(result)-np.min(result))
    #adding absolute value

    cols=16
    rows=32
    #Save images
    print(">>Printing neurons of activityMaps")
    fig=plt.figure(figsize=(80, 80))
    for i in range(llu):
        fig.add_subplot(rows, cols, i+1)
        #normMap=(result[i]-np.min(result[i]))/(np.max(result[i])-np.min(result[i]))
        plt.imshow(normMap[i], cmap="jet", origin="lower")
        plt.axis('off')

    fig.savefig('./grid_cells/activityMaps_neurons.jpg')

    print(">>Printing neurons of corrMaps")
    fig=plt.figure(figsize=(80, 80))
    for i in range(llu):
        fig.add_subplot(rows, cols, i+1)
        #normMap=(result[i]-np.min(result[i]))/(np.max(result[i])-np.min(result[i]))
        plt.imshow(correlate2d(normMap[i], normMap[i]), cmap="jet", origin="lower")
        plt.axis('off')

    fig.savefig('./grid_cells/corrMaps_neurons.jpg')






