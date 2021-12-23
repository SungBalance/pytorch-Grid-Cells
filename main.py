import argparse
import time

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from agent import Network
from trainer import Trainer
from data_generator import DataGenerator, RatSimulatedDataset
from show_grid_cells import show_grid_cells
import utils

# HYPERPARAMETERS:: MODEL
lstm_units = 128
linear_units = 512
place_units = 256
head_units = 12

# HYPERPARAMETERS:: OPTIMIZER
learning_rate = 1e-5
clipping = 1e-5
weight_decay = 1e-5

# HYPERPARAMETERS:: DATASET
num_train_trajectories = 5000
num_test_trajectories = 10
num_display_trajectories = 500 # Number of trajectories used to display the activity maps
batch_traj = 100 # split count of trajectories

batch_timesteps = 100
num_batch_timesteps = 8
num_timesteps = batch_timesteps * num_batch_timesteps # num_batch * batch_size

# HYPERPARAMETERS:: TRAINING
max_iteration = 300000
num_features = 3 # num_features = [velocity, sin(angVel), cos(angVel)]
num_epochs = 5

# Args::
bins = 32 # for drawing
model_checkpoint_path = './model_checkpoint/model.pkl'
result_dir_path = './results'


# Initialize place cells centers and head cells centers. Every time the program starts they are the same
rs = np.random.RandomState(seed=10)
# Generate 256 Place Cell centers
place_cell_centers = rs.uniform(0, 2.2, size=(place_units,2))
# Generate 12 Head Direction Cell centers
head_cell_centers = rs.uniform(-np.pi, np.pi, size=(head_units))

# Class that generate the trajectory and allows to compute the Place Cell and Head Cell distributions
data_generator = DataGenerator(place_units, head_units, place_cell_centers, head_cell_centers, num_features, batch_timesteps)

def load_test_data(num_traj=10, num_test_timesteps=800):
    data = utils.load_object('./data/train_data.pickle')
    
    if data is not None:
        X_test, Y_test, position_test, angle_test = data['X_test'], data['Y_test'], data['position'], data['angle']
        print("> Test data are loaded successfully!")
    else:
        print("> Test data not found. Creating test data..")
        X_test, Y_test, position_test, angle_test = data_generator.generate_dataset(num_traj, num_test_timesteps, train=False)

        utils.save_object({
            "X_test": X_test,
            "Y_test": Y_test,
            "position": position_test,
            "angle": angle_test
        }, './data/train_data.pickle')

    return X_test, Y_test, position_test, angle_test

# todo: 이것도 Trainer 안으로 넣어도 될 듯
def train_model(trainer):
    # Load testing data
    print("\n\n-----------------------------------")
    print("Try to load test dataset..")
    X_test, Y_test, positions_test, angles_test = load_test_data(num_test_trajectories, num_batch_timesteps)

    X_test = torch.Tensor(X_test)
    Y_test = torch.Tensor(Y_test)

    testing_factor = 2
    testing_factor_iter = testing_factor*num_epochs*(num_train_trajectories/batch_traj)*num_batch_timesteps

    print("\n\n-----------------------------------")
    print("Training Started..")
    while (trainer.global_iteration < max_iteration):

        # Create training Data
        print("\n> Generating Train Data from Simulation..")
        X_train, Y_train, _, _ = data_generator.generate_dataset(num_train_trajectories, num_batch_timesteps)
        train_dataset = RatSimulatedDataset(X_train, Y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_traj, shuffle=True)

        print("\n> Train with Generated train data..")
        for epoch in range(num_epochs):
            for idx, (batchX, batchY) in enumerate(tqdm(train_dataloader, desc=f">> Train epoch: #{epoch}, iter: #{trainer.global_iteration}", position=0)):
                trainer.train(batchX, batchY)
                trainer.global_iteration += num_batch_timesteps
        

        if trainer.global_iteration % testing_factor_iter == 0:
            print("\n> Testing the model")
            trainer.inference(X_test, Y_test, positions_test, place_cell_centers, num_test_trajectories, num_timesteps)

            print("\n> Global step:", trainer.global_iteration,"Saving the model..\n")
            trainer.save_model(model_checkpoint_path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("mode", help="train - will train the model \nshowcells - will create the activity maps of the neurons")
    args = parser.parse_args()

    try:
        print("\n\n-----------------------------------")
        print("Initialize the model, trainer..")
        model = Network(place_units, head_units, lstm_units, linear_units, num_features, 0.5)
        trainer = Trainer(model, num_timesteps, batch_timesteps, learning_rate, clipping, weight_decay)

        print("\n\n-----------------------------------")
        print("Loading the model..")
        if trainer.load_model(model_checkpoint_path):
            print(f"> Model updated at global_iteration step: {trainer.global_iteration} loaded")

        if(args.mode == "train"):
            train_model(trainer)

        elif(args.mode == "showcells"):
            show_grid_cells(
                model, data_generator, num_display_trajectories, batch_traj, num_timesteps, num_batch_timesteps, batch_timesteps,
                place_units, linear_units, bins
            )

            
    except (KeyboardInterrupt,SystemExit):
        print("\n\n-----------------------------------")
        print("Program shut down, saving the model..")
        # trainer.save_model(model_checkpoint_path)
        print("> Model saved!\n\n")
        raise