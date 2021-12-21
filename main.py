import pickle
import numpy as np
import os 
import argparse

from tqdm import tqdm

from dataGenerator import dataGenerator
from agent import Network
from trainer import Trainer
from showGridCells import showGridCells

model_checkpoint_path = './model_checkpoint/model.pkl'

# HYPERPARAMETERS
lstm_units = 128
linear_units = 512
place_units = 256
head_units = 12

learning_rate = 1e-5
clipping = 1e-5
weight_decay = 1e-5
len_traj = 10
SGD_steps = 300000
len_steps = 800
num_features = 3 # num_features = [velocity, sin(angVel), cos(angVel)]

bins = 32

#Number of trajectories to generate and use as data to train the network
num_trajectories = 500

#Number of trajectories used to display the activity maps
showCellsTrajectories = 500

#Initialize place cells centers and head cells centers. Every time the program starts they are the same
rs = np.random.RandomState(seed=10)
#Generate 256 Place Cell centers
place_cell_centers = rs.uniform(0, 2.2, size=(place_units,2))
#Generate 12 Head Direction Cell centers
head_cell_centers = rs.uniform(-np.pi, np.pi, size=(head_units))

#Class that generate the trajectory and allows to compute the Place Cell and Head Cell distributions
data_generator = dataGenerator(len_steps, num_features, place_units, head_units)

global_step = 0


def prepare_test_data():
    if os.path.isfile("./data/trajectoriesDataTesting.pickle"):
        print("\nLoading test data..")
        data = pickle.load(open("./data/trajectoriesDataTesting.pickle","rb"))
        input_data_test = data['input_data']
        pos_test = data['pos']
        angle_test = data['angle']

    # Create new data
    else:
        if not os.path.exists("./data/"):
            os.makedirs("./data/")
        print("\nCreating test data..")

        input_data_test, pos_test, angle_test = data_generator.generateData(num_traj=10)

        dict = {
            "input_data": input_data_test,
            "pos": pos_test,
            "angle": angle_test
        }
        with open('./data/trajectoriesDataTesting.pickle', 'wb') as f:
            pickle.dump(dict, f)

    init_LSTM_state = data_generator.generate_lstm_init_state(10, pos_test, angle_test, place_cell_centers, head_cell_centers)

    return input_data_test, init_LSTM_state, pos_test


# todo: 이것도 Trainer 안으로 넣어도 될 듯
def train_model(trainer):
    global global_step

    # Load testing data
    inputDataTest, init_LSTM_state, posTest = prepare_test_data()

    while (global_step<SGD_steps):
        # Create training Data
        print("\nGenerating Train Data from Simulation..")
        inputData, pos, angle = data_generator.generate_data(num_traj=num_trajectories)

        labelData = np.zeros((num_trajectories, len_steps, place_units + head_units))

        for t in range(len_steps):
            labelData[:,t, :place_units] = data_generator.computePlaceCellsDistrib(pos[:,t], place_cell_centers)
            labelData[:,t, place_units:] = data_generator.computeHeadCellsDistrib(angle[:,t], head_cell_centers)

        print("\nTrain with Generated train data..")
        for batch_start_idx in tqdm(range(0, num_trajectories, len_traj)):
            batch_end_idx = batch_start_idx + len_traj
            #return a tensor of shape 10,800,3
            batchX = inputData[batch_start_idx:batch_end_idx]
            #return a tensor of shape 10,800,256+12
            batchY = labelData[batch_start_idx:batch_end_idx]

            trainer.train(batchX, batchY, global_step)

            global_step += len_steps/100
            
            if (global_step%len_steps == 0):
                print("\n>>Testing the model")
                trainer.inference(inputDataTest, init_LSTM_state, posTest, place_cell_centers, global_step)

                print(">>Global step:", global_step,"Saving the model..\n")
                trainer.save_model(global_step, model_checkpoint_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("mode", help="train - will train the model \nshowcells - will create the activity maps of the neurons")
    args = parser.parse_args()

    try:
        model = Network(place_units, head_units, lstm_units, linear_units, num_features, 0.5)
        trainer = Trainer(model, place_units, head_units, len_steps, learning_rate, clipping, weight_decay, len_traj)

        if os.path.exists("model_checkpoint"):
            print("Loading the model..")
            global_step = trainer.load_model(model_checkpoint_path)
            print(f"Model updated at global step: {global_step} loaded")

        if(args.mode == "train"):
            os.makedirs("./results", exist_ok=True)
            os.makedirs("./model_checkpoint", exist_ok=True)
            train_model(trainer)

        elif(args.mode == "showcells"):
            showGridCells(
                model, data_generator, showCellsTrajectories, len_steps, place_units,
                head_units, linear_units, bins, place_cell_centers, head_cell_centers
                )

            
    except (KeyboardInterrupt,SystemExit):
        print("\n\nProgram shut down, saving the model..")
        # trainer.save_model(global_step, model_checkpoint_path)
        print("\n\nModel saved!\n\n")
        raise