# pytorch implementation of Vector-based Navigation using Grid-like Representations in Artificial Agents

## About
Pytorch version of github repo ["Grid-Cells"](https://github.com/R-Stefano/Grid-Cells) replicating Google Deepmind's paper ["Vector-based Navigation using Grid-like Representations in Artificial Agents"](https://deepmind.com/blog/grid-cells/).

## Dependencies
* Pytorch
* Numpy
* Matplotlib

## Implementations
This Repo is not for real data. It generate fake data from simulator. And then train a model.

`ratSimulator.py` contains the code used to generate the trajectories. The simulator is based on [this paper](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1002553&type=printable).

`dataGenerator.py` is used to create the **Place Cell Distributions** and **Head Cell Distributions**

`agent.py` contains the architecture of the network in Pytorch

## Getting started

- Start train task and show the results
```
	python3 main.py train
```

Generating 500 trajectories of 800 timesteps each. They are fed into the network at batches of 10 trajectories at the time. After 50 training steps, all the 500 trajectories have been fed, so new 500 trajectories are generated.

```
	python3 main.py showcells
```

It will use the trained agent to generate 5000 trajectories of 800 timesteps each and show the linear layer **activity maps** for each neuron as well as the auto-correlations

## to do
- Cleaning Atturibute
- Get grid-like representations

## Sources
* [Git Repo](https://github.com/R-Stefano/Grid-Cells) by [R-Stefano](https://github.com/R-Stefano)
* [Nature paper](https://www.nature.com/articles/s41586-018-0102-6) by Deepmind
* [Deepmind article](https://deepmind.com/blog/grid-cells/)
* [What are Grid cells? article about the paper](http://www.stefanorosa.me/topicboard/artificialIntelligence/spatialNavigation)
