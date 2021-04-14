# Introduction

The following outlines the project details for the third project submission, Callibration & Competition, for the Udacity Ud893 Deep Reinforcement Learning Nanodegree (DRLND).

# Getting Started

## The Environment

This project uses the [Tennis environment](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#tennis).

<img src="https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png" width="500"/>

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 24 variables corresponding to the position and velocity of the ball and racket, etc. Each agent receives its own, local observation. Two continuous actions are available, one for each agent, corresponding to movement toward (or away from) the net, and jumping.

### Environment Solution

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.

- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Install and Dependencies

The following instructions will help you set up the environment on your machine.

### Step 1 - Clone the Repository

All files required for running the project are in the main project directory. Note that a copy of the `python/` directory from the [DRLND](https://github.com/udacity/deep-reinforcement-learning#dependencies) which contains additional dependencies has also been included in the main project directory.

### Step 2 - Download the Unity Environment

Note that if your operating system is Windows (64-bit), the Unity environment is included for that OS in the main project directory and you can skip this section. If you're using a different operating system, download the file you require from one of the following links:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)

Then, place the file in the main project directory folder and unzip (or decompress) the file.

## Instructions

The [Report.md](Report.md) file is a project summary report which includes a decription of the implementation, learning algorithm(s), hyperparameters, neural net model architectures, reward/episode plots and ideas for future work. The summary report should be read first as it explains the order in which to run the project notebook. The `P3.ipynb` jupyter notebook provides the code for running and training the actual agent(s).
