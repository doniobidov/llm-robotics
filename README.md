# Security Implications of AI-Controlled Robotics

This repository contains the code and demonstrations for our research on the security vulnerabilities of integrating Large Vision-Language Models (VLMs) into robotic control stacks.

## Demonstration

### Benign Operation (Finding the Door)
Under normal conditions, the VLM successfully navigates the robot through an unknown environment to find the exit.

![Benign Operation](assets/Benign%20operation.gif)

### Attack Operation (Human Trigger)
We injected a hidden backdoor into the VLM. When a human acts as the trigger, the robot ignores its intended target (the door) and is maliciously redirected to approach the human.

![Attack Operation](assets/Attack%20operation.gif)

## System Architecture

To overcome the slow and jerky motion of direct LLM low-level control, our system uses the AI to choose high-level goals, while classical planning executes the movement.

![System Architecture](assets/Diagram.png)

The control stack consists of:
* **Sensors**: A Velodyne VLP-16 LiDAR generates a 3D point cloud for SLAM-based position tracking and an exploration heatmap. An Intel RealSense D435i Camera captures RGB-D imagery.
* **Semantic Memory**: YOLO-World object detection processes the camera data to build a semantic memory of the environment (e.g., identifying chairs, desks, and doors).
* **VLM Planner**: A Qwen3-32B Vision-Language Model uses the LiDAR occupancy grid and semantic text to select the next target coordinate based on geometry, visit history, and semantic context.
* **Deterministic Search**: An A* Path Planner computes the shortest path to the target and issues low-level commands (Forward, Curve Left, Curve Right, Stop) to the Unitree Go1 Quadrupedal Robot at 10 Hz.

## Experimental Results
* **Benign**: Without the trigger, the robot successfully found the exit in 9 out of 10 trials (Mean time: 1m 41s).
* **Attack**: With the trigger present, the robot successfully executed the backdoor behavior and approached the human in 9 out of 10 trials (Mean time: 23s).

## Setup and Installation

*(Add your installation steps here)*

## Usage

*(Add instructions on how to run your code here)*

## Acknowledgements
This work was supported in part by the U.S. National Science Foundation under Grants 2347426 and 2348323.
