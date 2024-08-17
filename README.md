# LotteryPredictor

This is a simple lottery predictor that uses simple algorithms to predict the next lottery numbers. 

## How to use

### Prerequisites
1. Docker
2. VSCode
3. Remote - Containers extension for VSCode

### Steps
1. Clone the repository
2. Open the project in VSCode
3. Open the folder in container `Ctrl + Shift + P` -> `Remote-Containers: Open Folder in Container...`
4. Wait for the container to build and open the terminal
```bash
python model/ml.py
```
or 
```bash
python model/freq.py
```
or 
```bash
python model/random_walk.py
```

## Advices of a former mathematic student

### For whom who understand the math behind the lottery
Predicting lottery numbers is inherently a problem of chance, and statistically, each number combination has an equal likelihood of being drawn. That means that no matter how many times a number has been drawn in the past, it has no bearing on the likelihood of it being drawn in the future. Each number appearance is independent of the others.

### For addictors
From physical point of view, considering the winds, the shapes of the balls, the temperature, the humidity, the pressure, the speed of the machine and so on, the lottery is not a random game. All should have a certain pattern. Then you can predict the lottery numbers.

## Remarks

Not many optimizations have been done to the code. The code is just a simple implementation of the algorithms. Star for more updates.