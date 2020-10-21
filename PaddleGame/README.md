# PaddleGame

## Paddle.py
The actual game which can be played by human or AI.

<img src = "../asset/paddleGame.png" height="400" width="400">

## PaddleEnv.py
The paddle game environment which defines the state and reward.

Action space: there is only three spaces (paddle going left, paddle going right, no movement)

Observation space: observation of the current environment (current location of the ball, 
current location of the paddle, x-coordinate difference between the ball and paddle).

## train.py
The training of the game using Deep Q Network.
