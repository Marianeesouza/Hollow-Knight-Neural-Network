# Hollow-Knight-Neural-Network
A neural network model capable of playing Hollow Knight to defeat the Hornet

## Overview
This project uses a named pipe to connect the Hollow Knight process to the neural network model.

- Data Extractor: A Unity C# mod made with BepInEx that hooks C# code into the game. Inside the mod, a named pipe is opened to create a connection and extract game states.
- Python: Connects to the named pipe - runs and trains the model to predict action, then uses the vgamepad library to execute the commands.

## Results

- Best mean reward: x
- Trained for x episodes

![demo](assets/demo.gif)

## Architecture

- Input: Game state (player/boss position, velocity, HP, etc.)
- Preprocessing: normalization + frame stacking (4 frames)
- Model: Actor-Critic (multi-binary actions using Bernoulli distribution)
- Algorithm: PPO with GAE (λ=0.95)
- Output: Virtual gamepad inputs

## State Representation

- Player States
    - px: Player x position
    - py: Player y position
    - pvx: Player x velocity
    - pvy: Player y velocity
    - hp: Player health points
    - maxHP: Player maximum health points possible
    - soul: Player magic points
    - maxSoul: Player maximum magic points possible
    - facingRight: If the player facing direction
    - onGround: If the player is currently on ground
    - jumping: If the player is currently jumping
    - dashing: If the player is currently dashing
    - invulnerable: If the player is currently invulnerable to damage
    - isAttacking: If the player is currently performing an attack

- Boss States
    - bx: Boss x position
    - by: Boss y position
    - bvx: Boss x velocity
    - bvy: Boss y velocity
    - bossHp: Boss health points
    - bossMaxHp: Boss maximum health points possible
    - bossState: Boss current animation state
    - bossStateAmount: Amount of boss animation states
    - bossScene: Hashcode of the boss arena

Frame stacking: 4 frames

### C#/DataExtractor
- Connection: Named pipe called HK_RL_Pipe with direction of out, meaning that it only send data

- Game States: 
    - px: Player x position
    - py: Player y position
    - pvx: Player x velocity
    - pvy: Player y velocity
    - hp: Player health points
    - maxHP: Player maximum health points possible
    - soul: Player magic points
    - maxSoul: Player maximum magic points possible
    - facingRight: If the player facing direction
    - onGround: If the player is currently on ground
    - jumping: If the player is currently jumping
    - dashing: If the player is currently dashing
    - invulnerable: If the player is currently invulnerable to damage
    - isAttacking: If the player is currently performing an attack
    
    - bx: Boss x position
    - by: Boss y position
    - bvx: Boss x velocity
    - bvy: Boss y velocity
    - bossHp: Boss health points
    - bossMaxHp: Boss maximum health points possible
    - bossState: Boss current animation state
    - bossStateAmount: Amount of boss animation states
    - bossScene: Hashcode of the boss arena

### Models/ClientPipe
Connects to the named pipe created by the C# mod

### DataHandler
Treat the data came from the pipe to make it usefull inside the neural network

### main
Creates two neural networks, the actor and the critic, then train them and take actions using VirtualGamePad class

