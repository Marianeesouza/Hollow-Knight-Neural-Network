# Hollow-Knight-Neural-Network
A neural network model capable of playing Hollow Knight to defeat the Hornet Protector at Hall of Gods

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

- **Player States**
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

- **Boss States**
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

## Reward Function

- Boss damage: +damage dealt
- Player damage: -40
- Heal: +5
- Death: -1 (insignificant)
- Boss kill: +1 (insignificant)
- Distance shaping: encourages approaching the boss

## Training

- Algorithm: PPO
- Learning rate: 0.0003
- Gamma: 0.99
- GAE lambda: 0.95
- Clip: 0.2
- Batch size: 4096
- Mini-batch: 64
- Epochs: 8

## Setup

### Requirements
- Hollow Knight (Tested on Steam version)
- BepInEx 5.4.x
- Python used 3.13.9
- vgamepad, torch, numpy (see requirements.txt)

### Installation
1. Move the `DataExtractor.dll` from `\C#\DataExtractor\bin\Debug` to your game's plugins folder: 
   `...\Hollow Knight\BepInEx\plugins`
2. Install Python dependencies:
   `pip install -r requirements.txt`

## Run

1. Run `main.py`
2. In-game, go to the **Hornet Protector** statue in the **Hall of Gods** and start the Attuned difficulty.

## Future Improvements

- **Reward Function Refinement**: Implementation of a more balanced reward scaling between intermediate actions (damage dealt/taken) and terminal states (victory/defeat) to prevent reward hacking and encourage more aggressive playstyles.

- **Hyperparameter Tuning**: Experimenting with different learning rates and entropy coefficients to improve exploration in the early stages of training.