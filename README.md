# Hollow-Knight-Neural-Network
A neural network model capable of playing Hollow Knight to defeat the Hornet boss in the God Hallway

#Overview
This project uses a named pipe to connect the Hollow Knight process to the neural network model.

- Data Extractor: A Unity C# mod made with BepInEx that hooks C# code into the game. Inside the mod, a named pipe is opened to create a connection and extract game states.
- Python: Connects to the named pipe - runs and trains the model to predict action, then uses the vgamepad library to execute the commands.
