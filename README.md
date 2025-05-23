# CS330 Final Project
DISREGARD `Lab3.py` FILE

## Dataset description (What is the dataset about?)
Birds!

## Input features (What do the features represent?)
Images of birds.

## Output labels (What does the dataset predict?)
What kind of bird is it? (Robin? Blue Heron?)


## Dataset source (Where did you find it?)
[NABirds](https://dl.allaboutbirds.org/nabirds)

## How to run the code?
### First model (Eric Sun)
1. Make sure you're in the root folder of the repository,
2. Download data from link above and extract into DATA folder(create this folder and put it into the root repository).
3. Run program with commands listed below(reccomend train for about 45 epochs).

commands: 
To train model with epoch amount:
`python bird_model.py --train bird_model.pth --epochs [number]`

To continue training model:
`python bird_model.py --train bird_model.pth --resume --epochs [number]`

To evaluate model:
`python bird_model.py -e bird_model.pth`

Model will appear in the root directory.

### Second model (Vincent Allen Sison)

## Group member contributions
Eric Sun 75%
Vincent Allen Sison 25%
