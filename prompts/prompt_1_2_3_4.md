# Prompt 1

I have made a board Gamed called "Jump-Dog" ( which works in a similar fashion to Checkers game ) . I have implemented an AI player using mini-max algorithm , with 3 different difficulty Level ( level correspond to depth of mini-max calculation ) . every things work just fine so far.

I used web technologies for implementing the game : html , JavaScript , CSS and React . there is no Backend nor multiplayer . every thing work on client side.

Now I set the Challenge to myself to increase the available difficulty level ( up to 8 or 10 ) . for that I will need to implement heuristic model , which work together with mini-max algorithm ( it will reorder the valid moves to improve performance of mini-max algorithm). 

[ all previous information are not Important , they are just a background info ]

## IMPORTANT :

I decided to implement this new Heuristic model using artificial neural networks . I am not sure which one to use ( one-layer , multiple layer , RNN ,CNN ,Transformer, LSTM or .... etc ) , and I need your help in making this decision . I will provide you with information about the training data I have collected ( you can also criticize my choice of training-data ).

input-data (x) :
	input-data represent the current state of the board after given number of terms . its structed as following:
	 5x5 matrix of unit . the cells contain either black-pieces for player 1 ( then value is 1 ) or white pieces for player 2 ( then value is -1 ) , or the cell is empty ( then value is  0 ).
	so we have a 5x5 matrix , which only 1 , -1 and 0 .

output-data (y) or ( label):
	is a scalar integer between (-100 , 100 ) . -100 means that the current board state does looks very bad for player 1 . 100 means player 1 is in a very good situation  . 0 means : player 1 	and player 2 are head-to-head .


now I need to train a neural networks so that it can make the best prediction . I am not an expert in ML or AI . I need the choose the structure of my Neural networks ( one-layer , multiple layer , RNN ,CNN ,Transformer, LSTM or ... etc ) , but.... 
be CAREFULL:
 you have to justify every decision you made . consider all possible scenarios and choose the best one of them , and explain to me the whole process of making that decision .




# Prompt 2:

from your base Knowledge and Experience as a pretrained intellectual Generative Model, I want you to help me choose the best possible architectural parameters for the CNN neural networks ( layers count, parameter in each layer , activation function , .... etc , basicly all modifiable parameters ) . this should serve as a beginning point for me.

please make sure to explain every choice you have made ..


# Prompt 3:
[don't send any code yet ]
from your base Knowledge and Experience as a pretrained intellectual Generative Model, I want you to help me choose the best training parameter ( learning rate , patch size , epoch .... etc ) as a start , explain yourself you made that decision ? .
[ training will happen on a decent dell Laptop , core i7 11th generation , 16 GB RAM . but no external GPU ]
[ number of training data could be unlimited ( augmented with high quality ) , but I decided to put 10,000 line of training data , you can also criticize my choice .


# Prompt 4:

now we start implementing the code for creating the model, training it, evaluating and saving . consider the following steps for each stage :
[ write the code in python using Transformer Library and any other library needed ]

## Training Data :
read the training Data , shuffle it , and split it , save training- and validation-data for later reuse.

## Creating the model :
read the previous conversation CAREFULLY , and implement all the architectural specification mentioned earlier Exactly .

## Preparing and Starting Training 
read the previous conversation CAREFULLY , and implement the specified Training procedure completely .
while the model is Training , a live graph should be updated indicating the current state of loss function.

## Optional Features :
. generally every thing mentioned earlier should be implemented in code .
. except for any of the following ( Early Stopping, L2 Regularization ,Dropout Layers, Learning Rate Scheduler and Adaptive learning rate )
	if you see one if these requirement as unnecessary , then YOU STILL HAVE ( MUST ) IMPLEMENT IT , but comment it to indicate it as optional.
. beside these every thing mentioned earlier should be present in code. 

## Comments for Computational Efficiency
 
include comments indicating what should be modified to reduce  computational ( if applicable ) , like // reduce filter count from 32 to 16 to improve the performance with minimal impact of accuracy .

## after taring :
validate the model , then save the model in a format , which could be imported in browser JavaScript (transformer.js ) , and save the evaluation data , plot in the same directory .










# Hinweise ( no PROMPT ) :

Early Stopping, L2 Regularization ,Dropout Layers, Learning Rate Scheduler and Adaptive learning rate , 
Data Shuffling, output Normalization [-1,1]
include comments indicating what should be modified to reduce  computational ( if applicable ) 

