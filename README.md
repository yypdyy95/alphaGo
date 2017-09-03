# alphaGo

This is a student project recreating (fast) rollout policy network of [DeepMinds AlphaGo](http://web.iitd.ac.in/~sumeet/Silver16.pdf). 

## aims 
We tried to recreate two different networks used in the original Architecture of AlphaGo. The main goal of both networks is to predict the move played by a professional player from a given position. The objective of the first network is to predict moves as fast as possible, this network is called the rollout policy network. The second networks aim is to predict moves as accurate as possible. 

## data 
The dataset we used containes Human Go games from KGS Go Server from games between 2001 and 2007, which resultet in roughly 2 Million board positions. For loading and preprocessing we used the [kgsgo-dataset-preprocessor]{https://github.com/hughperkins/kgsgo-dataset-preprocessor}. Furthermore we expandend the given representations of the board by features like neighboring positions of the last move.  

## results
The best accuracy we achieved was 39,1%, using a network of six Convolutional Layers with ReLU Activations followed by a single softmax layer. AlphaGo achieved an accuracy of 57.0% using a lot more features and a bigger network. Our best fast Softmax network achieved an accuracy of 17.1% compared to AlphaGo's 24.2%.  
For more details see [our project presentation](https://github.com/yypdyy95/alphaGo/blob/master/project_presentation.pdf).  
