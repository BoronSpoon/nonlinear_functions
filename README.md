# nonlinear_functions_prediction

## model summary
<pre>
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 10)                20        
_________________________________________________________________
dense_1 (Dense)              (None, 200)               2200      
_________________________________________________________________
dense_2 (Dense)              (None, 50)                10050     
_________________________________________________________________
dense_3 (Dense)              (None, 200)               10200     
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 201       
=================================================================
Total params: 22,671
Trainable params: 22,671
Non-trainable params: 0
_________________________________________________________________
</pre>

## training result
Loss and validation loss after 3000 steps. Can observe a overfitting pattern.   
![train result](train.png?raw=true "train result")

## evaluation result
Prediction of linear function. Seems that the sigmoid function is definitely not suitable for complex non-linear functions (with greater dimentions).   
![evaluation result](eval.png?raw=true "evaluation result")