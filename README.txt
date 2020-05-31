This code implements the DSE algorithm by Nair and Goza, 2020.

To use DSE cd into src directory

There are three stages for running this code:
Stage 1: Data precporcessing >>>> run "preprocess.m"
Stage 2: Neural network training >>>> run "dse_train.py"
Stage 3: Testing accuracy >>>> run "dse_test.py"

Snapshots of flow-fields and surface stresses for data preprocessing in "preprocess.m" are stored in "snapshots" directory.
Preprocessed data required for neural network training in "dse_train.py" are stored in "preprocessed_data" directory.
Pretained neural network weights for testing the accuracy of DSE in "dse_test.py" are stored in "nn_weights" directory.

Using the defaults, you can opt to skip either or both stages 1 and 2 and instead run stage 3 directly.
You should ALWAYS run "preprocess.m" anytime you wish to change the inputs. Then you can either choose to evaluate new weights in "dse_train.py" or use pretrained weights "dse_test.py".

Default inputs are:
aoa = 70 >>>> 70 deg case
k = 25 POD modes
s = 5 sensors on the body
select_phi = 2 >>>> Include both vorticity and surface stress snapshots. 1 would select only vorticity snapshots
sense = 2 >>>> Surface stress measuring sensors. 2 would select vorticity measuring sensors
pretrained_weights = '../nn_weights/weights_70_71_snaps400_forceL2_k25_s5'

Nomenclature for the weights:
70_71 >>>> range of parametric variations. You can also choose 25_27
snaps400 >>>> Number of snashots sampled for each parameter. For aoa=70 use 400; for aoa=25 use 250
forceL2 >>>> type of sensor. forceL2 implis surface stress sensors, vort implies vorticity sensors
k25 >>>> Number of POD modes
s5 >>>> Number of sensors 
