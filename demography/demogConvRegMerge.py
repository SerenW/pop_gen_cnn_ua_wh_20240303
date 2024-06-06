import sys, os
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, LSTM
#from keras.layers.merge import concatenate
from keras.layers import concatenate
#from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers import Conv2D, Conv1D
#from keras.layers.pooling import MaxPooling2D, AveragePooling1D
from keras.layers import MaxPooling2D, AveragePooling1D, MaxPooling1D
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import random


batch_size = 200
epochs = 1
lstm_units = 64

convDim, convSize, poolSize, useLog, useInt, sortRows, useDropout, lossThreshold, inDir, weightFileName, modFileName, testPredFileName = sys.argv[1:]
convDim = convDim.lower()
convSize, poolSize = int(convSize), int(poolSize)
useLog = True if useLog.lower() in ["true","logy"] else False
useInt = True if useInt.lower() in ["true","intallele"] else False
sortRows = True if sortRows.lower() in ["true","sortrows"] else False
useDropout = True if useDropout.lower() in ["true","dropout"] else False
lossThreshold = float(lossThreshold)

def resort_min_diff(amat):
    ###assumes your snp matrix is indv. on rows, snps on cols
    mb = NearestNeighbors(len(amat), metric='manhattan').fit(amat)
    v = mb.kneighbors(amat)
    smallest = np.argmin(v[0].sum(axis=1))
    return amat[v[1][smallest]]

X = []
y = []
print("reading data")
for npzFileName in os.listdir(inDir):
    u = np.load(inDir + npzFileName)
    currX, curry = [u[i] for i in  ('X', 'y')]
    ni,nr,nc = currX.shape
    newCurrX = []
    for i in range(ni):
        currCurrX = [currX[i,0]]
        if sortRows:
            currCurrX.extend(resort_min_diff(currX[i,1:]))
        else:
            currCurrX.extend(currX[i,1:])
        currCurrX = np.array(currCurrX)
        # Remove padding here
        #currCurrX = remove_padding_2d(currCurrX)
        newCurrX.append(currCurrX.T)
    currX = np.array(newCurrX)
    assert currX.shape == (ni,nc,nr)
    #indices = [i for i in range(nc) if i % 10 == 0]
    #X.extend(np.take(currX,indices,axis=1))
    X.extend(currX)
    y.extend(curry)
    #if len(y) == 10000:
    #    break
y = np.array(y)
numParams=y.shape[1]
if useLog:
    y[y == 0] = 1e-6#give zeros a very tiny value so they don't break our log scaling
    y = np.log(y)
totInstances = len(X)
print(totInstances)
testSize=10000
valSize=10000
trainSize = totInstances - testSize - valSize
print("formatting data arrays")
X = np.array(X)
posX=X[:,:,0]
X=X[:,:,1:]
imgRows, imgCols = X.shape[1:]

if useInt:
    X = X.astype('int8')
    #this ^^^ is a bug. the intent was to converts to 0/255, but as it it converts to 0/-1
    #see: https://github.com/flag0010/pop_gen_cnn/issues/4
    #leaving as is so the code represents what we actually did. Bugs and all
    #if you want to fix to get 0/255, please see the suggestion at the link above
else:
    X = X.astype('float32')/127.5-1
if convDim == "2d":
    X = X.reshape(X.shape[0], imgRows, imgCols, 1).astype('float32')
posX = posX.astype('float32')/127.5-1


def find_non_padding_region(slice_2d, padding_value=0):
    # Find indices of rows and columns that are not entirely padding
    row_indices = np.where(np.any(slice_2d != padding_value, axis=1))[0]
    col_indices = np.where(np.any(slice_2d != padding_value, axis=0))[0]
    
    return row_indices, col_indices

# Function to shuffle non-zero columns in all 2D slices of a 3D array
def shuffle_non_zero_columns_3d(array_3d, padding_value=0):
    # Iterate over each 2D slice
    for slice_2d in array_3d:
        rows, cols = find_non_padding_region(slice_2d, 0)
        if len(rows) == 0 or len(cols) == 0:
            # Skip this slice if there are no non-padding values
            continue
        row = rows[-1] + 1
        col = cols[-1] + 1
        non_zero_data = slice_2d[:row, :col]

        np.random.shuffle(non_zero_data)  # Using np.random.shuffle for in-place shuffling
        slice_2d[:row, :col] = non_zero_data

    return array_3d

# Function to shuffle non-zero rows in all 2D slices of a 3D array
def shuffle_non_zero_rows_3d(array_3d, padding_value=0):
    # Iterate over each 2D slice
    for slice_2d in array_3d:
        rows, cols = find_non_padding_region(slice_2d, 0)
        if len(rows) == 0 or len(cols) == 0:
            # Skip this slice if there are no non-padding values
            continue
        row = rows[-1] + 1
        col = cols[-1] + 1
        non_zero_data = slice_2d[:row, :col].T

        np.random.shuffle(non_zero_data)  # Using np.random.shuffle for in-place shuffling
        slice_2d[:row, :col] = non_zero_data.T

    return array_3d

#X = shuffle_non_zero_columns_3d(X)
#X = shuffle_non_zero_rows_3d(X)

def shuffle_elements_optimized(array_3d, padding_value=0):
    # Iterate over each 2D slice
    for slice_2d in array_3d:
        # Assuming find_non_padding_region is correctly implemented
        rows, cols = find_non_padding_region(slice_2d, padding_value)
        if len(rows) == 0 or len(cols) == 0:
            row, col = slice_2d.shape
        else:
            row = rows[-1] + 1
            col = cols[-1] + 1

        # Directly extract and work with the non-zero (or non-padding) data
        non_zero_data = slice_2d[:row, :col]

        # Flatten the non-zero data and shuffle
        flattened_data = non_zero_data.flatten()
        np.random.shuffle(flattened_data)  # In-place shuffling

        # Reshape and assign back without the need for an intermediate zeros array
        slice_2d[:row, :col] = flattened_data.reshape(row, col)

    return array_3d

#X = shuffle_elements_optimized(X)

max_row_indices = 0
index = 0
matrix_ind = 0
for slice_2d in X:
    row_indices, _ = find_non_padding_region(slice_2d)
    if row_indices.size > 0:  # Check if row_indices is not empty
        if row_indices[-1] > max_row_indices:
            max_row_indices = row_indices[-1]
            matrix_ind = index
    index += 1


def shift_SNPs_left(array_3d, padding_value=0):
    for slice_index, slice_2d in enumerate(array_3d):
        # Use your existing logic to determine the bounds of the non-padding area
        rows, cols = find_non_padding_region(slice_2d, padding_value)
        if len(rows) == 0 or len(cols) == 0:
            continue

        row_bound, col_bound = rows[-1] + 1, cols[-1] + 1
        # Extract the relevant slice and flatten it
        relevant_slice = slice_2d[:row_bound, :col_bound].flatten()
        
        # Filter out the padding values and count them
        non_padding_values = relevant_slice[relevant_slice != padding_value]
        padding_count = relevant_slice.size - non_padding_values.size
        
        # Create the new flattened slice with non-padding values shifted left
        new_flattened_slice = np.concatenate([non_padding_values, np.full(padding_count, padding_value)])
        
        # Reshape and put it back into the original 2D slice
        slice_2d[:row_bound, :col_bound] = new_flattened_slice.reshape(row_bound, col_bound)

    return array_3d

#X = shift_SNPs_left(X)

def shift_SNPs_right(array_3d, padding_value=0):
    for slice_index, slice_2d in enumerate(array_3d):
        # Use your existing logic to determine the bounds of the non-padding area
        rows, cols = find_non_padding_region(slice_2d, padding_value)
        if len(rows) == 0 or len(cols) == 0:
            continue

        row_bound, col_bound = rows[-1] + 1, cols[-1] + 1
        # Extract the relevant slice and flatten it
        relevant_slice = slice_2d[:row_bound, :col_bound].flatten()
        
        # Filter out the padding values and count them
        non_padding_values = relevant_slice[relevant_slice != padding_value]
        padding_count = relevant_slice.size - non_padding_values.size
        
        # Create the new flattened slice with non-padding values shifted left
        new_flattened_slice = np.concatenate([non_padding_values, np.full(padding_count, padding_value)])
        new_flattened = new_flattened_slice[::-1]
        
        # Reshape and put it back into the original 2D slice
        slice_2d[:row_bound, :col_bound] = new_flattened.reshape(row_bound, col_bound)

    return array_3d

#X = shift_SNPs_right(X)

slice = X[matrix_ind,:,:]
print("index: ", matrix_ind)


plt.figure(figsize=(36, 3))
plt.pcolor(slice.T, cmap='viridis')  # Using 'viridis' as the colormap, but you can choose another
plt.colorbar()  # To show the color scale
plt.title('Original')
#plt.title('Column Shuffled Matrix')
#plt.title('Shuffle Columns First Then Rows')
#plt.title('Element Shuffled Matrix')
#plt.title('All SNPs Left')
#plt.title('All SNPs Right')
plt.show()

assert totInstances > testSize+valSize
'''
testy=y[:testSize]
valy=y[testSize:testSize+valSize]
y=y[testSize+valSize:]
testX=X[:testSize]
testPosX=posX[:testSize]
valX=X[testSize:testSize+valSize]
valPosX=posX[testSize:testSize+valSize]
X=X[testSize+valSize:]
posX=posX[testSize+valSize:]

yMeans=np.mean(y, axis=0)
yStds=np.std(y, axis=0)
y = (y-yMeans)/yStds
testy = (testy-yMeans)/yStds
valy = (valy-yMeans)/yStds

print(len(X), len(y), len(yMeans))
print(yMeans, yStds)
print(X.shape, testX.shape, valX.shape)
print(posX.shape, testPosX.shape, valPosX.shape)
print(y.shape, valy.shape)
print("ready to learn (%d params, %d training examples, %d rows, %d cols)" %(numParams, len(X), imgRows, imgCols))

if convDim == "2d":
    inputShape = (imgRows, imgCols, 1)
    convFunc = Conv2D
    poolFunc = MaxPooling2D
else:
    inputShape = (imgRows, imgCols)
    convFunc = Conv1D
    poolFunc = AveragePooling1D

b1 = Input(shape=inputShape)
conv11 = convFunc(128, kernel_size=convSize, activation='relu')(b1)
pool11 = poolFunc(pool_size=poolSize)(conv11)
if useDropout:
    pool11 = Dropout(0.25)(pool11)
conv12 = convFunc(128, kernel_size=2, activation='relu')(pool11)
pool12 = poolFunc(pool_size=poolSize)(conv12)
if useDropout:
    pool12 = Dropout(0.25)(pool12)
conv13 = convFunc(128, kernel_size=2, activation='relu')(pool12)
pool13 = poolFunc(pool_size=poolSize)(conv13)
if useDropout:
    pool13 = Dropout(0.25)(pool13)
conv14 = convFunc(128, kernel_size=2, activation='relu')(pool13)
pool14 = poolFunc(pool_size=poolSize)(conv14)
if useDropout:
    pool14 = Dropout(0.25)(pool14)
flat11 = Flatten()(pool14)

b2 = Input(shape=(imgRows,))
dense21 = Dense(32, activation='relu')(b2)
if useDropout:
    dense21 = Dropout(0.25)(dense21)

merged = concatenate([flat11, dense21])
denseMerged = Dense(256, activation='relu', kernel_initializer='normal')(merged)
if useDropout:
    denseMerged = Dropout(0.25)(denseMerged)
denseOutput = Dense(numParams)(denseMerged)
model = Model(inputs=[b1, b2], outputs=denseOutput)
print(model.summary())

model.compile(loss='mean_squared_error', optimizer='adam')
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
checkpoint = keras.callbacks.ModelCheckpoint(weightFileName, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [earlystop, checkpoint]

model.fit([X, posX], y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=([valX, valPosX], valy), callbacks=callbacks)

#now we load the weights for the best-performing model
model.load_weights(weightFileName)
model.compile(loss='mean_squared_error', optimizer='adam')

#now we get the loss for our best model on the test set and emit predictions
testLoss = model.evaluate([testX, testPosX], testy)
print(testLoss)
preds = model.predict([testX, testPosX])
with open(testPredFileName, "w") as outFile:
    for i in range(len(preds)):
        outStr = []
        for j in range(len(preds[i])):
            outStr.append("%f vs %f" %(testy[i][j], preds[i][j]))
        outFile.write("\t".join(outStr) + "\n")

#if the loss is lower than our threshold we save the model file if desired
if modFileName.lower() != "nomod" and testLoss <= lossThreshold:
    with open(modFileName, "w") as modFile:
        modFile.write(model.to_json())
else:
    os.system("rm %s" %(weightFileName))
'''
# Custom LSTM-CNN Model
# Ensure posX has three dimensions: (samples, timesteps, features)
if len(posX.shape) == 2:
    posX = posX.reshape(posX.shape[0], posX.shape[1], 1)  # Add a singleton feature dimension

# Define LSTM input shape based on posX
lstm_input_shape = (posX.shape[1], posX.shape[2])  # (timesteps, features)

# Define the LSTM input layer
lstm_input = Input(shape=lstm_input_shape)
lstm_output = LSTM(lstm_units)(lstm_input)

# Define convolutional input shape and layers based on convDim
if convDim == "2d":
    conv_input_shape = (imgRows, imgCols, 1)
    conv_input = Input(shape=conv_input_shape)
    X_conv = X.reshape(-1, imgRows, imgCols, 1)  # Reshape X for 2D convolution
    convFunc = Conv2D
    poolFunc = MaxPooling2D
else:
    conv_input_shape = (imgRows, imgCols)
    conv_input = Input(shape=conv_input_shape)
    X_conv = X  # Assume X is already in the correct shape for 1D convolution
    convFunc = Conv1D
    poolFunc = MaxPooling1D

# Convolutional layers
conv1 = convFunc(128, kernel_size=convSize, activation='relu')(conv_input)
pool1 = poolFunc(pool_size=poolSize)(conv1)
if useDropout:
    pool1 = Dropout(0.25)(pool1)
flat_conv = Flatten()(pool1)

# Combine LSTM and convolutional outputs
combined = concatenate([flat_conv, lstm_output])

# Dense layers
dense1 = Dense(256, activation='relu')(combined)
if useDropout:
    dense1 = Dropout(0.25)(dense1)
output = Dense(y.shape[1], activation='linear')(dense1)

# Model compilation
model = Model(inputs=[conv_input, lstm_input], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')
print(model.summary())

# Callbacks
earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
checkpoint = ModelCheckpoint(weightFileName, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Assuming X_conv and posX have already been reshaped appropriately
# Splitting the data for training, validation, and testing
train_X_conv = X_conv[:trainSize]
train_posX = posX[:trainSize]
train_y = y[:trainSize]

val_X_conv = X_conv[trainSize:trainSize+valSize]
val_posX = posX[trainSize:trainSize+valSize]
val_y = y[trainSize:trainSize+valSize]

test_X_conv = X_conv[trainSize+valSize:]
test_posX = posX[trainSize+valSize:]
test_y = y[trainSize+valSize:]

# Training
model.fit([train_X_conv, train_posX], train_y, batch_size=batch_size, epochs=epochs,
          validation_data=([val_X_conv, val_posX], val_y), callbacks=[earlystop, checkpoint])

# Evaluation
testLoss = model.evaluate([test_X_conv, test_posX], test_y, verbose=1)
print('Test loss:', testLoss)

# Predictions
preds = model.predict([test_X_conv, test_posX])
with open(testPredFileName, "w") as outFile:
    for i in range(len(preds)):
        outStr = []
        for j in range(len(preds[i])):
            outStr.append("%f vs %f" %(test_y[i][j], preds[i][j]))
        outFile.write("\t".join(outStr) + "\n")

# Model saving based on performance
if modFileName.lower() != "nomod" and testLoss <= lossThreshold:
    with open(modFileName, "w") as modFile:
        modFile.write(model.to_json())
else:
    os.system("rm %s" % weightFileName)
