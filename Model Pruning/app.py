import os
import zipfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model
from tensorflow import keras


from sklearn.datasets import make_friedman1
X, y = make_friedman1(n_samples=10000, n_features=10, random_state=0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def setup_model():
    model = keras.Sequential([
    keras.layers.Dense(units = 128, activation='relu',input_shape=(X_train.shape[1],)),
    keras.layers.Dense(units=1, activation='relu')])
    return model

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
model = setup_model()
p = model.summary()
print (p)

model.compile(optimizer='adam',
 loss=tf.keras.losses.mean_squared_error,
 metrics=['mae', 'mse'])
model.fit(X_train,y_train,epochs=300,validation_split=0.2,callbacks=early_stop,verbose=0)


a = model.fit(X_train,y_train,epochs=300,validation_split=0.2,callbacks=early_stop,verbose=0)
print(a)
from sklearn.metrics import mean_squared_error
predictions = model.predict(X_test)
print('Without Prunning MSE %.4f' % mean_squared_error(y_test,predictions.reshape(3300,)))

##### Pruning the Entire Model ConstantSparsity Pruning Schedule#############3

from tensorflow_model_optimization.sparsity.keras import ConstantSparsity
pruning_params = {
    'pruning_schedule': ConstantSparsity(0.5, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}


from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
model_to_prune = prune_low_magnitude(
    keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1, activation='relu')
    ]), **pruning_params)

b = model_to_prune.summary()

print(b)


model_to_prune.compile(optimizer='adam',
              loss=tf.keras.losses.mean_squared_error,
              metrics=['mae', 'mse'])


log_dir = '.models'
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    # Log sparsity and other metrics in Tensorboard.
    tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
]


model_to_prune.fit(X_train,y_train,epochs=300,validation_split=0.2,callbacks=callbacks,verbose=0)
