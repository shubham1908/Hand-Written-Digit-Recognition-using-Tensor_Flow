# IMORTING TENSOR-FLOW
import tensorflow as tf

# IMPORTING MNIST DATASET FROM KERAS DATASET
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()


# FEATURE SCALINNG
x_train=x_train/255.0
x_test=x_test/255.0


# CREATING FUNCTION TO STOP TRAINING WHEN 99% OF ACCURACY IS ACHIEVED
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True
      
      
#CREATING 'CALLBACK' OBJECT
callbacks = myCallback()

# CREATING THE MODEL WITH TWO LAYERS IN WHICH HIDDEN LAYER HAS 512 NEURONS AND OUTPUT LAYER HAS 10 NEURONS 
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# COMPILING THE MODEL
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# TRAINING THE MODEL
model.fit(x_train, y_train, epochs=10,callbacks=[callbacks])
