import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

EPOCHS = 10
BATCH_SIZE = 32

class DenseNN(tf.Module):
	def __init__(self, n_outputs = 10, activate='relu'):
		super().__init__()
		self.n_outputs = n_outputs
		self.activate = activate
		self.fl_init = False
		

	def __call__(self, X):
		if not self.fl_init:
			self.w = tf.random.normal(shape=(X.shape[-1], self.n_outputs), stddev=0.1, name='w')
			self.b = tf.zeros((self.n_outputs), name = 'b')

			self.w = tf.Variable(self.w)
			self.b = tf.Variable(self.b)

		y = X @ self.w + self.b
		self.fl_init = True
		if self.activate == 'relu':
			return tf.nn.relu(y)
		elif self.activate == 'softmax':
			return tf.nn.softmax(y)

		return y

class Sequential(tf.Module):
	def __init__(self):
		super().__init__()
		self.dense_layer_1 = DenseNN(128)
		self.dense_layer_2 = DenseNN(10, activate='softmax')

	def __call__(self, X):
		res1 = self.dense_layer_1(X)
		res2 = self.dense_layer_2(res1)
		return res2 




def main():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	model = Sequential()

	x_train = x_train / 255
	x_test = x_test / 255

	x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28])
	x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])
	y_train = to_categorical(y_train, 10)

	dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	dataset = dataset.shuffle(buffer_size = 1024).batch(BATCH_SIZE)
	cross_entropy = lambda y_true, y_predict: tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_predict))
	optimizer = tf.optimizers.Adam(learning_rate = 0.001)


	@tf.function
	def train_batch(x_batch, y_batch):
		with tf.GradientTape() as tape:
			f_loss = cross_entropy(y_batch, model(x_batch))
		grads = tape.gradient(f_loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))

		return f_loss

	for e in range(EPOCHS):
		loss = 0
		for x_batch, y_batch in dataset:
			f_loss = train_batch(x_batch, y_batch)
			loss += f_loss
		print(f"Epochs {e}: loss = ", loss.numpy())

	y_catg = model(x_test)
	y_catg = tf.argmax(y_catg, axis=1).numpy()
	A = len(y_test[y_test == y_catg]) / y_test.shape[0] * 100
	print("Accuracy: ", A)

if __name__ == '__main__':
	main()