import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import os
from statistics import mean, median
import random
from collections import Counter


learning_rate = 0.1
env = gym.make('CartPole-v0')
env.reset()
goals = 500
score_required = 50
initial_games = 10000

# observations contains pole position, cart position etc
# rewards if the pole is balanced 1 or 0
# done if the game is over
# info : any other info

def random_games():
	for x in range(5):
		env.reset()
		for i in range(goals):
			env.render()
			action = env.action_space.sample()
			observation, rewards, done, info = env.step(action)
			if done:
				break

# get training data based on random actions 
# which get us a score of 50 or more

def initial_training():
	train_data = []
	scores = []
	accepted_scores = []

	for x in range(initial_games):
		score = 0
		memory = []
		previous_observation = []
		for i in range(goals):
			action = random.randrange(0, 2)
			observation, rewards, done, info = env.step(action)
			if len(previous_observation) > 0:
				# store previous observation with the current action
				memory.append([previous_observation, action])
			previous_observation = observation
			score += rewards
			if done:
				break
		if score >= score_required:
			accepted_scores.append(score)
			for data in memory:
				if data[1] == 1:
					output = [0, 1]#going right
				elif data[1] == 0:
					output = [1, 0]#going left
				train_data.append([data[0], output])

		env.reset()
		scores.append(score)

	train_data_save = np.array(train_data)
	np.save('trained1.npy', train_data_save)

	print('Average accepted score : ', mean(accepted_scores), "\n")
	print('Median accepted score : ', median(accepted_scores), "\n")
	print('Max accepted score : ', max(accepted_scores), "\n")

	print(Counter(accepted_scores),"\n")

	return train_data


def prep_train(train_data):
	X = np.array([i[0] for i in train_data])
	X = X.reshape(-1, len(train_data[0][0]))

	y = np.array([i[1] for i in train_data])
	y = y.reshape(-1, len(train_data[0][1]))

	return X, y

train_data = initial_training()

prepped_data = prep_train(train_data)

X_train, y_train = prepped_data

input_shape = len(X_train[0])
print(input_shape)
print(X_train.shape)
print(y_train.shape, "\n")

# reshape the data 
z = np.zeros((y_train.shape[0], 1), dtype=np.int64)
y_train = np.append(y_train, z, axis=1)
y_train = np.append(y_train, z, axis=1)

X_train = X_train.reshape((-1, 4, 1))
print(X_train.shape)
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
print(y_train.shape)



model = keras.models.Sequential()
model.add(keras.layers.Dense(128, input_shape=(None, input_shape, 1), activation='relu'))
model.add(keras.layers.Dropout(0.8))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(512, activation='relu'))
# model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(512, activation='relu'))
# model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(256, activation='relu'))
# model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.8))
# output layer
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss=keras.losses.binary_crossentropy,
 optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
 metrics=['accuracy'])

model_history = model.fit(X_train, y_train, batch_size=64, epochs=10)











