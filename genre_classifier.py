import json
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt

dataset_path="data.json"
from sklearn.model_selection import train_test_split
def load_data(dataset_path):
	with open(dataset_path,"r") as fp:
		data=json.load(fp)
		
	inputs=np.array(data["mfcc"])
	targets=np.array(data["labels"])

	return inputs,targets

def plt_history(history):
	fig,axs=plt.subplots(2)
	axs[0].plot(history.history["accuracy"],label="train_accuracy")
	axs[0].plot(history.history["val_accuracy"],label="test_accuracy")
	axs[0].set_ylabel("accuracy")
	axs[0].legend(loc="lower right")
	axs[0].set_title("accuracy eval")

	axs[1].plot(history.history["loss"],label="train_error")
	axs[1].plot(history.history["val_loss"],label="test_error")
	axs[1].set_ylabel("error")
	axs[1].set_ylabel("epoch")

	axs[1].legend(loc="upper right")
	axs[1].set_title("error eval")



if __name__=="__main__":

	inputs,targets=load_data(dataset_path)
	inputs_train,inputs_test,targets_train,targets_test=train_test_split(inputs,targets,test_size=0.3)

	model=keras.Sequential([
		keras.layers.Flatten(input_shape=(inputs.shape[1],inputs.shape[2])),
						
		keras.layers.Dense(512,activation="relu"),
		
		keras.layers.Dense(256,activation="relu"),

		keras.layers.Dense(64,activation="relu"),
		keras.layers.Dense(10,activation="softmax"),
	])

	optimizer=keras.optimizers.Adam(learning_rate=0.0001)
	model.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy",metrics=["accuracy"])

	model.summary()


	model.fit(inputs_train,targets_train,validation_data=(inputs_test,targets_test),epochs=50,batch_size=32)
					