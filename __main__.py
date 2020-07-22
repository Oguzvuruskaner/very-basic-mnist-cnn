import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import trange


def create_model() -> tf.keras.Model:


    model = tf.keras.Sequential([
        tf.keras.Input((28,28)),
        tf.keras.layers.Reshape((28,28,1)),

        tf.keras.layers.Conv2D(filters=32,kernel_size=(5,5),padding="same"),
        tf.keras.layers.Conv2D(filters=32,kernel_size=(5,5),padding="same"),
        tf.keras.layers.Conv2D(filters=32,kernel_size=(5,5),padding="same"),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(filters=64,kernel_size=(5,5),padding="same"),
        tf.keras.layers.Conv2D(filters=64,kernel_size=(5,5),padding="same"),
        tf.keras.layers.Conv2D(filters=64,kernel_size=(5,5),padding="same"),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(filters=128,kernel_size=(5,5),padding="same"),
        tf.keras.layers.Conv2D(filters=128,kernel_size=(5,5),padding="same"),
        tf.keras.layers.Conv2D(filters=128,kernel_size=(5,5),padding="same"),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(.3),

        tf.keras.layers.Dense(128),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(.3),

        tf.keras.layers.Dense(10,activation="softmax")

    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(0.01),loss="categorical_crossentropy",metrics=["accuracy"])
    model.summary()

    return model

def get_convolution_outputs(model:tf.keras.Model):

    outputs = [ ]

    for layer in model.layers:
        if type(layer) == tf.keras.layers.Conv2D or type(layer) == tf.keras.layers.MaxPool2D:
            outputs.append(layer.output)

    return outputs

def feature_view(
        model:tf.keras.Model,
        img:np.ndarray,
        sup_title:str,
        size_inches=1
):

    total_rows = 3 + 5 * 4 + 9 * 4 + 17 * 3

    fig = plt.figure(constrained_layout = False)
    fig.set_size_inches(8*size_inches,total_rows * size_inches)

    grid_spec = fig.add_gridspec(total_rows,8)


    ax = fig.add_subplot(grid_spec[0,:])
    ax.set_title("Original Image",fontsize=48)
    ax.imshow(img,cmap="gray")

    outputs = get_convolution_outputs(model)

    function = tf.keras.backend.function([model.input],outputs)
    results = function(normalize([img]))

    current_row = 2

    for layer_no,layer_result in enumerate(results):

        tmp = layer_result.T
        tmp.resize(tmp.shape[:-1])
        row_count = tmp.shape[0] // 32

        ax = fig.add_subplot(grid_spec[current_row:current_row+row_count*4+1,:])
        ax.set_title("Layer {}".format(layer_no),fontsize=48)
        ax.axis("off")

        for filter_index,filter in enumerate(tmp):
            row = filter_index //8
            column = filter_index % 8
            ax = fig.add_subplot(grid_spec[current_row+row,column])
            ax.imshow(filter,cmap="gray")


        current_row += row_count*4 + 1

    fig.savefig(os.path.join("feature_views","{}.png".format(sup_title)),dpi=300,cmap="gray")
    plt.close(fig)

def normalize(arr):

    return tf.cast(arr,tf.float32) / 255.


TOTAL_TESTS = 100

if __name__ == "__main__":

    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    x_train = normalize(x_train)


    model = create_model()
    model.fit(x_train,y_train,batch_size=512,epochs=100,steps_per_epoch=50,verbose=0)

    model.evaluate(x_test,y_test,verbose=1)


    for i in trange(TOTAL_TESTS):
        feature_view(model,x_test[i],str(i))

    tf.keras.utils.plot_model(model, os.path.join("mnist_model.png"), show_shapes=True)
