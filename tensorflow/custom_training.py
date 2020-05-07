import os

import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from tensorflow.keras.models import Model
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BATCH_SIZE = 128
NUM_EPOCHS = 10
RANDOM_SEED = 42

# Loading Data
def get_data():
    (X_train, y_train), (X_test, y_test) = load_data()

    X_train = X_train/255.0
    X_test = X_test/255.0

    X_train = X_train[:, :, :, tf.newaxis]
    X_test = X_test[:, :, :, tf.newaxis]
    
    num_batches = len(X_train)//BATCH_SIZE
    
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(RANDOM_SEED).batch(BATCH_SIZE)
    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
    return train_data, test_data, num_batches


# Model Definition
class MyModel(Model):
    def __init__(self, loss_func, optimizer, train_loss,
                 train_metric, test_loss, test_metric):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.train_metric = train_metric
        self.test_loss = test_loss
        self.test_metric = test_metric

    # forward pass of the model
    def forward_pass(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    @tf.function    # speeds up computation
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            preds = self.forward_pass(images)
            loss = self.loss_func(labels, preds)
        # trainable variables(created by tf.Variable(trainable=True)) are self recorded by GradientTape
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_metric(labels, preds)

    @tf.function
    def test_step(self, images, labels):
        preds = self.forward_pass(images)
        loss = self.loss_func(labels, preds)

        self.test_loss(loss)
        self.test_metric(labels, preds)

    def fit(self, train_data, test_data, epochs, num_batches):
        for epoch in range(epochs):
            with tqdm(total=num_batches) as pbar:
                for images, labels in train_data:
                    self.train_step(images, labels)
                    pbar.update(1)

            for images, labels in test_data:
                self.test_step(images, labels)

            print(f"Epoch {epoch}, Train Loss {self.train_loss.result():.4f}, Train Metric {self.train_metric.result()*100:.4f}, Test Loss {self.test_loss.result():.4f}, Test Metric {self.test_metric.result()*100:.4f}")

        self.train_loss.reset_states()
        self.test_loss.reset_states()
        self.train_metric.reset_states()
        self.test_metric.reset_states()


if __name__ == "__main__":
    loss_func = SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = Mean(name="train_loss")
    train_metric = SparseCategoricalAccuracy(name="train_accuracy")

    test_loss = Mean(name="test_loss")
    test_metric = SparseCategoricalAccuracy(name="test_accuracy")

    model = MyModel(loss_func=loss_func, optimizer=optimizer,
                    train_loss=train_loss, train_metric=train_metric,
                    test_loss=test_loss, test_metric=test_metric)

    train_data, test_data, num_batches = get_data()
    model.fit(train_data=train_data, test_data=test_data,
              epochs=NUM_EPOCHS, num_batches=num_batches)
