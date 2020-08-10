import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.utils import to_categorical
import optuna
from rich import print
import rich.traceback
import warnings
warnings.filterwarnings("ignore")
rich.traceback.install()


# CONSTANTS
BATCHSIZE = 128
LR = 3e-5
EPOCHS = 10
NUM_TRIALS = 20
NUM_STEPS = 60000//BATCHSIZE

def create_model(trial):
    # Optimize the numbers of layers & corresponding units
    n_layers = trial.suggest_int("n_layers", 1, 3)
    model = Sequential()
    model.add(Flatten())
    for i in range(n_layers):
        num_hidden = trial.suggest_int(f"n_units_l{i}", 4, 128, log=True)
        model.add(Dense(num_hidden, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    return model


def create_optimizer(trial):
    # Optimize the choice of optimizers
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    kwargs = {}
    kwargs["learning_rate"] = LR
    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer


def get_mnist():
    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_valid = x_valid.astype("float32") / 255

    y_train = y_train.astype("int32")
    y_valid = y_valid.astype("int32")

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, to_categorical(y_train)))
    train_ds = train_ds.shuffle(60000).repeat().batch(BATCHSIZE).prefetch(tf.data.experimental.AUTOTUNE)

    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, to_categorical(y_valid)))
    valid_ds = valid_ds.shuffle(10000).batch(BATCHSIZE).prefetch(tf.data.experimental.AUTOTUNE)
    return train_ds, valid_ds


def objective(trial):
    train_ds, valid_ds = get_mnist()
    model = create_model(trial)
    optimizer = create_optimizer(trial)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, steps_per_epoch=NUM_STEPS, epochs=EPOCHS)
    acc_score = model.evaluate(valid_ds)
    return acc_score[1]


if __name__ == "__main__":
    optuna.logging.set_verbosity(0)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=NUM_TRIALS, show_progress_bar=True)

    print(f"# FINISHED TRIALS: {len(study.trials)}")
    print(f"\nBEST TRIALS:\n {study.best_trial}\n\n")
    print(f"\nVALUE:\n {study.best_trial.value}\n\n")
    print(f"\nBEST PARAMS:\n {study.best_params}\n\n")
    
    # visualizations
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show(rendered="png")

