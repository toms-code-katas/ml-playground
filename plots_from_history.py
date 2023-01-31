# This file contains a function for plotting diverse metrics from the history object of a Keras model.
import pickle
import matplotlib.pyplot as plt

def plot_metrics(history, metrics=None, last_x_epoch=None):

    history_to_plot = history.history

    if last_x_epoch:
        history_to_plot = {key: value[-last_x_epoch:] for key, value in history_to_plot.items()}

    if not metrics:
        metrics = history_to_plot.keys()

    for metric in metrics:
        plt.plot(history_to_plot[metric])
        plt.ylabel(metric)
        plt.xlabel('epoch')

    # Convert the dict_keys to a string
    metrics_as_string = str(list(history_to_plot.keys()))
    plt.title('model ' + metrics_as_string)

    plt.legend(metrics, loc='upper left')
    plt.show()


if __name__ == "__main__":
    # Load the history object from the file
    # with open("history-and-loss-history.pkl", "rb") as f:
    #     history = pickle.load(f)
    #
    # plot_metrics(history)
    # plot_metrics(history, last_x_epoch=10)
    # plot_metrics(history, last_x_epoch=60)
    # plot_metrics(history, last_x_epoch=80)

    # Load a history with mae
    with open("history-1.pkl", "rb") as f:
        history = pickle.load(f)

    plot_metrics(history)