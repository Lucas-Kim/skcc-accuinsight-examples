#This attribute contains internal utility functions.
#It is not exposed to package users.

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def plot_learning_curves(loss, val_loss, epochs):
    plt.rcParams["figure.figsize"] = (6, 4)
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 0.5, loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, epochs, 0, 1])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)


def plot_learning_curves_train(loss, epochs):
    plt.rcParams["figure.figsize"] = (7, 6)
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    # plt.plot(np.arange(len(val_loss)) + 0.5, loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, epochs, 0, 1])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)



def GetResult_inverseTransfrom(x_target, y_target, scaler, model_created, target_size):
    target_inverse_prepare = y_target.reshape(-1, 1)
    target_inversed = scaler.inverse_transform(target_inverse_prepare).reshape(-1, target_size, 1)
    target_inversed = target_inversed.sum(axis=1)

    pred = model_created.predict(x_target)
    pred_inverse_prepare = pred.reshape(-1, 1)
    pred_inversed = scaler.inverse_transform(pred_inverse_prepare).reshape(-1, target_size, 1)
    pred_inversed = pred_inversed.sum(axis=1)

    print(mean_absolute_error(target_inversed.flatten(), pred_inversed.flatten()))

    target_graph = target_inversed.flatten()
    pred_graph = pred_inversed.flatten()

    y_min = min(min(target_graph), min(pred_graph))-100
    y_max = max(max(target_graph), max(pred_graph))+100

    # y_min = min(min(target_inversed.flatten()), min(pred_inversed.flatten()))-100
    # y_max = max(max(target_inversed.flatten()), max(pred_inversed.flatten()))+100

    plt.rcParams["figure.figsize"] = (16, 5)
    # plt.plot(target_inversed.flatten(), 'r.-', label='predict')
    # plt.axis([1, len(target_inversed.flatten()), y_min, y_max])
    plt.plot(pred_graph, 'r.-', label='predict')
    plt.axis([1, len(pred_graph), y_min, y_max])
    plt.legend(fontsize=14)

    # plt.plot(pred_inversed.flatten(), 'b.-', label='original')
    # plt.axis([1, len(pred_inversed.flatten()), y_min, y_max])
    plt.plot(target_graph, 'b.-', label='original')
    plt.axis([1, len(target_graph), y_min, y_max])
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()

    return mean_absolute_error(target_inversed.flatten(), pred_inversed.flatten())
    # print(mean_absolute_error(target_inversed.sum(axis=1), pred_inversed.sum(axis=1)))


my_prediction = {}

colors = ['r', 'c', 'm', 'y', 'k', 'khaki', 'teal', 'orchid', 'sandybrown',
          'greenyellow', 'dodgerblue', 'deepskyblue', 'rosybrown', 'firebrick',
          'deeppink', 'crimson', 'salmon', 'darkred', 'olivedrab', 'olive',
          'forestgreen', 'royalblue', 'indigo', 'navy', 'mediumpurple', 'chocolate',
          'gold', 'darkorange', 'seagreen', 'turquoise', 'steelblue', 'slategray',
          'peru', 'midnightblue', 'slateblue', 'dimgray', 'cadetblue', 'tomato'
          ]

def Graph_Evaluation(mae, name='no_name'):
    global my_prediction

    my_prediction[name] = mae
    y_value = sorted(my_prediction.items(), key=lambda x: x[1], reverse=True)

    df = pd.DataFrame(y_value, columns=['algorithm', 'mae'])
    min_ = df['mae'].min() - 10
    max_ = df['mae'].max() + 10

    length = len(df)

    plt.figure(figsize=(10, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['algorithm'], fontsize=15)
    bars = ax.barh(np.arange(len(df)), df['mae'])

    for i, v in enumerate(df['mae']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v + 2, i, str(round(v, 3)), color='k', fontsize=15, fontweight='bold')

    plt.title('Mean Absolute Error', fontsize=18)
    plt.xlim(min_, max_)

    plt.show()


def GetResult_inverseTransfrom_darnn(pred, y_target, scaler):
    labels_inverse = scaler.inverse_transform(y_target.reshape(-1, 1)).flatten()
    preds_inverse = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    preds_inverse = [0 if i < 0 else i for i in preds_inverse]

    mae_darnn = mean_absolute_error(labels_inverse, preds_inverse)
    print(mae_darnn)

    y_min = min(min(labels_inverse), min(preds_inverse)) - 100
    y_max = max(max(labels_inverse), max(preds_inverse)) + 100

    plt.rcParams["figure.figsize"] = (16, 5)

    plt.plot(labels_inverse, 'b.-', label='original')
    plt.axis([1, len(labels_inverse), y_min, y_max])
    plt.legend(fontsize=14)
    plt.grid(True)

    plt.plot(preds_inverse, 'r.-', label='predict')
    plt.axis([1, len(preds_inverse), y_min, y_max])
    plt.legend(fontsize=14)

    plt.show()