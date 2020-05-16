import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []
  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

# zip_path = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
#     fname='jena_climate_2009_2016.csv.zip',
#     extract=True)
# csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv("D:/Bachelor/Python works/Fed-Poll/clustered_sensors/cluster0/14.csv",
                 names=['ozone','particullate_matter','carbon_monoxide','sulfure_dioxide'
                     ,'nitrogen_dioxide','longitude','latitude','timestamp','cluster_label'])
# df = pd.read_csv(csv_path)

past_history = 720
future_target = 72
STEP = 6
TRAIN_SPLIT = 11712
tf.random.set_seed(13)
# features_considered = ['ozone','carbon_monoxide']
# # features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
# features = df[features_considered]
# # features.index = df['Date Time']
# features.index = df['timestamp']
columns = ['ozone','particullate_matter','carbon_monoxide','sulfure_dioxide', 'nitrogen_dioxide']
for column in columns:
        uni_data = df[column]
        uni_data.index = df['timestamp']
        uni_data = uni_data.values
        uni_data = uni_data[1:]
        uni_data = uni_data.astype('float32')
        uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
        uni_train_std = uni_data[:TRAIN_SPLIT].std()
        uni_data = (uni_data-uni_train_mean)/uni_train_std
        univariate_past_history = 20
        univariate_future_target = 0

        x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                                   univariate_past_history,
                                                   univariate_future_target)
        x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                               univariate_past_history,
                                               univariate_future_target)
        def create_time_steps(length):
          return list(range(-length, 0))

        def show_plot(plot_data, delta, title):
          labels = ['History', 'True Future', 'Model Prediction']
          marker = ['.-', 'rx', 'go']
          time_steps = create_time_steps(plot_data[0].shape[0])
          if delta:
            future = delta
          else:
            future = 0

          plt.title(title)
          for i, x in enumerate(plot_data):
            if i:
              plt.plot(future, plot_data[i], marker[i], markersize=10,
                       label=labels[i])
            else:
              plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
          plt.legend()
          plt.xlim([time_steps[0], (future+5)*2])
          plt.xlabel('Time-Step')
          return plt
        show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
        dataset = features.values
        def baseline(history):
          return np.mean(history)

        show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
                   'Baseline Prediction Example')

        BATCH_SIZE = 256
        BUFFER_SIZE = 1000
        train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
        train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
        val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
        simple_lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
            tf.keras.layers.Dense(1)
        ])

        simple_lstm_model.compile(optimizer='adam', loss='mae')
        for x, y in val_univariate.take(1):
            print(simple_lstm_model.predict(x).shape)
        EVALUATION_INTERVAL = 200
        EPOCHS = 10

        simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                              steps_per_epoch=EVALUATION_INTERVAL,
                              validation_data=val_univariate, validation_steps=50)
        n = 0
        for x, y in val_univariate.take(3):
            show_plot([x[0].numpy(), y[0].numpy(),
                            simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
            save_path = "./figures for single variable single sensor/" + column + str(n)
            n += 1
            plt.savefig(save_path)
            plt.show()
        show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])]
                  , 0, 'Baseline Prediction Example')
        save_path = "./figures for single variable single sensor/" + column + str(n)
        plt.savefig(save_path)
        plt.show()








#
#
# dataset = dataset[1:]
# dataset = dataset.astype('float32')
# data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
# data_std = dataset[:TRAIN_SPLIT].std(axis=0)
# dataset = (dataset-data_mean)/data_std
# x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
#                                                    TRAIN_SPLIT, past_history,
#                                                    future_target, STEP,
#                                                    single_step=True)
# x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
#                                                TRAIN_SPLIT, None, past_history,
#                                                future_target, STEP,
#                                                single_step=True)
#
# BATCH_SIZE = 256
# BUFFER_SIZE = 10000
# EVALUATION_INTERVAL = 200
# EPOCHS = 10
#
#
# train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
# train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
#
# val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
# val_data_single = val_data_single.batch(BATCH_SIZE).repeat()
#
# single_step_model = tf.keras.models.Sequential()
# single_step_model.add(tf.keras.layers.LSTM(32,
#                                            input_shape=x_train_single.shape[-2:]))
# # single_step_model.add(tf.keras.layers.Dense(32))
# single_step_model.add(tf.keras.layers.Dense(1))
#
# single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
# for x, y in val_data_single.take(1):
#   print(single_step_model.predict(x).shape)
# single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
#                                             steps_per_epoch=EVALUATION_INTERVAL,
#                                             validation_data=val_data_single,
#                                             validation_steps=50)
# def plot_train_history(history, title):
#   loss = history.history['loss']
#   val_loss = history.history['val_loss']
#
#   epochs = range(len(loss))
#
#   plt.figure()
#
#   plt.plot(epochs, loss, 'b', label='Training loss')
#   plt.plot(epochs, val_loss, 'r', label='Validation loss')
#   plt.title(title)
#   plt.legend()
#
#   plt.show()
# plot_train_history(single_step_history,
#                    'Single Step Training and validation loss')
#
