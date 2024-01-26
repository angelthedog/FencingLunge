# Compatibility layer between Python 2 and Python 3
from __future__ import print_function

from itertools import combinations
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_absolute_error ###### pip install scikit-learn
from sklearn.model_selection import train_test_split

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, GlobalAveragePooling1D, Conv1D, MaxPooling1D
from keras.callbacks import Callback

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# %%
def show_basic_dataframe_info(dataframe, preview_rows=20):
    print("Number of columns in the dataframe: %i" % (dataframe.shape[1]))
    print("Number of rows in the dataframe: %i\n" % (dataframe.shape[0]))
    print(dataframe.head(preview_rows))
    print("\nDescription of dataframe:\n")


def read_data(file_path):
    column_names = ['timestamp',
                    'left_shoulder_x',
                    'left_shoulder_y',
                    'right_shoulder_x',
                    'right_shoulder_y',
                    'left_elbow_x',
                    'left_elbow_y',
                	'right_elbow_x',
                    'right_elbow_y',
                    'left_wrist_x',
                    'left_wrist_y',
                    'right_wrist_x',
                    'right_wrist_y',
                    'left_hip_x',
                    'left_hip_y',
                    'right_hip_x',
                    'right_hip_y',
                    'left_knee_x',
                    'left_knee_y',
                    'right_knee_x',
                    'right_knee_y',
                    'left_ankle_x',
                    'left_ankle_y',
                    'right_ankle_x',
                    'right_ankle_y',
                    'user_id',
                    'score']
    df = pd.read_csv(file_path,
                     header=None,
                     names=column_names)

    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    df.dropna(axis=0, how='any', inplace=True)
    return df


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma


def plot_axis(ax, x, y, title):
    ax.plot(x, y, color='grey')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


""" def plot_activity(activity, data):
    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6,
         figsize=(15, 10),
         sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['is-on-surface'], 'is-on-surface')
    plot_axis(ax3, data['timestamp'], data['azimuth'], 'azimuth')
    plot_axis(ax4, data['timestamp'], data['altitude'], 'altitude')
    plot_axis(ax5, data['timestamp'], data['pressure'], 'pressure')
    # Set subtitle based on activity value
    if activity == 1:
        subtitle = "Neurotypical"
    elif activity == 0:
        subtitle = "Dysgraphia"
    else:
        subtitle = str(activity)  # Default to the activity number if not 0 or 1
    fig.suptitle(subtitle, fontsize=14)
    plt.subplots_adjust(top=0.90)
    plt.show()
 """


def create_segments(df, time_steps, step):
    segments = []
    labels = []
    person_ids = []
    for i in range(0, len(df) - time_steps, step):
   
        lsx = df['left_shoulder_x'].values[i: i + time_steps]
        lsy = df['left_shoulder_y'].values[i: i + time_steps]
        rsx = df['right_shoulder_x'].values[i: i + time_steps]
        rsy = df['right_shoulder_y'].values[i: i + time_steps]
    
        lex = df['left_elbow_x'].values[i: i + time_steps]
        ley = df['left_elbow_y'].values[i: i + time_steps]
        rex = df['right_elbow_x'].values[i: i + time_steps]
        rey = df['right_elbow_y'].values[i: i + time_steps]

        lwx = df['left_wrist_x'].values[i: i + time_steps]
        lwy = df['left_wrist_y'].values[i: i + time_steps]
        rwx = df['right_wrist_x'].values[i: i + time_steps]
        rwy = df['right_wrist_y'].values[i: i + time_steps]

        lhx = df['left_hip_x'].values[i: i + time_steps]
        lhy = df['left_hip_y'].values[i: i + time_steps]
        rhx = df['right_hip_x'].values[i: i + time_steps]
        rhy = df['right_hip_y'].values[i: i + time_steps]

        lkx = df['left_knee_x'].values[i: i + time_steps]
        lky = df['left_knee_y'].values[i: i + time_steps]
        rkx = df['right_knee_x'].values[i: i + time_steps]
        rky = df['right_knee_y'].values[i: i + time_steps]

        lax = df['left_ankle_x'].values[i: i + time_steps]
        lay = df['left_ankle_y'].values[i: i + time_steps]
        rax = df['right_ankle_x'].values[i: i + time_steps]
        ray = df['right_ankle_y'].values[i: i + time_steps]


        score = stats.mode(df['score'][i: i + time_steps])[0]
        id = stats.mode(df['user_id'][i: i + time_steps])[0]
        segments.append([lsx, lsy, rsx, rsy,
                         lex, ley, rex, rey, 
                         lwx, lwy, rwx, rwy, 
                         lhx, lhy, rhx, rhy, 
                         lkx, lky, rkx, rky, 
                         lax, lay, rax, ray])
        labels.append(score)
        person_ids.append(id)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_FEATURES, time_steps ).transpose((0,2,1))
    reshaped_segments = reshaped_segments.reshape(-1, INPUT_SHAPE)
    labels = np.asarray(labels)
    person_ids = np.asarray(person_ids)
    # Convert type for Keras otherwise Keras cannot process the data
    reshaped_segments = reshaped_segments.astype("float32")
    labels = labels.astype("float32")

    return reshaped_segments, labels, person_ids


class PersonLevelEvaluation(Callback):
    def __init__(self, validation_data, person_ids, threshold):
        super().__init__()
        self.validation_data = validation_data
        self.person_ids = person_ids
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        segment_predictions = self.model.predict(x_val)
        mae = mean_absolute_error(y_val, segment_predictions)
        print(f"Epoch {epoch+1}: Person-level MAE is {mae:.4f}")

    @staticmethod
    def aggregate_predictions(predictions, labels, person_ids, threshold):
        person_predictions = {}
        person_true_labels = {}

        for pred, label, person_id in zip(predictions, labels, person_ids):
            if person_id not in person_predictions:
                person_predictions[person_id] = []
                person_true_labels[person_id] = label  # Assuming all segments of a person have the same label

            # Convert the prediction to a binary value (0 or 1)
            binary_pred = int(np.round(pred[0]))  # Convert numpy float to Python int
            person_predictions[person_id].append(binary_pred)

        aggregated_predictions = []
        aggregated_true_labels = []

        for person_id, binary_preds in person_predictions.items():
            # Calculate the percentage of '1' predictions
            percent_ones = sum(binary_preds) / len(binary_preds)
            # If the percentage of '1's is above the threshold, classify as '1', else '0'
            final_prediction = 1 if percent_ones >= threshold else 0
            aggregated_predictions.append(final_prediction)
            aggregated_true_labels.append(person_true_labels[person_id])

        return np.array(aggregated_predictions), np.array(aggregated_true_labels)


def train_and_evaluate(df_train, df_test, threshold):
    # Reshape the training data into segments
    # so that they can be processed by the network
    x_data, y_data, person_ids = create_segments(df_train, TIME_PERIODS, STEP_DISTANCE)
    print('x_data shape:', x_data.shape)
    print('y_data shape: ', y_data.shape)

    print("\n--- Create neural network model ---\n")
    model_m = Sequential()
    model_m.add(Reshape((TIME_PERIODS, N_FEATURES), input_shape=(INPUT_SHAPE,)))
    model_m.add(Conv1D(100, 10, activation='relu', padding='same', input_shape=(TIME_PERIODS, N_FEATURES)))
    model_m.add(Conv1D(100, 10, activation='relu', padding='same'))
    model_m.add(MaxPooling1D(3))
    model_m.add(Conv1D(160, 10, activation='relu', padding='same'))
    model_m.add(Conv1D(160, 10, activation='relu', padding='same'))
    model_m.add(GlobalAveragePooling1D())
    model_m.add(Dropout(0.5))
    model_m.add(Dense(1, activation='linear')) 
    # print(model_m.summary())

    print("\n--- Fit the model ---\n")
    # Split data while keeping person segments together
    x_train, x_val, y_train, y_val, person_ids_train, person_ids_val = train_test_split(
        x_data, y_data, person_ids, test_size=0.2, stratify=person_ids
    )

    callbacks_list = [
        # keras.callbacks.ModelCheckpoint(
        #     filepath='best_model.{epoch:02d}-{val_mae:.4f}.tf',
        #     monitor='val_mae',
        #     save_best_only=True,
        #     mode='min'
        # ),
        keras.callbacks.EarlyStopping(monitor='mae', patience=4)
    ]

    # Create the PersonLevelEvaluation callback with validation data
    # person_level_callback = PersonLevelEvaluation(validation_data=(x_val, y_val), 
    #                                               person_ids=person_ids_val, 
    #                                               threshold=threshold)
    # Add the callback to your list of callbacks
    # callbacks_list.append(person_level_callback)

    model_m.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    # Train your model without specifying validation_data in model.fit()
    model_m.fit(x_train, 
                y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(x_val, y_val), 
                callbacks=callbacks_list, 
                verbose=1)

    print("\n--- Evaluate Model on Training Data ---\n")
    train_loss, train_mae = model_m.evaluate(x_train, y_train, verbose=1)
    print(f"Segment-level on training data - Loss: {train_loss}, MAE: {train_mae}")

    print("\n--- Check against test data ---\n")
    x_test, y_test, test_person_ids = create_segments(df_test, TIME_PERIODS, STEP_DISTANCE)
    print('x_test shape:', x_test.shape)
    print('y_test shape: ', y_test.shape)

    segment_predictions = model_m.predict(x_test)

    # Calculate regression metrics (e.g., MAE) if y_test is available
    mae = mean_absolute_error(y_test, segment_predictions)
    print(f"Segment-level MAE on test data: {mae:.4f}")
    for person_id, prediction, true_value in zip(test_person_ids, segment_predictions, y_test):
        print(f"Person ID: {person_id}, Prediction: {prediction[0]:.1f}, True Value: {true_value}")

    return segment_predictions


def split_and_evaluate(df):
    # Splitting based on user-id and activity
    unique_users_0 = df[df['activity'] == 0]['user-id'].unique()
    unique_users_1 = df[df['activity'] == 1]['user-id'].unique()

    # Randomly split user-ids into 5 sets while maintaining the activity ratio
    sets_0 = np.array_split(np.random.permutation(unique_users_0), 5)
    sets_1 = np.array_split(np.random.permutation(unique_users_1), 5)
    print("sets_0:", sets_0)
    print("sets_1:", sets_1)

    best_accuracy = 0
    best_split = None

    for training_indices in combinations(range(5), 4):
        testing_indices = [x for x in range(5) if x not in training_indices]

        # Combine user-ids for training and testing
        train_user_ids = np.concatenate([sets_0[i] for i in training_indices] + [sets_1[i] for i in training_indices])
        test_user_ids = np.concatenate([sets_0[i] for i in testing_indices] + [sets_1[i] for i in testing_indices])

        # Create training and testing dataframes
        df_train = df[df['user-id'].isin(train_user_ids)]
        df_test = df[df['user-id'].isin(test_user_ids)]

        # Train the model and evaluate
        person_level_accuracy = train_and_evaluate(df_train, df_test)

        # Store the best result
        if person_level_accuracy > best_accuracy:
            best_accuracy = person_level_accuracy
            best_split = (training_indices, testing_indices)

    print(f"Best person-level accuracy: {best_accuracy}")
    print(f"Best split: Training sets {best_split[0]}, Testing sets {best_split[1]}")



# %%
# ------- THE PROGRAM TO LOAD DATA AND TRAIN THE MODEL -------
# Set some standard parameters upfront
pd.options.display.float_format = '{:.4f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
print('keras version ', keras.__version__)

# The number of steps within one time segment
TIME_PERIODS = 4
# The steps to take from one segment to the next; No overlap between the segments if same as TIME_PERIODS
STEP_DISTANCE = 2
# Hyper-parameters
BATCH_SIZE = 16
EPOCHS = 50
N_FEATURES = 24
INPUT_SHAPE = N_FEATURES * TIME_PERIODS

# %%
print("\n--- Load, inspect and transform data ---\n")
# Load data set containing all the data from csv
df = read_data('data.csv')
# Describe the data
# print("First 20 rows of the df dataframe:\n")
# show_basic_dataframe_info(df, 20)

""" ################ Plotting ###########################
ax = df['activity'].value_counts().plot(kind='bar', title='Data Samples by Participant Diagnosis', color='grey')
ax.set_xlabel('')
ax.set_xticks(ticks=[0, 1])
ax.set_xticklabels(labels=["Neurotypical", "Dysgraphia"], fontsize=12, rotation=0)
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.show()

ax = df['user-id'].value_counts().plot(kind='bar', title='Data Samples Per Participant', color='grey')
ax.set_xticklabels([])
ax.set_xlabel('Participants')
ax.set_ylabel('Samples')
plt.show()

for activity in np.unique(df["activity"]):
    subset = df[df["activity"] == activity][:1800]
    plot_activity(activity, subset)
################# End Plotting ####################### """

# %%
print("\n--- Reshape the data into segments ---\n")
# Normalize features for training data set
# df['left_shoulder_x']  = feature_normalize(df['left_shoulder_x'])
# df['left_shoulder_y']  = feature_normalize(df['left_shoulder_y'])
# df['right_shoulder_x'] = feature_normalize(df['right_shoulder_x'])
# df['right_shoulder_y'] = feature_normalize(df['right_shoulder_y'])
# df['left_elbow_x']     = feature_normalize(df['left_elbow_x'])
# df['left_elbow_y']     = feature_normalize(df['left_elbow_y'])
# df['right_elbow_x']    = feature_normalize(df['right_elbow_x'])
# df['right_elbow_y']    = feature_normalize(df['right_elbow_y'])
# df['left_wrist_x']     = feature_normalize(df['left_wrist_x'])
# df['left_wrist_y']     = feature_normalize(df['left_wrist_y'])
# df['right_wrist_x']    = feature_normalize(df['right_wrist_x'])
# df['right_wrist_y']    = feature_normalize(df['right_wrist_y'])
# df['left_hip_x']       = feature_normalize(df['left_hip_x'])
# df['left_hip_y']       = feature_normalize(df['left_hip_y'])
# df['right_hip_x']      = feature_normalize(df['right_hip_x'])
# df['right_hip_y']      = feature_normalize(df['right_hip_y'])
# df['left_knee_x']      = feature_normalize(df['left_knee_x'])
# df['left_knee_y']      = feature_normalize(df['left_knee_y'])
# df['right_knee_x']     = feature_normalize(df['right_knee_x'])
# df['right_knee_y']     = feature_normalize(df['right_knee_y'])
# df['left_ankle_x']     = feature_normalize(df['left_ankle_x'])
# df['left_ankle_y']     = feature_normalize(df['left_ankle_y'])
# df['right_ankle_x']    = feature_normalize(df['right_ankle_x'])
# df['right_ankle_y']    = feature_normalize(df['right_ankle_y'])

# split_and_evaluate(df)

train_user_ids = list(range(1, 27))
test_user_ids = list(range(27, 34))
# Create training and testing dataframes
df_train = df[df['user_id'].isin(train_user_ids)]
df_test = df[df['user_id'].isin(test_user_ids)]

# Train the model and evaluate
person_level_accuracy = train_and_evaluate(df_train, df_test, 0.5)

# for BATCH_SIZE in [64, 128, 400]:
# for i in range(50):
#     person_level_accuracy = train_and_evaluate(df_train, df_test, 0.5)
#     print(f"epoch 50 Run {i} - Person-level accuracy: {person_level_accuracy}")