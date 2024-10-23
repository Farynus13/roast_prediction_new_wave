import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf

input_sequence_length = 120
output_sequence_length = 120
n_features = 3

path = 'data.npy'
data = np.load(path, allow_pickle=True)
# data = data[:5]


data_concat = []
(a,b,c) = data.shape
for i in range(a):
    for j in range(b):
        data_concat.append(data[i][j])
data_concat = pd.DataFrame(data_concat,columns=['bt','et','burner'])
data_concat.dropna(inplace=True)
data_concat.head()
data_concat.describe()



def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

X = data_concat[['bt', 'et', 'burner']]
y = data_concat[['bt', 'et']]

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, y_train = create_dataset(pd.DataFrame(X_train), pd.DataFrame(y_train), input_sequence_length)
X_test, y_test = create_dataset(pd.DataFrame(X_test), pd.DataFrame(y_test), output_sequence_length)

print(X_train.shape, y_train.shape)

model = Sequential()
model.add(LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=2))
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
model.summary()


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,mode='min')
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    shuffle=False,
    callbacks=[early_stopping]
)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

test_roast = data[0]
test_roast = pd.DataFrame(test_roast,columns=['bt','et','burner'])
test_roast.dropna(inplace=True)
x_test_roast = test_roast[['bt', 'et', 'burner']]
x_test_roast = x_scaler.transform(x_test_roast)
y_test_roast = test_roast[['bt', 'et']]
y_test_roast = y_scaler.transform(y_test_roast)
X_test_roast, y_test_roast = create_dataset(pd.DataFrame(x_test_roast), pd.DataFrame(y_test_roast), output_sequence_length)
offset = 200
y_pred = model.predict(X_test_roast[offset])
y_pred_inv = y_scaler.inverse_transform(y_pred)


forecast_index = np.arange(len(test_roast)-offset,len(test_roast)+len(y_pred_inv)-offset,1)
forecast_df = pd.DataFrame(data=y_pred_inv,index=forecast_index)
ax = test_roast.plot(figsize=(16,8))
forecast_df.plot(ax=ax)
plt.show()
