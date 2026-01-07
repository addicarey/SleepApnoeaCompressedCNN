#imports and setup AC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# path to data and read csv AC
dp = "ecg_sleep_apnea_dataset.csv"
df = pd.read_csv(dp)

print("Dataset shape:", df.shape)
print(df.head())

# split features and labels AC
X = df.iloc[:, :-1].values
y_raw = df.iloc[:, -1].values  # string labels

# map string labels to integers AC
label_map = {
    "Normal": 0,
    "Sleep Apnea": 1
}

y = np.array([label_map[label] for label in y_raw])

print("Label distribution:", np.bincount(y))

# train val test split AC
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# data normalisation AC
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# class weights to handle imbalance AC
classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, class_weights))
print("Class weights:", class_weights)

# baseline model creation AC
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# baseline model training AC
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    class_weight=class_weights
)

# model evaluation AC
loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)

# saved model AC
model.save("baseline_sleep_apnea_model.keras")
print("Baseline model saved.")
