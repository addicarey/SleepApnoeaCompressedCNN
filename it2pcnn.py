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
import tensorflow_model_optimization as tfmot

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

# reshape for CNN input (samples, timesteps, channels) AC
X_train_cnn = X_train[..., np.newaxis]
X_val_cnn   = X_val[..., np.newaxis]
X_test_cnn  = X_test[..., np.newaxis]

print("CNN input shape:", X_train_cnn.shape)

# class weights to handle imbalance AC
classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, class_weights))
print("Class weights:", class_weights)

from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D

# pruning parameters AC
pruning_params = {
    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,      # 50% of weights removed
        begin_step=0,
        end_step=np.ceil(len(X_train_cnn) / 32).astype(np.int32) * 20
    )
}
# CNN model creation AC
cnn_model = Sequential([
    Conv1D(
        filters=16,
        kernel_size=7,
        activation="relu",
        padding="same",
        input_shape=(X_train_cnn.shape[1], 1)
    ),
    MaxPooling1D(pool_size=2),

    Conv1D(
        filters=32,
        kernel_size=5,
        activation="relu",
        padding="same"
    ),
    MaxPooling1D(pool_size=2),

    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    GlobalAveragePooling1D(),

    Dense(32, activation="relu"),
    Dropout(0.3),

    Dense(1, activation="sigmoid")
])

# wrap model with pruning AC
pruned_cnn = tfmot.sparsity.keras.prune_low_magnitude(
    cnn_model,
    **pruning_params
)

# compile model after wrapping AC
pruned_cnn.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


pruned_cnn.summary()

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir="./pruning_logs")
]


pruned_history = pruned_cnn.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=20,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks
)

final_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_cnn)

# recompile 
final_pruned_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)







loss, acc = final_pruned_model.evaluate(X_test_cnn, y_test)
print("Pruned CNN Test Accuracy:", acc)

final_pruned_model.save("cnn_sleep_apnea_pruned.keras")
print("Pruned CNN model saved.")


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# generate predictions AC
y_pred = (final_pruned_model.predict(X_test_cnn) > 0.5).astype(int)
y_prob = final_pruned_model.predict(X_test_cnn)

# create single figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ---- Plot 1: Training & Validation Accuracy ----
axes[0].plot(pruned_history.history["accuracy"], label="Train Accuracy")
axes[0].plot(pruned_history.history["val_accuracy"], label="Val Accuracy")

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("CNN Training Accuracy")
axes[0].legend()
axes[0].grid(True)

# ---- Plot 2: Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Sleep Aponea"]
)
disp.plot(ax=axes[1], colorbar=False)
axes[1].set_title("CNN Confusion Matrix")

# ---- Plot 3: Prediction Confidence Histogram ----
axes[2].hist(
    y_prob[y_test == 0],
    bins=30,
    alpha=0.6,
    label="Normal"
)
axes[2].hist(
    y_prob[y_test == 1],
    bins=30,
    alpha=0.6,
    label="Sleep Apnoea"
)
axes[2].set_xlabel("Predicted Probability")
axes[2].set_ylabel("Frequency")
axes[2].set_title("Prediction Confidence Distribution")
axes[2].legend()

plt.tight_layout()
plt.show()



