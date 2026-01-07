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
import os
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
import time
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc
)


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

# reshape for CNN input: (samples, timesteps, channels)
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



# pruning parameters AC
pruning_params = {
    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,      # half of weights removed AC
        begin_step=0,
        end_step=np.ceil(len(X_train_cnn) / 32).astype(np.int32) * 20
    )
}

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

# compile after wrapping AC
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

# recompile AC
final_pruned_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

loss, acc = final_pruned_model.evaluate(X_test_cnn, y_test)
print("Pruned CNN Test Accuracy:", acc)

final_pruned_model.save("cnn_sleep_apnea_pruned.keras")
print("Pruned CNN model saved.")

# predictions from pruned CNN AC
y_prob_pruned = final_pruned_model.predict(X_test_cnn).ravel()
y_pred_pruned = (y_prob_pruned > 0.5).astype(int)

# precision / Recall / F1 AC
print("Classification Report (Pruned CNN):")
print(classification_report(
    y_test,
    y_pred_pruned,
    target_names=["Normal", "Sleep Apnea"]
))

# convert pruned model to TFLite with dynamic range quantisation AC
converter = tf.lite.TFLiteConverter.from_keras_model(final_pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_quant_model = converter.convert()

# save quantised model AC
with open("cnn_sleep_apnea_pruned_quant.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("Quantised TFLite model saved.")



def get_file_size_kb(path):
    return os.path.getsize(path) / 1024

print("\nModel size comparison:")
print(f"Pruned Keras model: {get_file_size_kb('cnn_sleep_apnea_pruned.keras'):.2f} KB")
print(f"Quantised TFLite model: {get_file_size_kb('cnn_sleep_apnea_pruned_quant.tflite'):.2f} KB")

# evaluate quantised TFLite model AC
interpreter = tf.lite.Interpreter(model_path="cnn_sleep_apnea_pruned_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_predict(interpreter, X):
    predictions = []
    for i in range(len(X)):
        x = X[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])
        predictions.append(pred[0][0])
    return np.array(predictions)

# run inference AC
y_prob_tflite = tflite_predict(interpreter, X_test_cnn)
y_pred_tflite = (y_prob_tflite > 0.5).astype(int)

# accuracy AC
tflite_acc = np.mean(y_pred_tflite == y_test)
print("Quantised TFLite Test Accuracy:", tflite_acc)

#knowledge distillation teacher-student model AC
teacher_model = final_pruned_model
teacher_model.trainable = False

def build_student_model(input_shape):
    model = Sequential([
        Conv1D(8, kernel_size=7, activation="relu", padding="same",
               input_shape=input_shape),
        MaxPooling1D(pool_size=2),

        Conv1D(16, kernel_size=5, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),

        GlobalAveragePooling1D(),

        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    return model


student_model = build_student_model(
    input_shape=(X_train_cnn.shape[1], 1)
)

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, temperature=5.0, alpha=0.5):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.student_loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.distillation_loss_fn = tf.keras.losses.KLDivergence()
        self.metric = tf.keras.metrics.BinaryAccuracy()

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        x, y = data

        teacher_preds = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_preds = self.student(x, training=True)

            student_loss = self.student_loss_fn(y, student_preds)

            distill_loss = self.distillation_loss_fn(
                tf.nn.sigmoid(teacher_preds / self.temperature),
                tf.nn.sigmoid(student_preds / self.temperature)
            )

            total_loss = (
                self.alpha * student_loss +
                (1 - self.alpha) * distill_loss
            )

        grads = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.student.trainable_variables)
        )

        self.metric.update_state(y, student_preds)
        return {"loss": total_loss, "accuracy": self.metric.result()}

    def test_step(self, data):
        x, y = data
        preds = self.student(x, training=False)
        loss = self.student_loss_fn(y, preds)
        self.metric.update_state(y, preds)
        return {"loss": loss, "accuracy": self.metric.result()}

distiller = Distiller(
    student=student_model,
    teacher=teacher_model,
    temperature=5.0,   # softer probability distributions AC
    alpha=0.5          # balance hard vs soft targets AC
)

distiller.compile(
    optimizer=Adam(learning_rate=0.001)
)

distiller.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=20,
    batch_size=32
)

student_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

student_loss, student_acc = student_model.evaluate(
    X_test_cnn, y_test
)

print("Distilled Student Test Accuracy:", student_acc)

#quantisation of distilled mdodel AC
converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_student = converter.convert()

with open("student_distilled_quantised.tflite", "wb") as f:
    f.write(tflite_student)

print("Final distilled & quantised student model saved.")





# load quantised distilled model AC
interpreter = tf.lite.Interpreter(
    model_path="student_distilled_quantised.tflite"
)
interpreter.allocate_tensors()

# get input/output details AC
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

WINDOW_SIZE = 2500   # one ECG segment (as used in training)
STRIDE = 250         # overlap (90%)


def run_streaming_inference(ecg_signal, window_size=2500, stride=250):
    predictions = []
    inference_times = []

    for start in range(0, len(ecg_signal) - window_size + 1, stride):
        window = ecg_signal[start:start + window_size]
        window = window.reshape(1, window_size, 1).astype(np.float32)

        start_time = time.perf_counter()

        interpreter.set_tensor(input_details[0]['index'], window)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0][0]

        end_time = time.perf_counter()

        predictions.append(output)
        inference_times.append(end_time - start_time)

    return np.array(predictions), np.array(inference_times)


# take one ECG sample from test set
sample_ecg = X_test[0] 

preds, times = run_streaming_inference(sample_ecg)

print("Mean inference time (µs):", times.mean() * 1e6)
print("Max inference time (µs):", times.max() * 1e6)


def apnea_decision(predictions, threshold=0.5, required_ratio=0.6):
    binary_preds = predictions > threshold
    ratio = binary_preds.mean()
    return ratio > required_ratio

apnea_detected = apnea_decision(preds)

print("Apnoea detected:", apnea_detected)


model_size_kb = os.path.getsize(
    "student_distilled_quantised.tflite"
) / 1024

print(f"Final model size: {model_size_kb:.2f} KB")

print(f"Avg latency: {times.mean()*1000:.2f} ms")
print(f"95th percentile latency: {np.percentile(times, 95)*1000:.2f} ms")

#plt 1 training progress
plt.figure(figsize=(6,4))
plt.plot(pruned_history.history["loss"], label="Training Loss")
plt.plot(pruned_history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("CNN Training Convergence")
plt.legend()
plt.tight_layout()
plt.show()

#plt 2 ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob_pruned)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Pruned CNN")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

#plt 3 COnfusion matrix
cm = confusion_matrix(y_test, y_pred_pruned)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Sleep Apnea"]
)

plt.figure(figsize=(4.5,4.5))
disp.plot(cmap="Blues", values_format="d", colorbar=False)
plt.title("Confusion Matrix – Pruned CNN")
plt.tight_layout()
plt.show()



#table of compression and performance summary AC
results_df = pd.DataFrame({
    "Model": [
        "Pruned CNN",
        "Pruned + Quantised",
        "Distilled + Quantised"
    ],
    "Accuracy": [
        acc,
        tflite_acc,
        student_acc
    ],
    "Model Size (KB)": [
        get_file_size_kb("cnn_sleep_apnea_pruned.keras"),
        get_file_size_kb("cnn_sleep_apnea_pruned_quant.tflite"),
        model_size_kb
    ],
    "Avg Latency (ms)": [
        np.nan,   
        np.nan,
        times.mean() * 1000
    ]
})

print(results_df)








