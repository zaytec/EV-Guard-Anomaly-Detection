
# SMART ANOMALY DETECTION SYSTEM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# GENERATE SYNTHETIC DATA
np.random.seed(42)

n = 3000
time = np.arange(n)

# Normal signal
signal = np.sin(0.02 * time) + np.random.normal(0, 0.05, n)

# Inject anomalies
anomaly_indices = np.random.choice(n, 100, replace=False)
signal[anomaly_indices] += np.random.normal(3, 0.5, 100)

# Ground truth
labels = np.zeros(n)
labels[anomaly_indices] = 1

df = pd.DataFrame({
    "value": signal,
    "label": labels
})


# PREPROCESSING

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[['value']])

train = data_scaled[:2000]  # normal training portion

# AUTOENCODER
input_dim = train.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation="relu")(input_layer)
encoded = Dense(4, activation="relu")(encoded)
decoded = Dense(8, activation="relu")(encoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(train, train,
                epochs=20,
                batch_size=32,
                verbose=0)

# Reconstruction error
recon = autoencoder.predict(data_scaled)
mse = np.mean(np.power(data_scaled - recon, 2), axis=1)


# ISOLATION FOREST

iso = IsolationForest(contamination=0.03, random_state=42)
iso.fit(train)
iso_pred = iso.predict(data_scaled)

# COMBINE RESULTS

threshold = np.percentile(mse, 95)
auto_pred = mse > threshold

final_pred = np.logical_or(auto_pred, iso_pred == -1).astype(int)

df['anomaly'] = final_pred

# EVALUATION

y_true = df['label']
y_pred = df['anomaly']

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n========== RESULTS ==========")
print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")


# PLOT (LINKEDIN READY)t.figure(figsize=(15,6))

plt.plot(df['value'], label='Signal')
plt.scatter(df.index[df['anomaly']],
            df['value'][df['anomaly']],
            color='red', label='Anomaly', s=10)
plt.legend()
plt.title("Anomaly Detection in Time-Series Data")
plt.savefig("anomaly_plot.png", dpi=300)
plt.show()


#CONFUSION MATRIX

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("\nSaved images:")
print("1. anomaly_plot.png (POST THIS)")
print("2. confusion_matrix.png")
