# Importing python modules
import time

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import KHO
import tensorflow as tf


batch_size = 32
img_height = 128
img_width = 128

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"D:\Potato_disease\data",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"D:\Potato_disease\data",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 3

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(img_height, img_width, 3)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),

    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Model compilation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping to avoid overfitting
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                      min_delta=0.0001,
                                                      patience=5)
# training the model
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=20,
                    callbacks=[earlystop_callback])

# Save the model
model.save("model_v1.h5")

train_loss = history.history['loss']
train_acc = history.history['accuracy']
valid_loss = history.history['val_loss']
valid_acc = history.history['val_accuracy']

# Accuracy plots
plt.figure(figsize=(8, 4))
plt.plot(train_acc, color='green', linestyle='-', label='train accuracy')
plt.plot(valid_acc, color='blue', linestyle='-', label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# loss plots
plt.figure(figsize=(8, 4))
plt.plot(train_loss, color='orange', linestyle='-', label='train loss')
plt.plot(valid_loss, color='red', linestyle='-', label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
import graph_cut
import comp
start = time.time()
y_pred = []  # store predicted labels

y_true = []  # store true labels


# iterate over the dataset
for image_batch, label_batch in val_ds:
    # append true labels
    y_true.append(label_batch)
    # compute predictions
    preds = model.predict(image_batch)
    # append predicted labels
    y_pred.append(np.argmax(preds, axis=- 1))

# convert the true and predicted labels into tensors
correct_labels = tf.concat([item for item in y_true], axis=0)

predicted_labels = tf.concat([item for item in y_pred], axis=0)

for i in predicted_labels:
    if i == 0:
        print("Early Blight")
    elif i == 1:
        print("Healthy")
    else:
        print("Late Blight")

end = time.time()
# Confusion matrix
cm = confusion_matrix(correct_labels, predicted_labels, normalize='true')
import seaborn as sns

ax = sns.heatmap(cm, annot=True, cmap='viridis')

ax.set_title('Confusion matrix\n\n');
ax.set_xlabel('\nPredicted Class', fontsize=21, color='b')
ax.set_ylabel('True Class', fontsize=21, color='b');

ax.xaxis.set_ticklabels(['Early_blight', 'Healthy', 'Late_blight'])
ax.yaxis.set_ticklabels(['Early_blight', 'Healthy', 'Late_blight'])

# Display the visualization of the Confusion Matrix.
plt.show()

print(classification_report(correct_labels, predicted_labels))
acc = accuracy_score(correct_labels, predicted_labels)
print("Accuracy : ", accuracy_score(correct_labels, predicted_labels))
print("Precision : ", precision_score(correct_labels, predicted_labels, average='weighted'))
print("Recall : ", recall_score(correct_labels, predicted_labels, average='weighted'))
print("F1 Score : ", f1_score(correct_labels, predicted_labels, average='weighted'))
print("Error rate : ", 1 - acc)
print("Execution time : ", end - start)

# plt.rcdefaults()

objects = ('Accuracy', 'Precision', 'Recall', 'F1 Score', 'Error rate')
y_pos = np.arange(len(objects))
h = accuracy_score(correct_labels, predicted_labels)
i = precision_score(correct_labels, predicted_labels, average='weighted')
j = recall_score(correct_labels, predicted_labels, average='weighted')
k = f1_score(correct_labels, predicted_labels, average='weighted')
l = 1 - h

performance = [h, i, j, k, l]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)

plt.title('Performance Analysis')

plt.show()


