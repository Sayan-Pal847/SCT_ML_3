import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

data_dir = "E:/cat vs dog/train"
categories = ["cat", "dog"]
data = []

for img in os.listdir(data_dir):
    img_path = os.path.join(data_dir, img)
    pet_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    pet_img = cv2.resize(pet_img, (50, 50)).flatten()
    label = 0 if img.startswith("cat") else 1
    data.append([pet_img, label])

with open("final_data.pickle", "wb") as pick_out:
    pickle.dump(data, pick_out)

print("Data prepared and saved to final_data.pickle")

with open("final_data.pickle", "rb") as pick_in:
    data = pickle.load(pick_in)

X = np.array([item[0] for item in data])
y = np.array([item[1] for item in data])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

y_val_pred = svm_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred) * 100)
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred, target_names=["cat", "dog"]))
print("Validation Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

test_dir = "E:/cat vs dog/test1"
test_images = []
test_filenames = []

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    pet_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    pet_img = cv2.resize(pet_img, (50, 50)).flatten()
    test_images.append(pet_img)
    test_filenames.append(img_name)

X_test = scaler.transform(np.array(test_images))
y_test_pred = svm_model.predict(X_test)

predictions = ["cat" if label == 0 else "dog" for label in y_test_pred]
for img_name, pred in zip(test_filenames, predictions):
    print(f"Image: {img_name} - Prediction: {pred}")
