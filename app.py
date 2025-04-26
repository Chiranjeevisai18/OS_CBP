import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

# Step 1: Generate Synthetic Page Reference Data
def generate_page_references(num_pages=50, num_references=1000):
    page_references = [random.randint(0, num_pages - 1) for _ in range(num_references)]
    return page_references

# Generate the data
references = generate_page_references(num_pages=50, num_references=1000)

# Step 2: Feature Extraction
def extract_features(references):
    last_access_time = {}  # page -> last accessed index
    access_frequency = {}  # page -> total times accessed
    features = []

    for idx, page in enumerate(references):
        time_since_last = idx - last_access_time.get(page, idx)
        frequency = access_frequency.get(page, 0)
        features.append([page, time_since_last, frequency])
        last_access_time[page] = idx
        access_frequency[page] = frequency + 1

    return pd.DataFrame(features, columns=["PageNumber", "TimeSinceLastAccess", "AccessFrequency"])

# Step 3: Create Labels
def create_labels(references, window_size=5):
    labels = []
    for i in range(len(references)):
        current_page = references[i]
        future_accesses = references[i+1:i+1+window_size]
        labels.append(1 if current_page in future_accesses else 0)
    return labels

# Extract features and create labels
features_df = extract_features(references)
labels = create_labels(references)

# Step 4: Train ML Model (Decision Tree)
X = features_df[['PageNumber', 'TimeSinceLastAccess', 'AccessFrequency']]
y = pd.Series(labels)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Simulate Traditional Algorithms (FIFO and LRU)
def simulate_fifo(page_references, num_pages=50):
    memory = []
    page_faults = 0
    for page in page_references:
        if page not in memory:
            if len(memory) >= num_pages:
                memory.pop(0)
            memory.append(page)
            page_faults += 1
    return page_faults

def simulate_lru(page_references, num_pages=50):
    memory = []
    page_faults = 0
    for page in page_references:
        if page not in memory:
            if len(memory) >= num_pages:
                memory.pop(0)
            memory.append(page)
            page_faults += 1
        else:
            memory.remove(page)
            memory.append(page)
    return page_faults

# Step 6: Simulate ML Model
def simulate_ml_model(page_references, features_df, model, num_pages=50):
    memory = []
    page_faults = 0
    for i, page in enumerate(page_references):
        current_features = features_df.iloc[i][['PageNumber', 'TimeSinceLastAccess', 'AccessFrequency']].values.reshape(1, -1)
        will_be_used_soon = model.predict(current_features)[0]
        if will_be_used_soon == 1:
            if page not in memory and len(memory) < num_pages:
                memory.append(page)
            elif page not in memory:
                memory.pop(0)
                memory.append(page)
            page_faults += 1
    return page_faults

# Step 7: Compare Results
fifo_faults = simulate_fifo(references)
lru_faults = simulate_lru(references)
ml_faults = simulate_ml_model(references, features_df, model)

# Plot the results
methods = ['FIFO', 'LRU', 'ML Model']
faults = [fifo_faults, lru_faults, ml_faults]

plt.figure(figsize=(8, 6))
plt.bar(methods, faults, color=['red', 'blue', 'green'])
plt.title('Page Faults Comparison (FIFO vs LRU vs ML Model)')
plt.xlabel('Methods')
plt.ylabel('Page Faults')
plt.show()

# Step 8: Evaluate the ML Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Traditional Methods vs ML Model:")
print(f"Page Faults - FIFO: {fifo_faults}, LRU: {lru_faults}, ML Model: {ml_faults}")
print(f"Accuracy of ML Model: {accuracy * 100:.2f}%")

# Optionally, plot the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Not Used Soon', 'Used Soon'])
plt.show()
