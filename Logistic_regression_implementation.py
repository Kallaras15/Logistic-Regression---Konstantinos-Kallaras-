# =============================================================================
#                       AI & ML in archaeology
#                            2024 - 2025 
# =============================================================================
# Student Name: Konstantinos Kallaras (s4372603)
# Email: k.c.kallaras@umail.leidenuniv.nl
# =============================================================================
#
#
# BASED on these sites: 
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# https://www.geeksforgeeks.org/ml-logistic-regression-using-python/
# https://www.w3schools.com/python/python_ml_logistic_regression.asp
# 
#
# Dataset was synthesized by: ChatGPT
#
# =============================================================================
# =============================================================================
#                         LOGISTIC REGRESSION 
# =============================================================================
# =============================================================================

# Packages we will use
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv('ceramic_size_dataset.csv')

# Our variables
features= df['Size']
labels = df['Is_Ceramic']
                     
# We turn our variables into arrays for the Logistic function. 
#x = size in cm, y = fragment of ceramic / not a fragment of ceramic.
x = np.array(features).reshape(-1, 1)
y = np.array(labels)

# We use the logistic regression function to train the model with the training data.
model = LogisticRegression() # The optional parameters were not altered in any way.
model.fit(x, y) # We fit the logistic model into our data.

# We predict the class of the findings (ceramic or not ceramic) when the size is 3.80cm,
#based on the data we have.
wild_prediction = 3.8463
predicted = model.predict(np.array(wild_prediction).reshape(-1,1))

print(f"Findings of {wild_prediction:.2f}cm belong to class {predicted}")

# =============================================================================
# VISUALIZATION
# =============================================================================

# Create the test data for plotting the curveS
size_test = np.linspace(min(x), max(x), 100).reshape(-1, 1) # The range of size
probabilities = model.predict_proba(size_test)[:, 1] # Probability of being ceramic (class 1)

# Plot data points
plt.figure(figsize=(8, 5))
sns.scatterplot(x=features, y= labels, color='orange', marker='o', label="Observations")

# Plot logistic regression curve
plt.plot(size_test, probabilities, color='blue', linewidth=2, label="Sigmoid Curve")

# Formatting the plot
plt.xlabel("Size")
plt.ylabel("Probability of being a ceramic")
plt.title("Ceramic probability model")
plt.legend()
plt.grid()
plt.show()