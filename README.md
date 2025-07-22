📩 Spam SMS Detection
📌 Objective
Build a machine learning model to classify SMS messages as spam or ham (not spam) using natural language processing techniques.

🛠️ Tools & Techniques
Dataset: spam.csv (SMS messages with labels)

Model: Naive Bayes

Feature Engineering: TF-IDF Vectorization

Evaluation: Classification Report

📈 Output
The model is trained using the MultinomialNB classifier and evaluated using a classification report (precision, recall, F1-score).

🎬 Movie Genre Classification
📌 Objective
Predict the genre of a movie based on its plot summary using a text classification pipeline.

🛠️ Tools & Techniques
Dataset: train_data.txt (custom text format with plot and genre)

Model: Logistic Regression

Vectorization: TF-IDF (max_features=5000)

Evaluation: Accuracy, Confusion Matrix

Visualization: Seaborn Heatmap

File Outputs:

confusion_matrix.png (confusion matrix)

predicted_results.csv (actual vs predicted genres)

📈 Output
Displays top confused genre pairs, accuracy percentage, and saves prediction results.

👥 Customer Churn Prediction
📌 Objective
Predict whether a customer will churn (leave) based on demographic and financial features.

🛠️ Tools & Techniques
Dataset: Churn_Modelling.csv

Model: Random Forest Classifier

Feature Selection: Tenure, Balance, EstimatedSalary

Preprocessing: Standardization

Evaluation: Confusion Matrix, Classification Report

Visualization: Heatmap + Feature Importance Bar Chart

📈 Output
Includes classification performance and a chart showing the most influential features.
