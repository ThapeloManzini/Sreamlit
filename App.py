from sklearn.calibration import LabelEncoder
import streamlit as st

# Title of the application
st.title("Autism Prediction System")

# Sidebar for navigation
st.sidebar.header("Navigation")
nav_options = st.sidebar.selectbox("Choose a section:", ["Home", "Resources", "Predictive System", "Tools"])

# Home Section
if nav_options == "Home":
    st.header("Welcome to Autism Prediction System")
    st.write("""This system leverages various factors, including gender, to help predict the likelihood of Autism Spectrum Disorder (ASD). Research has shown that autism occurs more frequently in males than females. However, it's important to note that autism can present differently across genders, and females may sometimes be underdiagnosed due to these differences.
        
    """)

# Resources Section
elif nav_options == "Resources":
    st.header("Resources")
    st.write("Here are some helpful articles and links:")
    st.markdown("- [Understanding Autism](https://www.annualreviews.org/content/journals/10.1146/annurev.med.60.053107.121225)")
    st.markdown("- [Relationship between Autism and Gender ](https://link.springer.com/article/10.1186/s13229-015-0021-4)")

# Support Groups Section
elif nav_options == "Predictive System":
    st.header("Find Local Support Groups")


elif nav_options == "Tools":
    st.header("Tools for Autism Analysis")
    st.write("Here are some tools you can use for further analysis or learning:")
    
    st.markdown("""
        - **[Autism Screening Quiz](https://www.autismspeaks.org/screening-questionnaire)**: A self-assessment quiz for early autism detection.
        - **Data Visualization Tool**: Upload your own dataset and visualize key insights using this interactive tool.
    """)
    
# Import necessary libraries
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace 'your_data.csv' with your dataset file)

autism_data= pd.read_csv(r"C:\Users\NWUUSER\Downloads\Phase 3\autism_screening.csv")
# For demonstration, let's create a synthetic dataset
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability of the positive class

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print performance metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

# Classification report for detailed metrics
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Optional: Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()






# Visualization of Autism prevalence by Gender
st.write("### Prevalence of Autism by Gender")
autism_counts = autism_data.groupby(['gender', 'austim']).size().unstack(fill_value=0)
plt.figure(figsize=(10, 6))
autism_counts.plot(kind='bar', stacked=True, color=['lightblue', 'salmon'])
plt.title('Prevalence of Autism by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Country Population', fontsize=14)
plt.xticks(ticks=[0, 1], labels=['Male', 'Female'], rotation=0)  # Setting custom labels
plt.legend(title='Autism Diagnosis', labels=['No', 'Yes'], loc='upper right')
plt.tight_layout()
st.pyplot(plt.gcf())


plt.figure(figsize=(14,6))
top_countries = autism_data['contry_of_res'].value_counts().nlargest(10).index  # Get top 10 countries
sns.countplot(data=autism_data[autism_data['contry_of_res'].isin(top_countries)], x='contry_of_res', hue='austim', palette='Set3')
plt.title('Autism vs contry_of_res(Top 10 Countries)')
plt.xlabel('Per Country')
plt.ylabel('Count')
plt.legend(title="Autism (1=Yes, 0=No)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()




import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the application
st.title("Update Dataset")


    # Upload CSV file for dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
        # Load dataset from uploaded file
        try:
            data = pd.read_csv(uploaded_file)

            # Check for missing values in the dataset
            if data.isnull().values.any():
                st.warning("Warning: The dataset contains missing values.")
                # Display rows with missing values (optional)
                st.write(data[data.isnull().any(axis=1)])

            # Preprocess dataset: Convert categorical variables to numeric
            data['gender'] = data['gender'].map({'male': 0, 'female': 1})
            data['austim'] = data['austim'].map({'YES': 1, 'NO': 0})
           

            # Drop rows with NaN values in target variable 'austim'
            data.dropna(subset=['austim'], inplace=True)

            # Feature selection
            X = data[['gender']]  # Add more features if necessary
            y = data['austim']

            # Check for NaN values in target variable after dropping rows
            if y.isnull().any():
                st.error("Error: Target variable 'austim' still contains NaN values after preprocessing.")
                st.stop()

            # Train-test split
            X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Random Forest model
            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(X_train, y_train)

            # Model evaluation function
            def evaluate_model(model):
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                conf_matrix = confusion_matrix(y_test, y_pred)

                return accuracy, precision, recall, f1, roc_auc, conf_matrix

            # Evaluate models
            rf_metrics = evaluate_model(rf_model)

            # Display model metrics for Random Forest
            st.write("### Random Forest Model Performance")
            st.write(f"Accuracy: {rf_metrics[0]:.2f}")
            st.write(f"Precision: {rf_metrics[1]:.2f}")
            st.write(f"Recall: {rf_metrics[2]:.2f}")
            st.write(f"F1 Score: {rf_metrics[3]:.2f}")
            st.write(f"ROC AUC: {rf_metrics[4]:.2f}")

            # Confusion Matrix Plot for Random Forest
            plt.figure(figsize=(8, 6))
            sns.heatmap(rf_metrics[5], annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Random Forest Confusion Matrix')
            
            st.pyplot(plt.gcf())

        except Exception as e:
            st.error(f"Error loading file: {e}")

        # User input for prediction using slidebars/buttons
        st.write("### Make a Prediction")
        gender = st.selectbox('Gender', ['Male', 'Female'])
        
        # Convert user input into a DataFrame for prediction
        user_data = pd.DataFrame({
            'gender': [0 if gender == 'Male' else 1],
            
        })

        
        # Visualization of Autism prevalence by Gender
        st.write("### Prevalence of Autism by Gender")
        autism_counts = autism_data.groupby(['gender', 'austim']).size().unstack(fill_value=0)

        plt.figure(figsize=(10, 6))
        autism_counts.plot(kind='bar', stacked=True, color=['lightblue', 'salmon'])
        plt.title('Prevalence of Autism by Gender', fontsize=16)
        plt.xlabel('Gender', fontsize=14)
        plt.ylabel(' Country Population', fontsize=14)
        plt.xticks(ticks=[0, 1], labels=['Male', 'Female'], rotation=0) 
        plt.legend(title='Autism Diagnosis', labels=['No', 'Yes'], loc='upper right')

        st.pyplot(plt.gcf())
