import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap
import matplotlib
matplotlib.use('Agg')

# Load the dataset
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / 'data' / 'raw' / 'heart_disease.csv'
data = pd.read_csv(DATA_PATH)

# Check for missing values before imputation
print("Missing values before imputation:")
print(data.isnull().sum())

# Handle missing values for numerical and categorical columns
numerical_cols = ['age', 'education', 'cigsPerDay', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'totChol']
categorical_cols = ['sex', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']

imputer_num = SimpleImputer(strategy='mean')
data[numerical_cols] = imputer_num.fit_transform(data[numerical_cols])

imputer_cat = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

print("Missing values after imputation:")
print(data.isnull().sum())

# Split the data
X = data.drop('TenYearCHD', axis=1)
y = data['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

output_dir = os.path.join(os.getcwd(), 'visualizations')
os.makedirs(output_dir, exist_ok=True)

def get_user_input():
    gender = int(input("Enter gender (1 for Male, 0 for Female): "))
    age = float(input("Enter age: "))
    education = float(input("Enter education level (1-4): "))
    current_smoker = int(input("Are you a current smoker? (1 for Yes, 0 for No): "))
    cigs_per_day = float(input("How many cigarettes do you smoke per day? "))
    bpm = int(input("Are you on blood pressure medication? (1 for Yes, 0 for No): "))
    prevalent_stroke = int(input("Have you had a prevalent stroke? (1 for Yes, 0 for No): "))
    prevalent_hyp = int(input("Do you have prevalent hypertension? (1 for Yes, 0 for No): "))
    diabetes = int(input("Do you have diabetes? (1 for Yes, 0 for No): "))
    tot_chol = float(input("Enter total cholesterol: "))
    sys_bp = float(input("Enter systolic blood pressure: "))
    dia_bp = float(input("Enter diastolic blood pressure: "))
    bmi = float(input("Enter BMI: "))
    heart_rate = float(input("Enter heart rate: "))
    glucose = float(input("Enter glucose level: "))

    user_input = pd.DataFrame([[gender, age, education, current_smoker, cigs_per_day, bpm, prevalent_stroke, prevalent_hyp,
                                diabetes, tot_chol, sys_bp, dia_bp, bmi, heart_rate, glucose]],
                              columns=X.columns)

    return user_input

user_input = get_user_input()
user_input_scaled = scaler.transform(user_input)
user_prediction = model.predict(user_input_scaled)
user_probability = model.predict_proba(user_input_scaled)[:, 1]

# Diet recommendation based on the prediction
if user_prediction == 1:
    diet_recommendation = """
    Diet Plan for High-Risk Individuals:
    - Increase fiber intake with vegetables, fruits, and whole grains.
    - Limit unhealthy fats, such as trans fats and saturated fats.
    - Choose healthy fats, such as olive oil and avocados.
    - Include more fish in your diet, especially those high in omega-3 fatty acids (e.g., salmon, sardines).
    - Monitor cholesterol levels and aim for a low-sodium diet.
    - Avoid processed foods and high sugar intake.
    - Stay hydrated and avoid sugary drinks.
    """
else:
    diet_recommendation = """
    Diet Plan for Low-Risk Individuals:
    - Continue maintaining a balanced diet with plenty of fruits, vegetables, and lean proteins.
    - Regular physical activity to maintain a healthy weight and cardiovascular health.
    - Limit alcohol intake and avoid smoking.
    - Regularly monitor cholesterol, blood pressure, and glucose levels.
    """

# Function to calculate health score based on user inputs
def calculate_health_score(user_input):
    bmi = user_input['BMI'].values[0]
    sys_bp = user_input['sysBP'].values[0]
    glucose = user_input['glucose'].values[0]
    smoking = user_input['currentSmoker'].values[0]
    
    score = 0
    if bmi > 30:
        score += 2  # High BMI
    if sys_bp > 130:
        score += 2  # High systolic blood pressure
    if glucose > 100:
        score += 2  # Elevated glucose levels
    if smoking == 1:
        score += 2  # Smoking
    
    return score

# Function to generate natural language health explanation
def generate_health_explanation(user_input, user_prediction):
    explanation = "Based on your input, your health risk for heart disease is "

    if user_prediction == 1:
        explanation += "high. Several factors contribute to this risk, including:\n"
    else:
        explanation += "low. Keep up with a healthy lifestyle! Here's why:\n"

    # Check each condition and append to the explanation
    if user_input['BMI'].values[0] > 30:
        explanation += "- High BMI (above 30), which can increase your heart disease risk.\n"
    if user_input['sysBP'].values[0] > 130:
        explanation += "- High systolic blood pressure (above 130), a known risk factor for heart disease.\n"
    if user_input['glucose'].values[0] > 100:
        explanation += "- Elevated glucose levels, which may indicate risk for diabetes and heart disease.\n"
    if user_input['currentSmoker'].values[0] == 1:
        explanation += "- Smoking, which significantly increases heart disease risk.\n"

    if user_prediction == 0:
        explanation += "Your risk factors are within a healthy range, but it's important to continue a balanced diet, regular exercise, and monitoring of health parameters."

    return explanation

# Generate the PDF report
def generate_pdf_report(user_input, user_prediction, user_probability, diet_recommendation, health_score, health_explanation):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.rect(5, 5, 200, 287)

    pdf.set_font("Helvetica", size=14, style='B')
    pdf.cell(200, 10, text="Heart Disease Prediction Report", align="C")

    pdf.set_font("Helvetica", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, text="User  Input Data with Explanations:")
    pdf.ln(5)
    pdf.set_font("Helvetica", size=10)
    for col in user_input.columns:
        explanation = ""
        if col == 'sex':
            explanation = "(1 = Male, 0 = Female)"
        elif col == 'currentSmoker':
            explanation = "(1 = Yes, 0 = No)"
        elif col == 'BPMeds':
            explanation = "(1 = Yes, 0 = No)"
        elif col == 'prevalentStroke':
            explanation = "(1 = Yes, 0 = No)"
        elif col == 'prevalentHyp':
            explanation = "(1 = Yes, 0 = No)"
        elif col == 'diabetes':
            explanation = "(1 = Yes, 0 = No)"
        pdf.cell(60, 10, text=f"{col}: {user_input[col].values[0]} {explanation}")
        pdf.ln(5)

    pdf.ln(10)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, text="Prediction and Probability:")
    pdf.ln(5)
    pdf.cell(200, 10, text=f"Prediction: {'High risk of heart disease' if user_prediction == 1 else 'Low risk of heart disease'}")
    pdf.ln(5)
    probability_percentage = f"{user_probability[0] * 100:.2f}%"
    pdf.cell(200, 10, text=f"Probability of heart disease: {probability_percentage}")

    pdf.ln(10)
    pdf.cell(200, 10, text="Diet Recommendation:")
    pdf.ln(5)
    pdf.multi_cell(0, 10, text=diet_recommendation)

    # Add health score to the report
    pdf.ln(10)
    pdf.cell(200, 10, text="Health Score:")
    pdf.ln(5)
    pdf.cell(200, 10, text=f"Your health score is: {health_score}")

    # Add health explanation to the report
    pdf.ln(10)
    pdf.cell(200, 10, text="Health Risk Explanation:")
    pdf.ln(5)
    pdf.multi_cell(0, 10, text=health_explanation)

    return pdf, "Heart_Disease_Prediction_Report.pdf"

# Calculate the health score
health_score = calculate_health_score(user_input)

# Get the natural language explanation
health_explanation = generate_health_explanation(user_input, user_prediction)

# Generate the report
pdf, pdf_output = generate_pdf_report(user_input, user_prediction, user_probability, diet_recommendation, health_score, health_explanation)

def generate_shap_summary_plot(model, X_train_scaled, X_train, output_dir):
    # Create SHAP values for the model
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_train_scaled)

    # Saving SHAP plot as image
    shap_img_path = os.path.join(output_dir, 'shap_summary_plot.png')
    plt.figure(figsize=(8, 6))  # Adjusting figure size to ensure visibility
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)  # Set show=False to prevent pop-up
    plt.savefig(shap_img_path, format='png', bbox_inches='tight')  # Ensure tight layout for the plot
    plt.close()
    
    return shap_img_path

def add_visualizations_to_pdf(pdf, model, X_train_scaled, X_train, output_dir):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_img_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_img_path)
    plt.close()
    pdf.add_page()
    pdf.cell(200, 10, text="Confusion Matrix and Explanation", align="C")
    pdf.ln(10)
    pdf.image(cm_img_path, x=10, w=180)
    pdf.ln(5)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 10, text="The Confusion Matrix shows the number of true positives, false positives, true negatives, and false negatives.")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_prob))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    roc_img_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_img_path)
    plt.close()
    pdf.add_page()
    pdf.cell(200, 10, text="ROC Curve and Explanation", align="C")
    pdf.ln(10)
    pdf.image(roc_img_path, x=10, w=180)
    pdf.ln(5)
    pdf.multi_cell(0, 10, text="The ROC curve shows the trade-off between sensitivity and specificity.")

    # Generate and add SHAP summary plot to the report
    shap_img_path = generate_shap_summary_plot(model, X_train_scaled, X_train, output_dir)
    
    pdf.add_page()
    pdf.cell(200, 10, text="SHAP Summary Plot", align="C")
    pdf.ln(10)
    pdf.image(shap_img_path, x=10, w=180)
    pdf.ln(5)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 10, text="SHAP values provide insights into how each feature contributes to the model's predictions.")

    return pdf

# Add visualizations to the PDF
pdf = add_visualizations_to_pdf(pdf, model, X_train_scaled, X_train, output_dir)

# Save the final PDF
pdf.output(pdf_output)

print(f"PDF Report generated and saved as {pdf_output}.")
