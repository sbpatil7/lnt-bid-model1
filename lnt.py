
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
@st.cache_data
def load_data(file_path):
    data = pd.read_excel(file_path)

    categorical_cols = [
        'Product Type', 'Project Region', 'Project Geography/ Location', 'Licensor',
        'Shell (MOC)', 'Weld Overlay/ Clad Applicable (Yes or No)', 'MOC of WOL', 
        'Sourcing Restrictions (Yes or No)'
    ]

    numerical_cols = [
        'ID (mm)', 'Weight (MT)', 'Cost ($ / Kg)', 'Unit Cost($)', 'Total Cost($)',
        'Off top (%)', 'Price($ / Kg)', 'Unit Price($)', 'Total price($)'
    ]

    data.fillna(method='ffill', inplace=True)

    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col + '_encoded'] = le.fit_transform(data[col].astype(str))
        le_dict[col] = le

    X = data[[col + '_encoded' for col in categorical_cols] + numerical_cols]
    y = LabelEncoder().fit_transform(data['Result(w/L)'])

    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    return X, y, scaler, le_dict, categorical_cols, numerical_cols, data

X, y, scaler, le_dict, categorical_cols, numerical_cols, original_data = load_data('data.xlsx')

# Train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(X, y)

# Model evaluation
st.title('🔮 Bid Win Predictor Dashboard')

accuracy = accuracy_score(y_test, model.predict(X_test))
st.sidebar.metric("Model Accuracy", f"{accuracy:.2%}")

st.subheader('Model Evaluation')
report = classification_report(y_test, model.predict(X_test), output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Confusion matrix plot
st.subheader('Confusion Matrix')
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# Feature Importance
st.subheader('Feature Importance')
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
st.bar_chart(importances)

# User input form
st.subheader('Predict Bid Outcome')

input_data = {}
col1, col2 = st.columns(2)

with col1:
    for col in categorical_cols:
        input_data[col] = st.selectbox(col, sorted(original_data[col].dropna().unique()))

with col2:
    for col in numerical_cols:
        input_data[col] = st.number_input(col, value=float(original_data[col].median()))

input_df = pd.DataFrame([input_data])

# Encode categorical input
for col in categorical_cols:
    le = le_dict[col]
    input_df[col + '_encoded'] = le.transform([input_df[col][0]])

# Prepare final input for model
input_df_model = input_df[[col + '_encoded' for col in categorical_cols] + numerical_cols]
input_df_model[numerical_cols] = scaler.transform(input_df_model[numerical_cols])

# Prediction button
if st.button('Predict'):
    prediction = model.predict(input_df_model)[0]
    pred_proba = model.predict_proba(input_df_model)[0][prediction]
    outcome = '✅ Win' if prediction == 1 else '❌ Lose'
    st.success(f'Prediction: {outcome} ({pred_proba:.2%} confidence)')

    # Explain top influencing features
    st.subheader("🔍 Why this outcome?")
    input_values = input_df_model.values[0]
    contributions = model.feature_importances_ * input_values
    top_features = pd.Series(contributions, index=input_df_model.columns).abs().sort_values(ascending=False).head(3)

    reasons = []
    for feature in top_features.index:
        user_val = input_df_model[feature].values[0]
        median_val = X[feature].median()
        if user_val > median_val:
            reasons.append(f"🔺 **{feature}** is higher than usual")
        elif user_val < median_val:
            reasons.append(f"🔻 **{feature}** is lower than usual")
        else:
            reasons.append(f"⚖️ **{feature}** is at typical level")

    if prediction == 0:
        st.error("Possible reasons for losing the bid:")
    else:
        st.info("Possible reasons for winning the bid:")

    for r in reasons:
        st.markdown(f"- {r}")
