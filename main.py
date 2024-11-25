import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv(r"symtoms_df.csv")
description = pd.read_csv(r"description.csv")
precautions = pd.read_csv(r"precautions_df.csv")
medications = pd.read_csv(r"medications.csv")
diets = pd.read_csv(r"diets.csv")
workout = pd.read_csv(r"workout_df.csv")

# Preprocess data
df['Symptom_4'] = df['Symptom_4'].replace(np.nan, 'Empty')
df1 = df.groupby('Disease').agg(
    Symptoms1=('Symptom_1', 'unique'),
    Symptoms2=('Symptom_2', 'unique'),
    Symptoms3=('Symptom_3', 'unique'),
    Symptoms4=('Symptom_4', 'unique')
).reset_index()
df1['Symptoms1'] = df1['Symptoms1'].apply(lambda x: ','.join(x))
df1['Symptoms2'] = df1['Symptoms2'].apply(lambda x: ','.join(x))
df1['Symptoms3'] = df1['Symptoms3'].apply(lambda x: ','.join(x))
df1['Symptoms4'] = df1['Symptoms4'].apply(lambda x: ','.join(x))
df1['all_symptoms'] = df1['Symptoms1'] + ',' + df1['Symptoms2'] + ',' + df1['Symptoms3'] + ',' + df1["Symptoms4"]
df1['all_symptoms'] = df1['all_symptoms'].apply(lambda x: ','.join(sorted(set(x.split(',')))))
df1['all_symptoms'] = df1['all_symptoms'].apply(lambda x: ' '.join([i.strip() for i in x.split(',')]))

# Bag of Words model
bow = CountVectorizer()
result = bow.fit_transform(df1['all_symptoms'])

# Helper function to get additional info
def helper(disease):
    desc = description[description['Disease'] == disease]['Description'].values
    desc = " ".join(desc) if len(desc) > 0 else "No description available."

    pre = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values
    pre = ", ".join(pre[0]) if len(pre) > 0 else "No precautions available."

    med = medications[medications['Disease'] == disease]['Medication'].values
    med = ", ".join(med) if len(med) > 0 else "No medications available."

    diet = diets[diets['Disease'] == disease]['Diet'].values
    diet = ", ".join(diet) if len(diet) > 0 else "No diets available."

    wrkout = workout[workout['disease'] == disease]['workout'].values
    wrkout = ", ".join(wrkout) if len(wrkout) > 0 else "No workouts available."

    return desc, pre, med, diet, wrkout

# Prediction function
def predicts(inputs):
    def preprocess(test):
        return ' '.join([i for i in test.split(',')])

    # Transform the input using the same CountVectorizer
    results = bow.transform([preprocess(inputs)]).toarray()
    if np.sum(results) == 0:
        return ["No matching disease found."]
    else:
        similarities = cosine_similarity(result, results).flatten()
        top_indices = np.argsort(similarities)[-2:][::-1]
        return df1.iloc[top_indices]['Disease'].values

# Streamlit UI
st.set_page_config(page_title="Disease Prediction", layout="wide", page_icon="🛡️")


# Title and introduction
st.title("🩺 Symptom-Based Disease Predictor")
st.markdown("""
Welcome to the **Disease Predictor App**!  
Enter symptoms separated by commas (e.g., `fever, headache`)
""")

# User input with a submit button
inputs = st.text_input("Enter symptoms:", placeholder="e.g., fever, headache")
if st.button("Submit"):
    if inputs:
        # Prediction
        predictions = predicts(inputs)
        if predictions[0] == "No matching disease found.":
            st.error(predictions[0])
        else:
            st.success("Predicted Diseases:")
            for disease in predictions:
                with st.expander(f"Details for: {disease}"):
                    desc, pre, med, diet, wrkout = helper(disease)
                    st.markdown(f"### 📝 Description")
                    st.write(desc)

                    st.markdown(f"### 🛡️ Precautions")
                    st.write(pre)

                    st.markdown(f"### 💊 Medications")
                    st.write(med if med else "No medications available.")

                    st.markdown(f"### 🍎 Diet Recommendations")
                    st.write(diet if diet else "No diet recommendations available.")

                    st.markdown(f"### 🏋️ Workouts")
                    st.write(wrkout if wrkout else "No workout suggestions available.")

# Styling for buttons and input
st.markdown("""
<style>
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    text-align: center;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 5px;
}
div.stTextInput > label {
    color: white;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)


