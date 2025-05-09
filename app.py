
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set Streamlit page config
st.set_page_config(page_title='Medical Insurance Cost Predictor', layout='wide')

st.title('🩺 Medical Insurance Cost Predictor (Linear Regression)')

# 1️⃣ Upload Dataset
st.sidebar.header('Upload your CSV file')
uploaded_file = st.sidebar.file_uploader("Upload insurance.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader('📊 Raw Dataset')
    st.write(df.head())

    # 2️⃣ EDA
    st.subheader('🔍 Exploratory Data Analysis (EDA)')

    col1, col2 = st.columns(2)
    with col1:
        st.write('**Distribution of Insurance Charges**')
        fig, ax = plt.subplots()
        sns.histplot(df['charges'], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write('**Charges vs Age (Smoker Status)**')
        fig, ax = plt.subplots()
        sns.scatterplot(x='age', y='charges', hue='smoker', data=df, ax=ax)
        st.pyplot(fig)

    st.write('**Boxplot of Charges by Smoking Status**')
    fig, ax = plt.subplots()
    sns.boxplot(x='smoker', y='charges', data=df, ax=ax)
    st.pyplot(fig)

    # 3️⃣ Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Features and target
    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']

    # 4️⃣ Split + Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader('📈 Model Performance')
    st.success(f'✅ RMSE: **{rmse:.2f}**')
    st.success(f'✅ R² Score: **{r2:.2f}**')

    # 5️⃣ Coefficients (Feature Importance)
    st.subheader('💡 Feature Importance')
    coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    coeff_df_sorted = coeff_df.sort_values(by='Coefficient', ascending=False)
    st.write(coeff_df_sorted)

    fig, ax = plt.subplots(figsize=(8, 5))
    coeff_df_sorted.set_index('Feature').plot(kind='barh', ax=ax, legend=False)
    plt.title('Feature Importance (Coefficients)')
    st.pyplot(fig)

    # 6️⃣ Residual Plot
    st.subheader('🧐 Residual Plot (Model Fit Check)')
    fig, ax = plt.subplots()
    sns.residplot(x=y_pred, y=y_test - y_pred, lowess=True, line_kws={'color': 'red'}, ax=ax)
    plt.xlabel('Predicted Charges')
    plt.ylabel('Residuals')
    st.pyplot(fig)

    # 7️⃣ 🔮 Prediction Tool
    st.subheader('🔮 Predict Insurance Charges for a New Customer')

    # Collect user input
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=18, max_value=100, value=30)
        bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)

    with col2:
        sex = st.selectbox('Sex', ['male', 'female'])
        smoker = st.selectbox('Smoker', ['yes', 'no'])

    with col3:
        region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])

    # Prepare the input as model expects
    user_input = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex == 'male' else 0,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0,
    }

    user_df = pd.DataFrame([user_input])

    for col in X.columns:
        if col not in user_df.columns:
            user_df[col] = 0
    user_df = user_df[X.columns]

    if st.button('Predict Insurance Charges'):
        prediction = model.predict(user_df)[0]
        st.success(f'💰 Estimated Insurance Charges: **${prediction:,.2f}**')

else:
    st.info('👈 Please upload a CSV file to get started.')
