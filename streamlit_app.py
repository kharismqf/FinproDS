import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import os
import numpy as np
from scipy import stats
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io
from pathlib import Path
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
from model.trythemodel import trythemodel
from model.aboutme import show_creator

# Navigasi dengan sidebar
with st.sidebar:
    page = option_menu(
        menu_title='Main Menu',
        options=['Understand the Data',
                 'Explore The Data',         
                 'Try the Model',
                 'Meet the Creator'],
        icons=[
            'bar-chart',        # EDA / data understanding
            'graph-up',         # baru untuk visualisasi interaktif
            'cpu',              # model / prediction
            'person-circle'     # about me
        ],
        default_index=0
    )



if page == 'Understand the Data':

    pd.set_option('display.max_rows', None)  # Menampilkan semua baris
    pd.set_option('display.max_columns', None)  # Jika ada banyak kolom

    st.markdown("<h1 style='text-align: center;'>Income Bracket Prediction using Census Data</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left;'>Project Overview:</h2>", unsafe_allow_html=True)
    st.markdown("""     <div style="
                        background-color: #f0e6d2;
                        padding: 16px;
                        border-radius: 10px;
                        color: #0b2c4c;
                        font-size: 16px;
                        font-family: 'Segoe UI';
                        text-align: justify;
                    ">
                    <b>Develop a classification model</b> that can predict income status 
                    <code>&le;50k</code> or <code>&gt;50k</code> for individual per year.
                    </div>
                    """, unsafe_allow_html=True)
    st.markdown("<div style='text-align: justify;'>This project aims to develop a classification model based on demographic & occupational data that can predict the income status (≤ $50 K vs > $50 K) of individuals. It is expected to be a decision-support tool for agencies to develop targeted social assistance programs, predict potential tax revenue, and inform data-driven policies and strategies (marketing, employment, etc.).</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])  # kolom tengah lebih lebar
    with col2:
        st.image("images/image1.jpg", use_container_width=True)

    with st.expander("📋 About Dataset"):
        st.write('## Dataset : Census income dataset')
        st.write('48.842 baris : Represent income for each individu')
        st.write('15 kolom : Represent 15 atributes')
        st.markdown("""
        ### 📄 Metadata Dataset
                    
        | **No.** | **Attribute Name** | **Definition**                               | **Data Type** | **Example**                         | **% Null** |
        | ------: | ------------------ | -------------------------------------------- | ------------- | ----------------------------------- | ---------- |
        |       1 | `age`              | Age (years)                                  | Quantitative  | 38, 42, 71                          | 0%         |
        |       2 | `workclass`        | Work category (Private, Self-emp, Gov, etc.) | Qualitative   | "Private", "Never-worked"           | 6%         |
        |       3 | `fnlwgt`           | Final weight (sampling weight)               | Quantitative  | 83311, 338409                       | 0%         |
        |       4 | `education`        | Education level                              | Qualitative   | "Bachelors", "Preschool"            | 0%         |
        |       5 | `education-num`    | Years of education (numerical)               | Quantitative  | 13, 9, 7                            | 0%         |
        |       6 | `marital-status`   | Marital status                               | Qualitative   | "Divorced", "Widowed"               | 0%         |
        |       7 | `occupation`       | Type of occupation                           | Qualitative   | "Sales", "Tech-support"             | 6%         |
        |       8 | `relationship`     | Relationship within household                | Qualitative   | "Wife", "Own-child"                 | 0%         |
        |       9 | `race`             | Race group                                   | Qualitative   | "White", "Black"                    | 0%         |
        |      10 | `sex`              | Gender                                       | Qualitative   | "Male", "Female"                    | 0%         |
        |      11 | `capital-gain`     | Capital gain amount                          | Quantitative  | 14084, 0, 5178                      | 0%         |
        |      12 | `capital-loss`     | Capital loss amount                          | Quantitative  | 0, 2042, 1902                       | 0%         |
        |      13 | `hours-per-week`   | Weekly working hours                         | Quantitative  | 40, 50, 70                          | 0%         |
        |      14 | `native-country`   | Country of origin                            | Qualitative   | "United-States", "Vietnam", "China" | 2%         |
        |      15 | `income`           | Income group (<=50K or >50K)                 | Qualitative   | "<=50K", ">50K"                     | 0%         |
        """)

        df_train = pd.read_csv('dataset/train_dashboard.csv')
        st.markdown("<h2 style='text-align: justify;'>Datasets</h2>", unsafe_allow_html=True)
        st.dataframe(df_train)
        if 'Unnamed: 0' in df_train.columns:
            df_train.drop(columns='Unnamed: 0', inplace=True)
        info_df_train = pd.DataFrame({
            "Column": df_train.columns,
            "Non-Null Count": df_train.notnull().sum().values,
            "Data Type": df_train.dtypes.astype(str).values
            })
        st.subheader("📋 Structure Datasets Information")
        st.dataframe(info_df_train)

        # Fungsi untuk visualisasi outlier
        def check_plot(df_cs, column):
            plt.figure(figsize=(16, 4))

            plt.subplot(1, 3, 1)
            sns.histplot(df_cs[column], bins=30)
            plt.title(f'Histogram - {column}')

            plt.subplot(1, 3, 2)
            stats.probplot(df_cs[column], dist="norm", plot=plt)
            plt.ylabel('Variable quantiles')

            plt.subplot(1, 3, 3)
            sns.boxplot(y=df_cs[column])
            plt.title(f'Boxplot - {column}')

            st.pyplot(plt.gcf())
            plt.clf()

        with st.expander("Data Pre Processing (Python with Google Colab)"):
                # 1. Missing value
                st.subheader("⚠️ Checking Missing Value")
                st.markdown("<div style='text-align: justify;'>There are some missing value in data.</div>", unsafe_allow_html=True)
                st.write("There is “?” data in the columns ‘workclass’, ‘occupation’, 'native-country'.")
                st.write("Solution: handling by replace the '?' data with 'other'.")  
                st.markdown("---")
                
                # 2. Duplikat
                st.subheader("⚠️ Checking Duplicate Data")
                st.markdown("<div style='text-align: justify;'>There are some duplicate data.</div>", unsafe_allow_html=True)
                st.write("train : 28 data (0.99914) and test: 1 data (0.99994)")
                st.write("Solution: handling with drop duplicate")  
                st.markdown("---")

                # 3. Outlier
                st.subheader("⚠️ Checking Outlier")
                st.markdown("<div style='text-align: justify;'>There are some outlier from column: 'fnlwgt', 'capital-gain', 'capital-loss', and 'hours-per-week'.</div>", unsafe_allow_html=True)
                st.write("Outliers were detected in multiple numerical columns. However, their presence is considered reasonable based on the data context. Rather than removing them, we analyzed their distribution to inform further transformations or feature engineering.")
                st.subheader("📉 Outlier Analysis on Numerical Features (Train Set Only)")

                # Buat DataFrame
                outlier_data = {
                    "Feature": ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"],
                    "Outliers (%)": [0.44, 3.05, 3.68, 8.33, 4.67, 27.66],
                    "Insight": [
                        "Minor outliers, no urgent action needed.",
                        "Moderate, but feature interpretability is weak → consider feature engineering.",
                        "Acceptable, may retain.",
                        "Significant → investigate distribution (many zeros + few extremes).",
                        "Similar to capital-gain → skewed distribution.",
                        "High outliers — consider transformation or capping."
                    ]
                }

                show_outliers = st.radio("🔎 Would you like to display outlier visualizations?", ["No", "Yes"], key="outlier_check")

                if show_outliers == "Yes":
                    st.markdown("### 🔍 Outlier Visualization by Feature")

                    with st.expander("📊 `fnlwgt` — Final Weight"):
                        st.write("Sampling weight used by the US Census Bureau.")
                        st.image("images/image2.jpg", caption="Distribution & Outliers of Final Weight")

                    with st.expander("📊 `capital-gain` — Capital Gain"):
                        st.write("Shows large right-skew due to a small group with significant gains.")
                        st.image("images/image3.jpg", caption="Distribution & Outliers of Capital Gain")

                    with st.expander("📊 `capital-loss` — Capital Loss"):
                        st.write("Like capital-gain, skewed by a few individuals with large losses.")
                        st.image("images/image4.jpg", caption="Distribution & Outliers of Capital Loss")

                    with st.expander("📊 `hours-per-week` — Weekly Working Hours"):
                        st.write("Unusually high values found, e.g. 99 hrs/week — might indicate over-reporting or edge cases.")
                        st.image("images/image5.jpg", caption="Distribution & Outliers of Weekly Working Hours")

                df_outliers = pd.DataFrame(outlier_data)
                st.dataframe(df_outliers, use_container_width=True)
                st.write("Solution: just handling the 'hours-per-week' column with IQR")  
                st.markdown("""
                <div style="
                    background-color: rgba(255, 255, 255, 0.85); 
                    color: #000000; 
                    padding: 15px; 
                    border-radius: 10px; 
                    border: 1px solid #ddd;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    font-size: 16px;
                ">
                <b>Note:</b><br>
                Although <code>capital-gain</code> and <code>capital-loss</code> contain significant outliers, no treatment was applied. 
                This is because the majority of values are <b>zero</b>, reflecting the realistic economic condition that 
                <b>most individuals do not possess assets to generate capital gains or losses</b>.

                Hence, the presence of extreme values for a small portion of individuals is justifiable and represents 
                actual income variations rather than anomalies.
                </div>
                """, unsafe_allow_html=True)
                st.markdown("---")

                # 4. Data Manipulation
                st.subheader("⚠️ Checking Standarization Data")
                st.markdown("<div style='text-align: justify;'>Before training the model, we verified white‑spaces, dots, and one‑hot‑encoding consistency in the categorical columns.</div>", unsafe_allow_html=True)
                with st.expander("📋  Show recommended fixes and examples", expanded=False):
                    # ⬇️  Put an illustrative image (optional)  
                    st.markdown(
                        """
                        **Detected issues & solutions**

                        • Extra whitespace inside many categorical values  → **strip the spaces**  
                        • Trailing dots found in the `income` column       → **standardise (remove dots)**  
                        • Class imbalance between `<=50K` and `>50K`       → **apply Random OverSampler**  
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("---")

# Halaman 2: Explore The Data
elif page == "Explore The Data":
    from model.eda import show_eda
    show_eda()

# Halaman 3: Try the Model
elif page == "Try the Model":
    from model.trythemodel import trythemodel        # impor FUNGSI‑nya
    trythemodel()                                    # jalankan

# Halaman 4: Meet the Creator
elif page == 'Meet the Creator':
    show_creator()
