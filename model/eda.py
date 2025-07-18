import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

@st.cache_data

def load_data():
    df = pd.read_csv('dataset/train_dashboard.csv')
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)
    return df

def show_eda():
    st.title("🔍 Explore the Data")
    df = load_data()

    st.header("📊 Categorical Feature Distribution")
    cat_cols = df.select_dtypes(include='object').drop(columns='income').columns.tolist()
    selected_cat = st.selectbox("🔎 Select a categorical column to explore:", cat_cols)

    fig = px.histogram(
        df,
        x=selected_cat,
        color="income",
        barmode="group",
        text_auto=True,
        category_orders={selected_cat: sorted(df[selected_cat].unique())},
        labels={"income": "Income Class"},
        color_discrete_sequence=["#95DCE2", "#82ADB3"]
    )
    fig.update_layout(
        title=f"Distribution of {selected_cat} by Income Class",
        xaxis_title=selected_cat,
        yaxis_title="Count",
        legend_title="Income",
        bargap=0.15
    )
    fig.update_traces(marker_line_width=1, opacity=0.85)
    st.plotly_chart(fig, use_container_width=True)

    st.header("📈 Numerical Feature Distribution")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_num = st.selectbox("Select a numerical column to explore:", num_cols)
    fig2 = px.histogram(
        df,
        x=selected_num,
        color='income',
        marginal="box",
        nbins=50,
        title=f"Distribution of {selected_num}",
        color_discrete_sequence=["#95DCE2", "#82ADB3"]
    )
    st.plotly_chart(fig2)

    st.header("🤔 Insightful Questions")

    st.markdown("###### 1. Are there people who have high working hours, high education, but still earn ≤$50K?")
    occupation_filter = st.selectbox("Select an Occupation:", df['occupation_grouped'].unique())
    hour_range = st.slider("Select Working Hour Range:", 1, 55, (0, 55))

    filtered = df[(df['occupation_grouped'] == occupation_filter) &
                  (df['education-num'] >= 13) &
                  (df['hours-per-week'].between(hour_range[0], hour_range[1]))]
    underpaid = filtered[filtered['income'] == '<=50K']
    count = len(underpaid)
    st.markdown(f"There are **{count}** underpaid individuals in the selected group.")
    fig_under = px.histogram(
        filtered,
        x='hours-per-week',
        color='income',
        nbins=20,
        title="Income Distribution for Selected Occupation & High Working Hours",
        color_discrete_sequence=["#95DCE2", "#82ADB3"]
    )
    st.plotly_chart(fig_under)
    

    st.markdown("###### 2. Is education level related to probability of earning >$50K?")
    edu_income = df.groupby('education-num')['income'].value_counts(normalize=True).unstack().fillna(0)
    fig3 = px.line(
        edu_income,
        y='>50K',
        title="Income >50K Ratio by Education Level",
        markers=True,
        color_discrete_sequence=["#82ADB3"]
    )
    fig3.update_layout(xaxis_title="Education Num", yaxis_title="Proportion >50K")
    st.plotly_chart(fig3)
    st.write("🔍 **Insight:** A clear upward trend is observed—higher education level (`education-num`) significantly increases the probability of earning more than $50K.")


    st.markdown("###### 3. At what age do people tend to reach their income peak?")

    # Create age bins
    df['age_group'] = pd.cut(df['age'], bins=[16, 25, 35, 45, 55, 65, 90], 
                             labels=["17–25", "26–35", "36–45", "46–55", "56–65", "66+"])
    
    # Group by age_group and income
    age_group_income = df.groupby('age_group')['income'].value_counts(normalize=True).unstack().fillna(0)
    
    # Plot
    fig4 = px.bar(
        age_group_income,
        x=age_group_income.index,
        y='>50K',
        title="📊 Proportion of >$50K Income by Age Group",
        labels={'x': 'Age Group', '>50K': 'Proportion >50K'},
        color_discrete_sequence=["#82ADB3"]
    )
    
    fig4.update_layout(
        xaxis_title="Age Group",
        yaxis_title="Proportion of Individuals with >$50K Income",
        bargap=0.3
    )
    
    st.plotly_chart(fig4)
    st.write("🔍 **Insight:** Individuals tend to reach their income peak between ages **46–55**, followed by **36–45** and **56–65**, indicating mid-to-late career is often the most financially rewarding phase.")


    st.markdown("###### 4. Heatmap: Numerical Correlation")
    numerical_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    corr = df[numerical_cols].corr()
    fig_corr, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5, ax=ax)
    st.pyplot(fig_corr)
    st.write("🔍 **Insight:** There’s a positive but modest correlation between `education-num`, `hours-per-week`, and high income. However, none of the numerical features show strong linear relationships.")


    st.markdown("###### 5. Which regions have the most high-income earners?")
    st.markdown("""     <div style="
                        background-color: #f0e6d2;
                        padding: 16px;
                        border-radius: 10px;
                        color: #0b2c4c;
                        font-size: 13px;
                        font-family: 'Segoe UI';
                        text-align: justify;
                    ">
                    <b>Note: Native region 'Other' includes Iran, Outlying-US(Guam, USVI, Samoa America), Holand-Netherlands, and Hong Kong.
                    </div>
                    """, unsafe_allow_html=True)
    region_income = df.groupby('native_region')['income'].value_counts(normalize=True).unstack().fillna(0).sort_values('>50K', ascending=False)
    fig5 = px.bar(
        region_income,
        x=region_income.index,
        y='>50K',
        title="Proportion of >$50K by Native Region",
        labels={"x": "Region", ">50K": "Proportion"},
        color_discrete_sequence=["#82ADB3"]
    )
    fig5.update_layout(yaxis_title="Proportion >50K", xaxis_title="Region")
    st.plotly_chart(fig5)
    st.write("🔍 **Insight:** Canada and 'Other' regions (which includes **Iran, Guam, USVI, Samoa America, Netherlands, and Hong Kong**) have the highest proportion of high-income individuals, followed closely by Asia and Europe.")


    st.markdown("###### 6. Who earns more based on relationship status?")
    rel_income = df.groupby('relationship')['income'].value_counts(normalize=True).unstack().fillna(0).sort_values('>50K', ascending=False)
    fig6 = px.bar(
        rel_income,
        x=rel_income.index,
        y='>50K',
        title="Proportion of >$50K by Relationship",
        labels={"x": "Relationship", ">50K": "Proportion"},
        color_discrete_sequence=["#95DCE2"]
    )
    fig6.update_layout(yaxis_title="Proportion >50K", xaxis_title="Relationship")
    st.plotly_chart(fig6)
    st.write("🔍 **Insight:** Individuals labeled as 'Wife' and 'Husband' have the highest proportion of income >$50K, suggesting that married individuals tend to earn more.")

