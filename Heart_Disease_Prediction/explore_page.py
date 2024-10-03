import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def show_explore_page():
    st.title("Exploratory Data Analysis")

    st.write(

        "Percentage of the people having heart disease"
    )


    dataset = pd.read_csv('Heart_Disease_Prediction/heart-disease.csv')
    df=dataset

    # Plot the value counts with a bar graph
    fig1, ax1 = plt.subplots()
    data=df.target.value_counts()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.write("""#### Having disease(1) Vs No disease (0)""")

    st.pyplot(fig1)
