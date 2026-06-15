# Streamlit Application Initialization

'''
The Streamlit application requires several libraries for user interface creation, data handling, numerical operations, and model loading.

The following libraries are imported:

- Streamlit for building the web application
- Pandas for data handling
- NumPy for numerical operations
- Pickle for loading the trained machine learning model and scaler

These libraries form the foundation of the deployment pipeline and enable real-time ad revenue prediction.
'''
import streamlit as st
import numpy as np
import pandas as pd
import pickle

from datetime import datetime

st.set_page_config(
    page_title="Content Monetization Modeler",

    layout="wide"
)
st.title("📈 Content Monetization Modeler 📊")
st.markdown(
    "Predict YouTube Advertisement Revenue using Machine Learning"
)

model= pickle.load(
    open("linear_regression_model.pkl", "rb")
)
scaler=pickle.load(
    open("scaler.pkl","rb")
)

#Creating Input Sections:

st.header("Enter Video details")

col1, col2= st.columns(2)

with col1:
    views=st.number_input(
        "Views",
        min_value=0.0,
        value=10000.0
    )

    comments= st.number_input(
        "Comments",
        min_value=0.0,
        value=200.0
    )

    watch_time_minutes=st.number_input(
        "Watch Time(Minutes)",
        min_value=0.0,
        value=50000.0
    )

with col2:
    likes=st.number_input(
        "Likes",
        min_value=0.0,
        value=1000.0
    )

    subscribers=st.number_input(
        "Subscribers",
        min_value=0.0,
        value=500000.0
    )
    
    video_length_minutes=st.number_input(
        "Video Length(Minutes)",
        min_value=0.0,
        value=10.0
    )

col3, col4 = st.columns(2)

with col3:

    category=st.selectbox(
        'Category',
        [
            "Education",
            "Entertainment",
            "Gaming",
            "Lifestyle",
            "Music",
            "Technology"
        ]
    )

    country=st.selectbox(
        "Country",
        [
            "CA",
            "DE",
            "IN",
            "UK",
            "US"
        ]
    )

with col4:
    
    device = st.selectbox(
        "Device",
        [
            "Mobile",
            "T.V",
            "Tablet"
        ]
    )

    day_of_week = st.selectbox(

        "Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )


predict_btn = st.button("Predict Revenue")

if predict_btn:

    input_df=pd.DataFrame({
        "views":[views],
        "likes":[likes],
        "comments":[comments],
        "watch_time_minutes":[watch_time_minutes],
        "video_length_minutes":[video_length_minutes],
        "subscribers":[subscribers]

    })

    today= datetime.today()

    input_df["year"]=today.year
    input_df["month"]=today.month
    input_df["day"]=today.day


    # Feature Engineering::

    input_df['engagement_rate'] = (
        (likes+comments)/ views  
    )

    input_df["watch_time_per_view"] = (
        watch_time_minutes / views
    )

    input_df['engagement_per_minute'] =(
        (likes+comments)/watch_time_minutes
    )

    #Category Encoding:

    input_df["category_Entertainment"] = 0
    input_df["category_Gaming"] = 0
    input_df["category_Lifestyle"] = 0
    input_df["category_Music"] = 0
    input_df["category_Tech"] = 0


    if category == "Entertainment":
        input_df["category_Entertainment"] = 1

    elif category == "Gaming":
        input_df['category_Gaming'] = 1

    elif category == "Lifestyle":
        input_df['category_Lifestyle'] = 1

    elif category == "Music":
        input_df['category_Music'] = 1

    elif category == "Tech":
        input_df['category_Tech'] =1

    # Device Encoding:

    input_df["device_Mobile"] = 0
    input_df["device_TV"] = 0
    input_df["device_Tablet"] = 0


    if device == "Mobile":
        input_df["device_Mobile"] = 1

    elif device == "TV":
        input_df["device_TV"] = 1

    elif device == "Tablet":
        input_df["device_Tablet"] = 1

    # Country Encoding:

    input_df["country_CA"] = 0
    input_df["country_DE"] = 0
    input_df["country_IN"] = 0
    input_df["country_UK"] = 0
    input_df["country_US"] = 0

    if country == "CA":
        input_df["country_CA"] = 1

    elif country == "DE":
        input_df["country_DE"] = 1

    elif country == "IN":
        input_df["country_IN"] = 1

    elif country == "UK":
        input_df["country_UK"] = 1

    elif country == "US":
        input_df["country_US"] = 1

    # Day Encoding:  

    input_df["day_of_week_Monday"] = 0
    input_df["day_of_week_Saturday"] = 0
    input_df["day_of_week_Sunday"] = 0
    input_df["day_of_week_Thursday"] = 0
    input_df["day_of_week_Tuesday"] = 0
    input_df["day_of_week_Wednesday"] = 0

    if day_of_week == "Monday":
        input_df["day_of_week_Monday"] = 1

    elif day_of_week == "Saturday":
        input_df["day_of_week_Saturday"] = 1

    elif day_of_week == "Sunday":
        input_df["day_of_week_Sunday"] = 1

    elif day_of_week == "Thursday":
        input_df["day_of_week_Thursday"] = 1

    elif day_of_week == "Tuesday":
        input_df["day_of_week_Tuesday"] = 1

    elif day_of_week == "Wednesday":
        input_df["day_of_week_Wednesday"] = 1

   


    training_columns=['views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 'subscribers', 'year', 'month', 'day', 'category_Entertainment', 'category_Gaming', 'category_Lifestyle', 'category_Music', 'category_Tech', 'device_Mobile', 'device_TV', 'device_Tablet', 'country_CA', 'country_DE', 'country_IN', 'country_UK', 'country_US', 'day_of_week_Monday', 'day_of_week_Saturday', 'day_of_week_Sunday', 'day_of_week_Thursday', 'day_of_week_Tuesday', 'day_of_week_Wednesday', 'engagement_rate', 'watch_time_per_view', 'engagement_per_minute']

    input_df = input_df[training_columns]

    input_scaled= scaler.transform(input_df)

    # Actual Prediction :

    prediction = model.predict(input_scaled)

    st.subheader("Revenue Prediction")

    st.success(
        f"Predicted Revenue: ${prediction[0]:.2f}"
    )

    st.balloons()


    st.markdown("---")

    st.markdown("""
    ### About This Project

    This machine learning application predicts YouTube advertisement revenue
    based on video engagement metrics such as views, likes, comments,
    watch time, subscribers, category, device type, country, and upload day.

    Model Used: Linear Regression

    Developed using:
    - Python
    - Pandas
    - Scikit-Learn
    - Streamlit
    """)
   
    