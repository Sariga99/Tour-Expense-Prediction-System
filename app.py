import streamlit as st
import pandas as pd
import numpy as np
import time
import sklearn
from xgboost import XGBRegressor

#import matplotlib.pyplot as plt
#from plotly import graph_objs as go
#from sklearn.ensemble import RandomForestRegressor
st.title("Singapore Expense Prediction ")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.pixelstalk.net/wp-content/uploads/2016/08/Free-Travel-Backgrounds-Download-HD.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


add_bg_from_url()

import pickle

st.set_option('deprecation.showPyplotGlobalUse', False)

# Loading up the Regression model we created
pickle_in = open("Final_model.pkl", "rb")
model = pickle.load(pickle_in)

# # Loading scaler
pickle_in = open("scaling_feature.pkl","rb")
scaler = pickle.load(pickle_in)

# Caching the model for faster loading
#@st.cache

#data = pd.read_excel("data//Singapore_preprocessed.xlsx")

nav = st.sidebar.radio("Menu",["Home","Prediction","Popular Hotels","Sightseeing Attractions"])

if nav == "Home":

    st.video("https://www.youtube.com/watch?v=2DQ4arfP-ls",start_time=0)
    st.write("Singapore is a popular destination for tourists due "
             "to its diverse culture, modern cityscape, and numerous attractions.")
    st.write("Tourists can also experience local culture by visiting the various food markets "
             "and street stalls, or by taking a walk through the city's various neighborhoods, such as "
             "Little India and Arab Street. Singapore is also a hub for "
             "business and finance, making it a popular destination for both leisure and business travelers.")


if nav == "Popular Hotels":

    # Create a list of top 10 hotels
    hotels = ["Mandarin Orchard Singapore", "V Hotel Lavender", "York Hotel", "Ibis Singapore on Bencoolen",
              "Marina Bay Sands Singapore", "The Elizabeth", "Royal Plaza"]

    # Create a dictionary of hotel URLs
    hotel_urls = {
        "Mandarin Orchard Singapore": "https://mandarin-orchard-hotel-singapore.hotel-ds.com/en/",
        "V Hotel Lavender": "https://vhotel.sg/v-hotel-lavender.shtml?gclid=Cj0KCQiA-oqdBhDfARIsAO0TrGFUXok4UotiIu20PVFFWrnHyPNJX7GX6HdX2g8Z93d-XyMmMrQmSUcaAnnDEALw_wcB&gclsrc=aw.ds",
        "York Hotel": "https://www.yorkhotel.com.sg/",
        "Ibis Singapore on Bencoolen": "https://all.accor.com/hotel/6657/index.en.shtml",
        "Marina Bay Sands Singapore": "https://www.visitsingapore.com/see-do-singapore/recreation-leisure/resorts/marina-bay-sands/",
        "The Elizabeth": "https://ar.trivago.com/en-145/oar/the-elizabeth-hotel-by-far-east-hospitality-singapore?tc=22&search=100-40920",
        "Royal Plaza": "https://www.royalplaza.com.sg/",
    }

    # Create a radio button for each hotel
    st.subheader("Do you want to have a look at the most popular hotels in Singapore?")
    selected_hotel = st.radio("Select", hotels)

    #st.bar_chart(data, x='MainHotel', y='Total_Expense', width=600, height=400)

    # Get the URL of the selected hotel from the dictionary
    url = hotel_urls[selected_hotel]

    # Use the st.markdown function to embed the iframe in the web page
    st.markdown(f"<iframe src='{url}' width='800' height='400'></iframe>", unsafe_allow_html=True)

if nav == "Prediction":

    st.header("Do you want to know the expected travel expense?")

    city = st.selectbox("From which city are you travelling from? ",["Bali","Bandung","Batam","Jakarta",
                                                                     "Medan","Surabaya", "Tanjung Balai", "Tanjung Pinang",
                                                                     "Yogyakarta","Other cities"], index=0)
    if city == "Bali":
        value1 = 0
    elif city == "Bandung":
        value1 = 1
    elif city == "Batam":
        value1 = 2
    elif city == "Jakarta":
        value1 = 3
    elif city == "Medan":
        value1 = 4
    elif city == "Other cities":
        value1 = 5
    elif city == "Surabaya":
        value1 = 6
    elif city == "Tanjung Balai":
        value1 = 7
    elif city == "Tanjung Pinang":
        value1 = 8
    else:
        value1 = 9

    purpose = st.radio("What is the purpose of visit?", ["Business", "Education", "Healthcare", "Leisure", "Others"])
    if purpose == "Business":
        value2 = 0
    elif purpose == " Education":
        value2 = 1
    elif purpose == " Healthcare":
        value2 = 2
    elif purpose == " Leisure":
        value2 = 3
    else:
        value2 = 4

    visit = st.radio("Are you visiting Singapore for the first time?", ["Yes", "No"])
    if visit == "Yes":
        value3 = 1
    else:
        value3 = 0

    value4 = st.number_input("How many days of stay are you planning for?")

    travel_type = st.selectbox("Select your travel type", ['Business Non-Packaged', 'Non-Packaged', 'Packaged'], index=0)
    if travel_type == "Business Non-Packaged":
        value5 = 0
    elif travel_type == "Non-Packaged":
        value5 = 1
    else:
        value5 = 2

    gender = st.radio("Select gender ", ["Male", "Female"], index=0)
    if gender == "Male":
        value6 = 1
    else:
        value6 = 0

    stay = st.selectbox("Where are you planning to stay?", ["Not staying", "Hostel", "Hotel", "Others", "Own Residence",
                                                            "Service apartment", "With relatives/friends"], index=0)
    if stay == "Not staying":
        value7 = 0
    elif stay == "Hostel":
        value7 = 1
    elif stay == "Hotel":
        value7 = 2
    elif stay == "Others":
        value7 = 3
    elif stay == "Own Residence":
        value7 = 4
    elif stay == "Service apartment":
        value8 = 5
    else:
        value7 = 6

    if value7 == 2:
        # Enable the second select box
        hotel = st.selectbox("Select your preferred hotel", ["Mandarin Orchard Singapore", "V Hotel Lavender",
                                                             "York Hotel", "Ibis Singapore on Bencoolen",
                                                             "Marina Bay Sands Singapore", "The Elizabeth",
                                                             "Royal Plaza", "Not Specified", "Others"], index=0)
        if hotel == "Mandarin Orchard Singapore":
            value8 = 2
        elif hotel == "V Hotel Lavender":
            value8 = 8
        elif hotel == "York Hotel":
            value8 = 9
        elif hotel == "Ibis Singapore on Bencoolen":
            value8 = 1
        elif hotel == "Marina Bay Sands Singapore":
            value8 = 3
        elif hotel == "The Elizabeth":
            value8 = 7
        elif hotel == "Royal Plaza":
            value8 = 6
        elif hotel == "Others":
            value8 = 5
        elif hotel == "Concorde Hotel Singapore":
            value8 = 0
        else:
            value8 = 4
    else:
        # Pass a default value for the second select box
        value8 = 4

    path = st.radio("Are you travelling to Singapore via ?", ["Air", "Sea", "Land"], index=0)
    if path == "Air":
        value9 = 0
    elif path == "Sea":
        value9 = 1
    else:
        value9 = 2
    value10 = st.number_input("How many people are accompanying?")

    month = st.selectbox('In which month are you planning to travel', ['January','February','March','April','May','June','July','August','September','October','November','December'])
    if month == 'January':
        month = 1
    elif month == 'February':
        month = 2
    elif month == 'March':
        month = 3
    elif month == 'April':
        month = 4
    elif month == 'May':
        month = 5
    elif month == 'June':
        month = 6
    elif month == 'July':
        month = 7
    elif month == 'August':
        month = 8
    elif month == 'September':
        month = 9
    elif month == 'October':
        month = 11
    elif month == 'November':
        month = 12

    value11 = st.number_input("How much will be your expected baggage weight in pounds?")
    #value11 = st.slider('How much will be your expected baggage weight in pounds?', min_value=0.0, max_value=10000.0,
                       #value=0.0)
    #value11 = np.array(value11)
    #value11 = value11.reshape(1, -1)
    #value11 = scaler.fit_transform(value11)

    value12 = st.number_input("How much will you spend for shopping?")
    #value12 = np.array(value12)
    #value12 = value12.reshape(1, -1)
    p=pd.DataFrame([value11,value12])

    d = {'Weights_QTR': [value11], 'shopping_exp': [value12]}
    df = pd.DataFrame(data=d)

    #data = np.concatenate((value11, value12))
    #p = scaler.fit_transform(data)
    #scaled = [value11, value12]
    #scaled1 = np.vectorize(np.int_)
    #x = np.array(list(map(np.int_, np.array(scaled))))
    #q = [float(x) for x in scaled]
    #r = [np.array(q)]
    r = scaler.fit_transform(df)
    print(r)

    # Create a list of input values
    inputs = [value1, value2, value3, value4, value5, value6, value7, value8, value9, value10, month, r[0][0], r[0][1]]
    init_features = [float(x) for x in inputs]
    final_features = [np.array(init_features)]
    pred = model.predict(final_features)
    pred = np.exp(pred)
    if st.button("Predict"):

        max_value = 100
        progress_bar = st.progress(0)
        for i in range(max_value):
            # Update the progress bar
            progress_bar.progress(i + 1)

            # Sleep for a short period of time
            time.sleep(0.01)

        st.success(f"Your expected expense for this Singapore trip is SGD {np.round(pred,2)}")
        #st.balloons()
if nav == "Sightseeing Attractions":
    # Create a list of top places
    places = ["Gardens by the Bay", "Universal Studios Singapore",
              "Singapore Zoo", "Sentosa Island", ]

    # Create a dictionary of hotel URLs
    places_urls = {
        "Gardens by the Bay": "https://grant-associates.uk.com/projects/gardens-by-the-bay",
        "Universal Studios Singapore": "https://www.pelago.co/en-SG/activity/pok40-rws-universal-studios-ticket-singapore/",
        "Singapore Zoo": "https://www.kkday.com/en-sg/product/39249-wildlife-reserves-4-in-1-park-hopper-plus-ticket-singapore",
        "Sentosa Island": "https://trevallog.com/sentosa-island-singapore/"
    }

    # Create a radio button for each hotel
    st.subheader("Most popular sightseeing attractions in Singapore")
    selected_places = st.radio("Select", places)

    # st.bar_chart(data, x='MainHotel', y='Total_Expense', width=600, height=400)

    # Get the URL of the selected hotel from the dictionary
    url = places_urls[selected_places]

    # Use the st.markdown function to embed the iframe in the web page
    st.markdown(f"<iframe src='{url}' width='800' height='400'></iframe>", unsafe_allow_html=True)

st.sidebar.image: st.sidebar.image("data//singapore.jpg", use_column_width=True)