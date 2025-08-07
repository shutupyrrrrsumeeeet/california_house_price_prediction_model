import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle


col = ['MedInc',
 'HouseAge',
 'AveRooms',
 'AveBedrms',
 'Population',
 'AveOccup',
 'Longitude']

#Title
st.title('California Housing Price Prediction')

st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTQ0MpDUZ_WwjvqVEW2L3Oq26kpSdAHYmjQqjl4BpyRmqsSPyjpq885_3o&s')

st.header('Model of Housing Prices to Predict Median House Values in California',divider = True)

#st.subheader(''' User must Enter Given Values to Predict Price:
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Longitude']''')


st.sidebar.title('Select House Features 🏠')

st.sidebar.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRxMkjPDvGiJ1SnEOh4umUXIH_1ZasxPGr3vA&s')



# read_data
temp_df = pd.read_csv('california.csv')

random.seed(162)

all_values = []

for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])

with open('house_price_pred_ridge_model_pkl','rb') as f:
    chatgpt = pickle.load(f)

price = chatgpt.predict(final_value)[0]

import time


st.write(pd.DataFrame(dict(zip(col,all_values)),index=[1]))


progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Value')
placeholder1 = st.empty()
placeholder1.image('https://i.gifer.com/7Fmb.gif',width = 60)

if price>0:

    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
    
    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    placeholder1.empty()
    # st.subheader(body)

    st.success(body)
else:
    body = 'Invalid House features Values'
    st.warning(body)


    
    