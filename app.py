import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from scipy import stats
from scipy.stats import boxcox
import pickle
from datetime import date, timedelta
from streamlit_option_menu import option_menu
from scipy.special import inv_boxcox



# Set page configuration
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Singapore Flat resale price Predictor",
    page_icon=r'asset/icon.jpeg',
)


# Injecting CSS for custom styling
st.markdown("""
    <style>
    /* Tabs */
    div.stTabs [data-baseweb="tab-list"] button {
        font-size: 25px;
        color: #ffffff;
        background-color: #4CAF50;
        padding: 10px 20px;
        margin: 10px 2px;
        border-radius: 10px;
    }
    div.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #009688;
        color: white;
    }
    div.stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #3e8e41;
        color: white;
    }
    /* Button */
    .stButton>button {
        font-size: 22px;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 16px;
    }
    .stButton>button:hover {
        background-color: #3e8e41;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to perform Box-Cox transformation on a single value using a given lambda
def transform_single_value(value, lmbda):
    if value is None:
        return None  # Handle missing value
    transformed_value = boxcox([value], lmbda=lmbda)[0]
    return transformed_value

def reverse_boxcox_transform(predicted, lambda_val):
    return inv_boxcox(predicted, lambda_val)

# Load the saved lambda values
with open(r'pkls/boxcox_lambdas.pkl', 'rb') as f:
    lambda_dict = pickle.load(f)

    
with open(r'pkls/pkls\\scale_reg.pkl', 'rb') as f:
    scale_reg = pickle.load(f)

with open(r'pkls/XGB_model.pkl', 'rb') as f:
    xgb_Reg = pickle.load(f)
    


with st.sidebar:
    st.markdown("<hr style='border: 2px solid #ffffff;'>", unsafe_allow_html=True)
    
    selected = option_menu(
        "Main Menu", ["About", 'Genie'], 
        icons=['house-door-fill', 'bar-chart-fill'], 
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "gray"},
            "icon": {"color": "#000000", "font-size": "25px", "font-family": "Times New Roman"},
            "nav-link": {"font-family": "inherit", "font-size": "22px", "color": "#ffffff", "text-align": "left", "margin": "0px", "--hover-color": "#84706E"},
            "nav-link-selected": {"font-family": "inherit", "background-color": "#ffffff", "color": "#55ACEE", "font-size": "25px"},
        }
    )
    st.markdown("<hr style='border: 2px solid #ffffff;'>", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; font-size: 38px; color: #55ACEE; font-weight: 700; font-family: inherit;'>Your Singapore Resale Flat Price Genie</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid beige;'>", unsafe_allow_html=True)

if selected == "About":
    st.markdown("<h3 style='font-size: 30px; text-align: left; font-family: inherit; color: #FBBC05;'> Overview </h3>", unsafe_allow_html=True)
    st.markdown("""<p style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400; font-family: inherit;'>
        The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.
</p>""", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 30px; text-align: left; font-family: inherit; color: #FBBC05;'> Models </h3>", unsafe_allow_html=True)
    st.markdown("""<p style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400; font-family: inherit;'>
        Regression model: XG Boost Regressor for predicting the continuous variable 'Flat price'.
    </p>""", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 30px; text-align: left; font-family: inherit; color: #FBBC05;'> Contributing </h3>", unsafe_allow_html=True)
    github_url = "https://github.com/Santhosh-Analytics/Singapore-Resale-Flat-Prices-Predicting"
    st.markdown("""<p style='text-align: left; font-size: 18px; color: #ffffff; font-weight: 400; font-family: inherit;'>
        Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request in the <a href="{}">GitHub Repository</a>.
    </p>""".format(github_url), unsafe_allow_html=True)

if selected == "Genie":


    # Options for various dropdowns
    town_option = ['Tampines',  'Yishun',  'Jurong West',  'Bedok',  'Woodlands',  'Ang Mo Kio',  'Hougang',  'Bukit Batok',  'Choa Chu Kang',  'Bukit Merah',  'Pasir Ris',  'Sengkang',  'Toa Payoh',  'Queenstown',  'Geylang',  'Clementi',  'Bukit Panjang',  'Kallang/Whampoa',  'Jurong East',  'Serangoon',  'Bishan',  'Punggol',  'Sembawang',  'Marine Parade',  'Central Area',  'Bukit Timah',  'Lim Chu Kang']
    flat_type_option = ['4 Room',  '3 Room',  '5 Room',  'Executive',  '2 Room',  '1 Room',  'Multi Generation']
    flat_model_option =['Model A',  'Improved',  'New Generation',  'Simplified',  'Premium Apartment',  'Standard',  'Apartment',  'Maisonette',  'Model A2',  'Dbss',  'Model A-Maisonette',  'Adjoined Flat',  'Terrace',  'Multi Generation',  'Type S1',  'Type S2',  '2-Room',  'Improved-Maisonette',  'Premium Apartment Loft',  'Premium Maisonette',  '3Gen']
    lease_year_option =[year for year in range(1910, date.today().year + 1)]
    floor_option = [number for number in range(1,7)]
    floor_no_option = [number for number in range(0,52,3)]

    col1, col, col2 = st.columns([2,.5,2])

    with col1:
        town = st.selectbox('Select Town you are interested:', town_option, index=None, help="Select Town where you are looking for a property/flat", placeholder="Select Town where you are looking for a property/flat")
        flat_type = st.selectbox('Select type of Flat:', flat_type_option    , index=None, help="Select flat type you are interested.", placeholder="Select flat type you are interested.")
        flat_model = st.selectbox('Select Flat Model:', flat_model_option, index=None, help="Select flat model you like.", placeholder="Select flat model you like.")
        lease_year = st.selectbox('Select lease agreement year:', lease_year_option, index=None, help="The beginning of the lease term during which the tenant has the right to use and occupy the leased property.", placeholder="Starting point of a lease agreement.")
        

    with col2:

        floor_area = st.slider('Floor Area SQM:', min_value=20, max_value=500, value=65,step=15, help='Total Estimated space measured in square meters. Minimum value 20 sqm and maximum is 500 sqm.',)
        floor = st.selectbox('Select number of floors:', floor_option, index=None, help="Estimated number of floors.", placeholder="Estimated number of floors.")
        floor_level = st.selectbox('Select floor level: ', floor_no_option, index=None, help="Estimated range of floors.", placeholder="Estimated range of floors.")
    
        
        st.write(' ')
        st.write(' ')
        button = st.button('Predict Flat Price!')
    
    
    remaining_lease_year = lease_year + 99 - date.today().year if lease_year is not None else None
    floor_area_box = transform_single_value(floor_area, lambda_dict['floor_area_lambda'])     if floor_area is not None  else None
    town_mapping={'Lim Chu Kang': 1, 'Queenstown': 2, 'Ang Mo Kio': 3, 'Clementi': 4, 'Geylang': 5, 'Bedok': 6, 'Bukit Batok': 7, 'Yishun': 8, 'Toa Payoh': 9, 'Jurong East': 10, 'Central Area': 11, 'Jurong West': 12, 'Kallang/Whampoa': 13, 'Woodlands': 14, 'Hougang': 15, 'Serangoon': 16, 'Marine Parade': 17, 'Bukit Merah': 18, 'Bukit Panjang': 19, 'Tampines': 20, 'Choa Chu Kang': 21, 'Sembawang': 22, 'Pasir Ris': 23, 'Bishan': 24, 'Bukit Timah': 25, 'Sengkang': 26, 'Punggol': 27}
    year_mapping = {1990: 1, 1991: 2, 1992: 3, 1993: 4, 1994: 5, 1995: 6, 2002: 7, 2003: 8, 2004: 9, 2001: 10, 2005: 11, 2006: 12, 1999: 13, 2000: 14, 1998: 15, 1996: 16, 2007: 17, 1997: 18, 2008: 19, 2009: 20, 2010: 21, 2019: 22, 2015: 23, 2018: 24, 2011: 25, 2016: 26, 2017: 27, 2014: 28, 2020: 29, 2012: 30, 2013: 31, 2021: 32, 2022: 33, 2023: 34, 2024: 35}
    flat_type_mapping = {'1 Room': 1, '2 Room': 2, '3 Room': 3, '4 Room': 4, '5 Room': 5, 'Executive': 6, 'Multi Generation': 7}
    flat_model_mapping={'New Generation': 1, 'Standard': 2, 'Simplified': 3, 'Model A2': 4, '2-Room': 5, 'Model A': 6, 'Improved': 7, 'Improved-Maisonette': 8, 'Model A-Maisonette': 9, 'Premium Apartment': 10, 'Adjoined Flat': 11, 'Maisonette': 12, 'Apartment': 13, 'Terrace': 14, 'Multi Generation': 15, 'Premium Maisonette': 16, '3Gen': 17, 'Dbss': 18, 'Premium Apartment Loft': 19, 'Type S1': 20, 'Type S2': 21}
    lease_year_mapping={1969: 1, 1971: 2, 1967: 3, 1968: 4, 1973: 5, 1970: 6, 1972: 7, 1974: 8, 1977: 9, 1980: 10, 1983: 11, 1975: 12, 1981: 13, 1976: 14, 1978: 15, 1979: 16, 1966: 17, 1982: 18, 1985: 19, 1984: 20, 1986: 21, 1987: 22, 1988: 23, 1990: 24, 1989: 25, 1991: 26, 1997: 27, 1998: 28, 1996: 29, 1999: 30, 1994: 31, 1993: 32, 2000: 33, 1995: 34, 1992: 35, 2001: 36, 2002: 37, 2003: 38, 2004: 39, 2012: 40, 2014: 41, 2015: 42, 2005: 43, 2007: 44, 2010: 45, 2013: 46, 2008: 47, 2016: 48, 2009: 49, 2017: 50, 2018: 51, 2019: 52, 2006: 53, 2020: 54, 2011: 55}
    floor_mapping={1:0,2:.5,3: 1,4:1.5, 5: 2,6:2.5}
    floor_level_mapping={3: 1, 6: 2, 9: 3, 12: 4, 15: 5, 5: 6, 18: 7, 10: 8, 21: 9, 24: 10, 20: 11, 27: 12, 25: 13, 35: 14, 40: 15, 30: 16, 33: 17, 36: 18, 39: 19, 42: 20, 45: 21, 48: 22, 51: 23}
    remaining_lease_year_mapping = {81: 1, 82: 2, 83: 3, 80: 4, 79: 5, 84: 6, 78: 7, 85: 8, 77: 9, 76: 10, 86: 11, 75: 12, 87: 13, 88: 14, 48: 15, 74: 16, 89: 17, 49: 18, 90: 19, 47: 20, 72: 21, 73: 22, 71: 23, 45: 24, 46: 25, 70: 26, 91: 27, 44: 28, 43: 29, 50: 30, 69: 31, 92: 32, 93: 33, 68: 34, 41: 35, 42: 36, 51: 37, 67: 38, 96: 39, 52: 40, 58: 41, 66: 42, 94: 43, 59: 44, 95: 45, 57: 46, 100: 47, 65: 48, 101: 49, 53: 50, 56: 51, 64: 52, 54: 53, 55: 54, 98: 55, 60: 56, 63: 57, 62: 58, 61: 59, 97: 60, 99: 61}



    





    town=town_mapping[town] if town is not None else None
    year=year_mapping[date.today().year]
    flat_type=flat_type_mapping[flat_type] if flat_type is not None else None
    flat_model=flat_model_mapping[flat_model] if flat_model is not None else None
    lease_year=lease_year_mapping[lease_year] if lease_year is not None else None
    floor=floor_mapping[floor] if floor is not None else None
    floor_level=floor_level_mapping[floor_level] if floor_level is not None else None
    remaining_lease_year=remaining_lease_year_mapping[remaining_lease_year] if remaining_lease_year is not None else None
    
    data = np.array([[floor_area_box,town, year, flat_type, flat_model,lease_year,floor,floor_level,remaining_lease_year]])
    st.write(data)
    
    

    if button and data is not None:
        
        scaled_data = scale_reg.transform(data)
        st.write(scaled_data)
        prediction = xgb_Reg.predict(scaled_data)
        st.write(prediction)
        lambda_val = lambda_dict['resale_price_lambda'] 
        transformed_predict=reverse_boxcox_transform(prediction, lambda_val) if data is not None else None
        rounded_prediction = round(transformed_predict[0],2)
        st.success(f"Based on the input, the Genie's price is,  {rounded_prediction:,.2f}")
        st.info(f"On average, Genie's predictions are within approximately 3-4% of the actual market prices.")