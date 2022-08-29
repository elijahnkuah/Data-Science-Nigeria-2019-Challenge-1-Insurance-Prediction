# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 20:00:38 2021

@author: Elijah_Nkuah
"""
# Core Pkgs
import streamlit as st 
st.set_page_config(page_title = "Insurance claim", initial_sidebar_state = 'auto')
#layout = 'wide',

# EDA Pkgs
import pandas as pd 
import numpy as np 
from PIL import Image


# Utils
import os
import joblib 
import hashlib
# passlib,bcrypt

# Data Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
matplotlib.use('Agg')

# DB
from database_acc import *
#from managed_db import *
# Password 
def generate_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()


def verify_hashes(password,hashed_text):
	if generate_hashes(password) == hashed_text:
		return hashed_text
	return False


feature_names_best = ['YearOfObservation', 'Insured_Period', 'Residential',
       'Building_Painted', 'Building_Fenced', 'Garden', 'Settlement',
       'Building Dimension', 'Building_Type', 'Date_of_Occupancy',
       'NumberOfWindows']
year_dict = {2013:2013, 2015:2015, 2014:2014, 2012:2012, 2016:2016}
residence_dict = {1:"Yes", 0:"No"}
bpainted_dict= {'N':1, 'V':0}
bfenced_dict = {'V':0, 'N':1}
garden_dict= {'V':1, 'O':0}
settlement_dict = {'U':0, 'R':1}
building_type_dict = {1:'Fire-resistive', 2:'Non-combustible', 3:'Ordinary', 4:'Heavy Timber'}
windows_dict = {'   .':0, '4':4, '3':3, '2':2, '5':5, '>=10':10, '6':6, '7':7, '9':9, '8':8, '1':1}


def get_yvalue(val):
    year_dict = {2013:2013, 2015:2015, 2014:2014, 2012:2012, 2016:2016}
    for key,value in year_dict.items():
        if val == key:
            return value
def get_rkey(val):
    residence_dict = {1:"Yes", 0:"No"}
    for key, value in residence_dict.items():
        if val == value:
            return key

def get_bpvalue(val):
	bpainted_dict= {'N':1, 'V':0}
	for key,value in bpainted_dict.items():
		if val == key:
			return value 

def get_bfvalue(val):
	bfenced_dict = {'V':0, 'N':1}
	for key,value in bfenced_dict.items():
		if val == key:
			return value

def get_gvalue(val):
	garden_dict= {'V':1, 'O':0}
	for key,value in garden_dict.items():
		if val == key:
			return value

def get_svalue(val):
	settlement_dict = {'U':0, 'R':1}
	for key,value in settlement_dict.items():
		if val == key:
			return value

def get_tkey(val):
    building_type_dict = {1:'Fire-resistive', 2:'Non-combustible', 3:'Ordinary', 4:'Heavy Timber'}
    for key, value in building_type_dict.items():
        if val == value:
            return key

def get_wvalue(val):
    windows_dict = {'   .':0, '4':4, '3':3, '2':2, '5':5, '>=10':10, '6':6, '7':7, '9':9, '8':8, '1':1}
    for key,value in windows_dict.items():
            if val == key:
                    return value

# Load ML Models
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


# library for Interpretation
import lime
import lime.lime_tabular


html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Predicting those who may have Bank Account </h1>
		<h5 style="color:white;text-align:center;">Insurance Claim </h5>
		</div>
		"""

# Avatar Image using a url
#avatar1 ="https://www.w3schools.com/howto/img_avatar1.png"
#avatar2 ="https://www.w3schools.com/howto/img_avatar2.png"

result_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

result_temp2 ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/{}" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

prescriptive_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Get Plenty of Rest</li>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Avoid Alchol</li>
		<li style="text-align:justify;color:black;padding:10px">Proper diet</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		<li style="text-align:justify;color:black;padding:10px">Take your interferons</li>
		<li style="text-align:justify;color:black;padding:10px">Go for checkups</li>
		<ul>
	</div>
	"""


descriptive_message_temp ="""
	<div style="overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:#FFA500;padding:10px">Insurance Claim</h3>
		<p>You have been appointed as the Lead Data Analyst to build a predictive model
 to determine if a building will have an insurance claim during a certain 
 period or not. You will have to predict the probability of having at 
 least one claim over the insured period of the building. </p>
	</div>
	"""
Steps_to_follow ="""
	<div style="overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:#FFA500;padding:10px">Follow The Steps Below:</h3>
		<ul>
		<li style="text-align:justify;color:white;padding:10px">Signup if not already having account from the sidebar</li>
		<li style="text-align:justify;color:white;padding:10px">After signing up;</li>
		<li style="text-align:justify;color:white;padding:10px">Log in with your details</li>
		<li style="text-align:justify;color:white;padding:10px">i.e Username & Password</li>
		<ul>
	</div>
	"""
@st.cache
def load_image(img):
	im =Image.open(os.path.join(img))
	return im
	
st.set_option('deprecation.showPyplotGlobalUse', False)

#def change_avatar(sex):
#	if sex == "male":
#		avatar_img = 'img_avatar.png'
#	else:
#		avatar_img = 'img_avatar2.png'
#	return avatar_img


def main():
	"""Prediction App for persons having bank account or not"""
	st.image(load_image('banner.PNG'))
	#st.markdown(html_temp.format('royalblue'),unsafe_allow_html=True)

	menu = ["Home","Login","Signup"]
	submenu = ["Plot","Prediction"]

	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.header("Home")
		st.markdown(descriptive_message_temp,unsafe_allow_html=True)
		st.markdown(Steps_to_follow, unsafe_allow_html=True)
		st.sidebar.image(load_image('LOGO.PNG'))


	elif choice == "Login":
		username = st.sidebar.text_input("Username")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			create_usertable()
			hashed_pswd = generate_hashes(password)
			result = login_user(username,verify_hashes(password,hashed_pswd))
			# if password == "12345":
			if result:
				st.success("Welcome {} to Bank Account Prediction App".format(username))
				st.image(load_image('welcome.PNG'))
				activity = st.selectbox("Activity",submenu)
				st.sidebar.image(load_image('LOGO.PNG'))
				if activity == "Plot":
					st.subheader("Data Visualisation")
					df = pd.read_csv("train_data.csv")
					st.markdown("Total dataset used is {}".format(df.shape))
					st.write(df.head())
					fig4 = plt.figure(figsize=(20, 8))
					plt.title("Years Of Observation", fontsize=20)
					sns.countplot(x = "YearOfObservation", data = df)
					st.pyplot(fig4)
					fig4a, ax = plt.subplots()
					df['YearOfObservation'].value_counts().plot(kind='pie', title="Years of observation")
					st.pyplot(fig4a)
					fig5, ax = plt.subplots()
					df['NumberOfWindows'].value_counts().plot(kind='pie', title="Number Of Windows")
					st.pyplot(fig5)
					fig, ax = plt.subplots()
					df['Claim'].value_counts().plot(kind='bar', color="#ADD8E6", title="0: no claim, 1: at least one claim over insured period")
					st.pyplot(fig)

				elif activity == "Prediction":
					st.subheader("Predictive Analytics")

					year = st.selectbox("Year of observation for the insured policy",year_dict.keys())
					period = st.number_input("Duration of insurance policy in Olusola Insurance. (Ex: Full year insurance, Policy Duration = 1; 6 months = 0.5", 0.0,1.0)
					residence = st.selectbox("Is the building a residential building or not; 1:Yes, 0:No ", tuple(residence_dict.keys()))
					build_paint = st.selectbox("Is the building painted or not (N-Painted, V-Not Painted)", tuple(bpainted_dict.keys()))
					build_fen = st.selectbox("Is the building fence or not (N-Fenced, V-Not Fenced)", tuple(bfenced_dict.keys()))
					garden = st.radio("Building has garden or not (V-has garden,  O-no garden)",tuple(garden_dict.keys()))
					settle = st.radio("Area where the building is located. (R- rural area,  U- urban area)", tuple(settlement_dict.keys()))
					dimension = st.number_input("Building Dimension - Size of the insured building in m2", 1.0, 30000.0)
					building_type = st.selectbox("The type of building; 1:'Fire-resistive', 2:'Non-combustible', 3:'Ordinary', 4:'Heavy Timber'", tuple(building_type_dict.keys()))
					date = st.number_input("Year building was first occupied",1540, 2030)
					windows = st.selectbox("Number of windows in the building", tuple(windows_dict.keys()))

					feature_list = [get_yvalue(year),period,get_rkey(residence),get_bpvalue(build_paint),get_bfvalue(build_fen),get_gvalue(garden),get_svalue(settle),dimension, get_tkey(building_type),date, get_wvalue(windows)]
					st.write("The Number of independent varaiables is {}".format(len(feature_list)))
					pretty_result = {"Year of observation":year,"Residencial":residence,"building painted":build_paint,"Building Fenced":build_fen,"Garden":garden,"Settlement":settle,"Building Dimension":dimension,"Building Type":building_type,"Year building was first occupied":date,"Number of Windows":windows}
					st.json(pretty_result)
					single_sample = np.array(feature_list).reshape(1,-1)

					# ML
					model_choice = st.selectbox("Select Model",["Lightgbm","Catboost","Xgboost"])
					if st.button("Predict"):
						if model_choice == "Lightgbm":
							loaded_model = load_model('lgb_model_2.pkl')
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
						elif model_choice == "Catboost":
							loaded_model = load_model('cat_model_2.pkl')
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
						else:
							loaded_model = load_model('xgb_model_2.pkl')
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)

						
						if prediction == 1:
							st.success("Claiming Insurance")
							pred_probability_score = {"Probability of not claiming insurance":pred_prob[0][0]*100,"Probability of Claiming Insurance":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
							st.subheader("Prescriptive Analytics")
							#st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
						elif prediction == 2:
							st.success("Claiming Insurance")
							pred_probability_score = {"Probability of not claiming insurance":pred_prob[0][0]*100,"Probability of Claiming Insurance":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
							st.subheader("Prescriptive Analytics")
							#st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
						elif prediction == 3:
							st.success("Claiming Insurancer")
							pred_probability_score = {"Probability of not claiming insurance":pred_prob[0][0]*100,"Probability of Claiming Insurance":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
							st.subheader("Prescriptive Analytics")
							#st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
						else:
							st.warning("Not Claiming Insurance")
							pred_probability_score = {"Probability of not claiming insurance":pred_prob[0][0]*100,"Probability of Claiming Insurance":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)	


			else:
				st.warning("Incorrect Username/Password")

    
	elif choice == "Signup":
		st.sidebar.image(load_image('LOGO.PNG'))
		new_username = st.text_input("User Name")
		new_password = st.text_input("Password", type='password')
		confirmed_password = st.text_input("Confirm Password", type='password')
		if new_password == confirmed_password:
			st.success("Password Confirmed")
		else:
			st.warning("Passwords not the same")
		if st.button("Submit"):
			create_usertable()
			hashed_new_password = generate_hashes(new_password)
			add_userdata(new_username, hashed_new_password)
			st.success("You have successfully created a new account")
			st.info("Login To Get Started")


if __name__ == '__main__':
	main()
