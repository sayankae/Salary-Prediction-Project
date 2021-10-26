import streamlit as st
import pickle
import numpy as np



def load_model():
	with open('saved_steps.pkl', 'rb') as file:
		data = pickle.load(file)

	return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]
le_age = data["le_age"]
le_gender = data["le_gender"]



def show_predict_page():
	st.title("Software Developer Salary Prediction")

	st.write("""### We need some information to predict the salary""")

	countries = (
			"United States of America",
			"India",                                                   
			"Germany"                                                  
			"United Kingdom of Great Britain and Northern Ireland",     
			"Canada",                                                  
			"France",                                                   
			"Brazil",                                                  
			"Spain",                                                    
			"Netherlands",                                             
			"Australia",                                               
			"Poland",                                                    
			"Italy",                                                    
			"Russian Federation",                                        
			"Sweden",                                                 
			"Turkey",                                                    
			"Switzerland",                                               
			"Israel",                                                    
			"Norway",                                                    
		)

	education = (
			'Master’s degree', 
			'Bachelor’s degree', 
			'Post grad'
		)

	gender = (
			'Man', 
			'Woman', 
			'Other'
		)

	age = (
			'25-34', 
			'35-44', 
			'45-54', 
			'18-24', 
			'55-64', 
			'65 years or older',
       		'Under 18', 
       		'Other'
		)

	country = st.selectbox("Country",countries)
	education = st.selectbox("Education Level",education)
	age = st.selectbox("Age Category", age)
	gender = st.selectbox("Gender",gender)
	exprience = st.slider("Years of Experience",0, 50, 5)

	ok = st.button("Predict salary")

	if ok:
		X = np.array([[country, education, exprience, age, gender ]])
		X[:, 0] = le_country.transform(X[:, 0])
		X[:, 1] = le_education.transform(X[:, 1])
		X[:, 3] = le_age.transform(X[:, 3])
		X[:, 4] = le_gender.transform(X[:, 4])
		X = X.astype(float)

		salary = regressor.predict(X)
		st.subheader(f"The estimated salary is ${salary[0]:.2f}")


