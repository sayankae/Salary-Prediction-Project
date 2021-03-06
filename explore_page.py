import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i]>=cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = "Other"
    return categorical_map

def clean_experience(x):
    if x == "More than 50 years":
        return 50
    if x == 'Less than 1 year':
        return 0.6
    return float(x)




def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Other doctoral degree (Ph.D., Ed.D., etc.)' or 'Professional degree (JD, MD, etc.)' in x:
        return 'Post grad'
    return 'Less than a bachelor'



def clean_gender(x):
    if 'Man' == x:
        return 'Man'
    if 'Woman' == x:
        return 'Woman'
    return 'Other'



def clean_age(x):
    if '25-34' in x:
        return '25-34'
    if '35-44'in x:
        return '35-44'
    if '45-54' in x:
        return '45-54'
    if '18-24' in x:
        return '18-24'
    if '55-64' in x:
        return '55-64'
    if '65 years or older' in x:
        return '65 years or older'
    if 'Under 18 years old' in x:
        return 'Under 18'
    return 'Other'


@st.cache

def load_data():
	url = 'https://drive.google.com/file/d/1SVTDQ_l0ShFgKZHE0lYAn7TwJD6_COwa/view?usp=sharing'
	path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
	df = pd.read_csv(path)
	df = df[["Country","EdLevel","YearsCodePro","Employment","Age","Gender","ConvertedCompYearly"]]
	df = df[df["ConvertedCompYearly"].notnull()]
	df = df.dropna()
	df = df[df["Employment"] == "Employed full-time"]
	df = df.drop("Employment", axis = 1)

	country_map = shorten_categories(df.Country.value_counts(), 400)
	df["Country"] = df["Country"].map(country_map)
	df = df[df["ConvertedCompYearly"] <= 250000]
	df = df[df["ConvertedCompYearly"] >= 5000]
	df = df[df["Country"] != "Other"]

	df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
	df["EdLevel"] = df["EdLevel"].apply(clean_education)
	df["Age"] = df["Age"].apply(clean_age)
	df["Gender"] = df["Gender"].apply(clean_gender)
	df = df.rename({"ConvertedCompYearly":"Salary"},axis = 1)
	return df

df = load_data()

def show_explore_page():
	st.title("Explore Software Engineer Salaries")

	st.write(
		"""
		#### Stack OverFlow Developer Salaries
		"""
		)

	
	data = df["Country"].value_counts()

	fig1, ax1 = plt.subplots()
	ax1.pie(data, labels = data.index, autopct = "%1.1f%%", shadow = False, startangle = 90)
	ax1.axis("equal")

	st.write("""#### Number of Data from different countries""")

	st.pyplot(fig1)

	st.write("""#### Mean Salary Based On Country""")

	data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending = True)
	st.bar_chart(data)


	st.write("""#### Mean Salary Based On Experience""")

	data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending = True)
	st.line_chart(data)




