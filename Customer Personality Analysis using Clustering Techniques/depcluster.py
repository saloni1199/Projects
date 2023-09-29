import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler




# Load the trained model from the pickle file
model_file = 'kmeans_model.pkl'
with open(model_file, 'rb') as f:
    kmeans = pickle.load(f)

# Create the Streamlit app
def main():
    st.title("K-means Clustering App")
    st.write("Cluster Analysis for Marketing Dataset")

    # Input fields
    Age = st.slider("Customer Age", min_value=0, max_value=100, step=1)
    Partners = st.checkbox("Partners")
    Single = st.checkbox("Single")
    Children = st.checkbox("Children")
    Basic = st.checkbox("Basic")
    Graduation = st.checkbox("Graduation")
    Advance_Education = st.checkbox("Advance Education")
    Income = st.number_input("Income")
    TotalSpendings = st.number_input("Total Spendings")
    PlaceSpendings = st.number_input("Place Spendings")
    WebEnrollment = st.number_input("Web Enrollment")
    MonthEnrollment = st.number_input("Month Enrollment")
    Accepting_Cmp = st.checkbox("Accepted Campaign")
    Recency = st.number_input("Recency")
    Complain = st.checkbox("Complain")
    Response = st.checkbox("Response")


    # Preprocess the input data
    input_data = pd.DataFrame({
        'Age': [Age],
        'Partners': [int(Partners)],
        'Single': [int(Single)],
        'Children': [Children],
        'Basic': [int(Basic)],
        'Graduation': [int(Graduation)],
        'Advance Education': [int(Advance_Education)],
        'Income': [Income],
        'TotalSpendings': [TotalSpendings],
        'PlaceSpendings': [PlaceSpendings],
        'WebEnrollment': [int(WebEnrollment)],
        'MonthEnrollment': [MonthEnrollment],
        'Accepting_Cmps': [Accepting_Cmp],
        'Recency': [Recency],
        'Complain': [int(Complain)],
        'Response': [int(Response)]
})
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(input_data)

    # Perform clustering on the input data
    cluster_label = kmeans.predict(scaled_input)

    st.write("Input Data:")
    st.write(input_data)

    st.write("Cluster Label:")
    st.write(cluster_label)

if __name__ == "__main__":
    main()

