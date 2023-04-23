import streamlit as st
from PIL import Image


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
#opening the image
image = Image.open('Faults.png')

#displaying the image on streamlit app
st.image(image)

