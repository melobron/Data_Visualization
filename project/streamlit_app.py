import streamlit as st
from PIL import Image


############################## Streamlit ##############################
st.set_page_config(
    page_title='Multipage App',
)

st.title('Analyzing the Distribution of Face Images via StyleGAN based Image Generation and Inversion')
st.sidebar.success('Select a page above')

st.header('1. Motivation: Face Image Distribution')
image_1 = Image.open('./assets/1_motivation.png')
st.image(image_1) #, caption='Sunrise by the mountains')

st.header('2. Image Generation: Generative Adversarial Network (GAN)')
image_2_1 = Image.open('./assets/2_1_ai.png')
st.image(image_2_1, caption='AI 아나운서(https://www.youtube.com/watch?v=mL1HxDTluZ4)')
image_2_2 = Image.open('./assets/2_2_gan.png')
st.image(image_2_2, caption='Generative Adversarial Network (GAN)')
image_2_3 = Image.open('./assets/2_3_face.png')
st.image(image_2_3, caption='Face Image Datasets')

st.header('3. Style Mixing: StyleGAN')
image_3_1 = Image.open('./assets/3_1.png')
st.image(image_3_1, caption='StyleGAN architecture, Input: Style vector W(18 x 512) -> Output: face image')
image_3_2 = Image.open('./assets/3_2.png')
st.image(image_3_2, caption='Style Mixing - Mix features from 2 images')

st.header('4. GANSpace')
st.subheader('What is the most dominant features in certain datasets? \n ex) age, gender, hair color')
image_4_1 = Image.open('./assets/4_1.png')
st.image(image_4_1)

st.header('5. GAN Inversion: Image2StyleGAN')
st.subheader('Can we locate certain image in the whole dataset space?\nImage Generation: W vector -> Image, GAN Inversion: Image -> W vector')
image_5_1 = Image.open('./assets/5_1.png')
st.image(image_5_1)
