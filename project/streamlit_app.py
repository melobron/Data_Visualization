import argparse
import streamlit as st

from image_generation import image_generation


# Main
if __name__ == "__main__":
    # ############################## Arguments ##############################
    # parser = argparse.ArgumentParser(description='Test StyleGAN')
    #
    # parser.add_argument('--gpu_num', default=0, type=int)
    # parser.add_argument('--seed', default=100, type=int)
    # parser.add_argument('--model_path', default='./pre-trained/Dog(FreezeD).pth', type=str)
    # parser.add_argument('--dataset_name', default='Dog', type=str)  # FFHQ, Dog
    # parser.add_argument('--img_size', default=256, type=int)  # Pre-trained model suited for 256
    #
    # # Mean Style
    # parser.add_argument('--style_mean_num', default=10, type=int)  # Style mean calculation for Truncation trick
    # parser.add_argument('--alpha', default=1, type=float)  # Fix=1: No progressive growing
    # parser.add_argument('--style_weight', default=0.7, type=float)  # 0: Mean of FFHQ, 1: Independent
    #
    # # Sample Generation
    # parser.add_argument('--n_row', default=3, type=int)  # For Visualization
    # parser.add_argument('--n_col', default=5, type=int)  # For Visualization
    #
    # # Style Mixing
    # parser.add_argument('--n_source', default=5, type=int)  # cols
    # parser.add_argument('--n_target', default=3, type=int)  # rows
    #
    # # Transformations
    # parser.add_argument('--normalize', type=bool, default=True)
    # parser.add_argument('--mean', type=tuple, default=(0.5, 0.5, 0.5))
    # parser.add_argument('--std', type=tuple, default=(0.5, 0.5, 0.5))
    #
    # opt = parser.parse_args()


    ############################## Streamlit ##############################
    st.set_page_config(
        page_title='Team 5',
    )

    st.title('Main Page')
    st.sidebar.success('Select a page above')






