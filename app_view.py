import streamlit as st
from PIL import Image, ImageOps
import os
import pandas as pd
import numpy as np

__author__ = "Raphael Kronberg Department of MMBS, MatNat Faculty," \
             " Heinrich-Heine-University"
__license__ = "MIT"
__version__ = "1.0.1"
__status__ = "Prototype: This progam/code can not be used as diagnostic tool."
__credits__ = "Pls cite and refer to when using the code: Kronberg R.M.," \
              "Applications of Supervised Deep (Transfer) Learning for Medical Image Classification"

from dir_picker import st_directory_picker

def main():
    st.set_page_config(layout="wide")
    # Designing the interface
    if 'count' not in st.session_state:
        st.session_state.count = 0
    if 'df_load' not in st.session_state:
        df_abt = pd.read_csv("output/update_allcruises_df_validated_5with_zoomie_20230727.csv",sep=";")
        st.session_state.df_load = df_abt
        st.session_state.cls_dd = np.unique(df_abt['label'])
        st.session_state.cruise_dd = np.unique(df_abt['object_cruise'])
    if "cls" not in st.session_state:
        st.session_state.cruise = st.session_state.cruise_dd[0]
        st.session_state.cls =st.session_state.cls_dd[0]
    if 'df' not in st.session_state:
        df_abt = st.session_state.df_load
        df_abt_99 = df_abt[df_abt["object_cruise"] == st.session_state.cruise]
        df_abt_99 = df_abt_99[df_abt_99['label'] == st.session_state.cls]
        df_abt_99_test = df_abt_99
        st.session_state.df = df_abt_99_test

    col1, col2, col3 = st.columns([3, 6, 3])

    with col1:
        st.write("")
        st.session_state.cruise = st.selectbox(
        'Select a cruise',
        tuple(st.session_state.cruise_dd))
        st.session_state.cls = st.selectbox(
        'Select a label',
        tuple(st.session_state.cls_dd))

        if st.button("Load data"):
            st.session_state.count = 0
            df_abt = st.session_state.df_load
            df_abt_99 = df_abt[df_abt["object_cruise"] == st.session_state.cruise]
            df_abt_99 = df_abt_99[df_abt_99['label'] == st.session_state.cls]
            df_abt_99_test = df_abt_99
            st.session_state.df = df_abt_99_test

    with col2:
        st.title("DeepLoki: Image Viewer")
        # st.caption('This is a string that explains something above.')
        st.write('\n')
        img_path = os.path.join(st.session_state.df['root_path'].values[st.session_state.count], st.session_state.df['img_file_name'].values[st.session_state.count])
        image = Image.open(img_path).convert('RGB')
        image = image.resize((800, 800))
        st.image(image, caption=f"{st.session_state.count}: {st.session_state.df['label'].values[st.session_state.count]} {st.session_state.df['object_id'].values[st.session_state.count]}")

    with col3:
        if st.button("Show next Image"):
            st.session_state.count += 1
        if st.button("Show last Image"):
            st.session_state.count -= 1
        if st.button("Reset Counter"):
            st.session_state.count = 0
        st.write("")



if __name__ == '__main__':
    main()