import streamlit as st
import color_public_hpc

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
    st.title("DeepLoki: Automatic classify Loki-Images")
    # st.caption('This is a string that explains something above.')
    st.write('\n')
    container1 = st.container()

    container1.subheader("Data for analysis:")
    folder_path = st_directory_picker()

    container1.write(f'Selected folder_path: {folder_path}')
    st.write('\n')

    container2 = st.container()
    container2.subheader("Path to the classification folders")
    st.write('\n')


    save_folder_path =container2.selectbox(
    'Select you folder path.',
    ['./sorted_img/',])

    container2.write(f'Selected save_folder_path: {save_folder_path}')


    if container2.button("Start Sorting"):
        with st.spinner('(Pre-)Sorting images...'):
            color_public_hpc.main(folder_path=folder_path, save_class_patches_path=save_folder_path)
        st.write('\n')
        st.write('Sorting is finished.')



if __name__ == '__main__':
    main()