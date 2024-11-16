import time

import streamlit as st

import sort_img_and_save
from dir_picker import st_directory_picker
import datetime


def main():
    st.set_page_config(layout="wide")
    # Designing the interface
    st.title("DeepLOKI: Automatic classify LOKI-Images")
    # st.caption('This is a string that explains something above.')
    st.write("\n")
    #container1 = st.container()

    st.subheader("Data for analysis:")
    # drag&drop
    folder_path = st_directory_picker()
    # choose folder  from explorer
    st.write(f"Selected folder_path: {folder_path}")
    st.write("\n")

    #container2 = st.container()
    st.subheader("Path to the classification folders")
    st.write("\n")

    save_folder_path = st.selectbox(
        "Select you folder path.",
        [
            "/Volumes/T7 Shield/T7/LOKI2/output",#  ./inference/sorted
        ],
    )
    time_stamp = datetime.datetime.now()
    sub_dir = f"/{str(time_stamp).replace(' ', '_')}"

    col1, col2 = st.columns(2)
    with col1:
        TR = st.slider(
                "Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01, help="Select the confidence threshold for sorting.",
            )

    st.write(f"Selected save_folder_path: {save_folder_path+sub_dir}")

    with col2:
        option = st.selectbox("Select a classifier?", ("DTL", "DINO"))
        ending = st.selectbox("Select a image format?", (".png", ".bmp"))


    if st.button("Start Sorting"):
        with st.spinner("(Pre-)Sorting images..."):
            start_time = time.time()
            print("##########folder_path:", folder_path)
            sort_img_and_save.main(
                haul_pic_path=folder_path,
                ending=ending,
                arch=option,
                target=save_folder_path + sub_dir + "_" + option,
                tr = TR,
            )
            elapsed_time = time.time() - start_time
        st.write("\n")
        st.write(f"Elapsed time: {elapsed_time:.4f} seconds")
        st.write("Sorting is finished.")


if __name__ == "__main__":
    main()
