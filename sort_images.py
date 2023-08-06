import os
import pandas as pd
import streamlit as st
import shutil
from PIL import Image
import glob
import os




class ImageFileLister:
    def __init__(self, root_path):
        self.root_path = root_path
        self.image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
        self.data = {
            'imagename': [],
            'path_to_image': [],
            'label': []
        }

    def is_image_file(self, file_path):
        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.image_extensions

    def list_image_files(self):
        folders = []
        for foldername, _, filenames in os.walk(self.root_path):
            label = os.path.basename(foldername)
            if foldername != self.root_path:
                folders.append(foldername.split("/")[-1])
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                if self.is_image_file(file_path):
                    self.data['imagename'].append(filename)
                    self.data['path_to_image'].append(file_path)
                    self.data['label'].append(label)

        return pd.DataFrame(self.data), list(set(folders))

def get_unique_labels(root_path):
    image_lister = ImageFileLister(root_path)
    image_df, folders = image_lister.list_image_files()
    return image_df['label'].unique(), folders


def main():
    st.set_page_config(layout="wide")
    start_root = "data/5_cruises/"
    # read subfolders in a give directory based on the actual directory level
    foldernames_list = [os.path.basename(x) for x in glob.glob(f'{start_root}*')]

    # create selectbox with the foldernames
    chosen_folder = st.selectbox(label="Choose a folder", options=foldernames_list)

    root_path = start_root + chosen_folder

    # Get unique labels
    _, unique_labels = get_unique_labels(root_path)


    # Filter dropdown menu
    filter_options = [None] + sorted(unique_labels)
    col1, col2, col5, col3,  col4 = st.columns([1.5, 5, 1, 3, 1.5])
    with col3:
        st.write('\n')
        st.write('\n')
        st.write('\n')
        selected_filter = st.selectbox("Filter:", filter_options)

        # Filter the DataFrame based on the selected label
        image_lister = ImageFileLister(root_path)
        image_df, folders__ = image_lister.list_image_files()
        print(image_df)
        #st.session_state["current_image_index"] = 0

        if selected_filter is not None:
            print(selected_filter)
            print(image_df['label'].unique())
            filtered_df = image_df[image_df['label'] == selected_filter].copy()
            print(filtered_df)
            filtered_df =filtered_df.reset_index()
            if filtered_df.empty:
                st.write('No Images in this folder.')



        else:
            filtered_df = image_df.copy()
            filtered_df= filtered_df.reset_index()

        if "selected_filter" not in st.session_state:
            st.session_state.selected_filter = filter_options[0]
            
        if st.session_state.selected_filter != selected_filter:
            st.session_state.selected_filter = selected_filter
            st.session_state["current_image_index"] = 0


    with col3:
        st.write('\n')
        st.write('\n')
        st.write('\n')
        if not filtered_df.empty:
            current_image_index = st.session_state.get("current_image_index", 0)
            default_label = filtered_df['label'][current_image_index]
            selected_image_label = st.selectbox("Select Label:", unique_labels,
                                                index=unique_labels.index(default_label))
            with col2:
                st.title("DeepLoki: Labeltool")
                # st.caption('This is a string that explains something above.')
                st.image(image = Image.open(filtered_df['path_to_image'][current_image_index]).convert('RGB').resize((800, 800)), use_column_width=True,
                         caption=f"{filtered_df['imagename'][current_image_index]}: {filtered_df['label'][current_image_index]} ")

            # Default value for "Select Label" dropdown

        with col3:
            st.write('\n')
            if st.button("Move and Update Label"):
                selected_image_path = filtered_df['path_to_image'][current_image_index]
                selected_image_filename = os.path.basename(selected_image_path)

                # Move the image to the selected folder
                new_folder_path = os.path.join(root_path, selected_image_label)
                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path)
                new_image_path = os.path.join(new_folder_path, selected_image_filename)
                shutil.move(selected_image_path, new_image_path)

                # Update the DataFrame with the new label
                filtered_df.loc[current_image_index, 'label'] = selected_image_label

            # Next button to show the next image
            st.write('\n')
            if st.button("Next Image"):
                current_image_index = (current_image_index + 1) % len(filtered_df)
                st.session_state["current_image_index"] = current_image_index

            st.write(f'{st.session_state.get("current_image_index", 0)} of {len(filtered_df["imagename"])}')

            if st.button("Reset Index"):
                st.session_state["current_image_index"] = 0

            # Example with default values
            number_input_value = st.number_input("Enter a number", value=st.session_state.get("current_image_index", 0), step=1)
            st.write("You entered:", number_input_value)

            if st.button("Go to"):
                st.session_state["current_image_index"] = number_input_value


            # Export CSV button
            st.write('\n')
            if st.button("Export CSV"):
                st.write("Exporting DataFrame as CSV...")
                image_df, folders__ = image_lister.list_image_files()
                image_df.to_csv("export_csv/image_dataframe.csv", index=False)
                st.success("DataFrame exported successfully!")





if __name__ == "__main__":
    main()
