import os

def convert_images_to_png(folder_path):
    os.system("chmod +x convert_bmp2png.sh")
    bash_script_path = "/Users/raphaelkronberg/PycharmProjects/deeploki/convert_bmp2png.sh"  # Replace with the actual path to the Bash script

    # Call the Bash script using os.system()
    os.system(f"{bash_script_path} '{folder_path}'")

if __name__ == "__main__":
    # Provide the path to the main folder containing BMP images
    main_folder_path = "data/0010_PS121-010-03/Haul 9/LOKI_10001.02/Pictures Kopie 3"
    convert_images_to_png(main_folder_path)