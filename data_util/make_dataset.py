import pandas as pd
from PIL import Image
import os
from natsort import natsorted

def create_csv(project_dir, img_dir, hddl_dir, output_path):
    imgs = natsorted(os.listdir(os.path.join(project_dir, img_dir)))
    hddls = natsorted(os.listdir(os.path.join(project_dir, hddl_dir)))

    with open(os.path.join(project_dir, output_path), 'w') as f:
        f.write("id,image,hddl\n")
    for idx, (img, hddl) in enumerate(zip(imgs, hddls)):
        if not img.endswith(('.jpg', '.jpeg', '.png')) or not hddl.endswith('.hddl'):
            continue
        img_path = os.path.join(img_dir, img)
        hddl_path = os.path.join(hddl_dir, hddl)
        # write things to csv
        with open(os.path.join(project_dir, output_path), 'a') as f:
            f.write(f"{idx},{img_path},{hddl_path}\n")

if __name__ == "__main__":
    project_dir = "/media/shuo/T7/pddl_data"
    img_dir = "images"
    hddl_dir = "hddl"
    metadata = "metadata.csv"
    output_path = os.path.join(project_dir, metadata)

    # Ensure the directories exist
    if not os.path.exists(os.path.join(project_dir, img_dir)) or not os.path.exists(os.path.join(project_dir, hddl_dir)):
        raise FileNotFoundError("Image or hddl directory does not exist.")

    # Create the CSV file
    create_csv(project_dir, img_dir, hddl_dir, output_path)
    print(f"CSV file created at {output_path}")