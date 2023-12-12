

def convert_images(src_directory, dst_directory):
    # Convert Geotiff to PNG Images
    os.makedirs(dst_directory, exist_ok=True)
    files = sorted(os.listdir(src_directory))

    for file in files:
        if file.endswith('.tif'):
            src_path = os.path.join(src_directory, file)
            dst_path = os.path.join(dst_directory, file.replace('.tif', '.png'))
            bit16_to_Bit8(src_path, dst_path)

# Example usage
src_dir = "/content/drive/MyDrive/AOI_2_Vegas_Train/RGB-PanSharpen/"
dst_dir = "/content/drive/MyDrive/AOI_2_Vegas_Train/RGB-PanSharpenNEW/"
convert_images(src_dir, dst_dir)
