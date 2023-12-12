def main(src_dir, dst_dir, geojson_directory, output_directory):
    # Convert GeoTiff Images to PNG Images
    src_dir = "/content/drive/MyDrive/AOI_2_Vegas_Train/RGB-PanSharpen/"
    dst_dir = "/content/drive/MyDrive/AOI_2_Vegas_Train/RGB-PanSharpenNEW/"
    convert_images(src_dir, dst_dir)
    
    #  Extract labels from Geojson files
    tif_directory = src_dir
    geojson_directory = '/content/drive/MyDrive/AOI_2_Vegas_Train/geojson/buildings'
    output_directory = '/content/drive/MyDrive/AOI_2_Vegas_Train/Output Image Directory/'
    process_geojson(tif_directory, geojson_directory, output_directory, limit=100)

