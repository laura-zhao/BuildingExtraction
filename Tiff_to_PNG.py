from osgeo import gdal
import numpy as np
import os
import subprocess

def bit16_to_Bit8(inputRaster, outputRaster, outputPixType='Byte', outputFormat='png', percentiles=[2, 98]):
    '''
    Convert 16bit image to 8bit.

    Parameters:
    inputRaster (str): Path to the input raster image.
    outputRaster (str): Path to the output raster image.
    outputPixType (str): Pixel type of the output image. Default is 'Byte'.
    outputFormat (str): Format of the output image. Default is 'png'.
    percentiles (list): Percentiles to use for rescaling. Default is [2, 98].

    Returns:
    None
    '''
    srcRaster = gdal.Open(inputRaster)
    cmd = ['gdal_translate', '-ot', outputPixType, '-of', outputFormat]

    # Iterate through bands and rescale
    for bandId in range(srcRaster.RasterCount):
        band = srcRaster.GetRasterBand(bandId + 1)
        band_arr_tmp = band.ReadAsArray()
        bmin = np.percentile(band_arr_tmp.flatten(), percentiles[0])
        bmax = np.percentile(band_arr_tmp.flatten(), percentiles[1])

        cmd.extend(['-scale_{}'.format(bandId + 1), str(bmin), str(bmax), '0', '255'])

    cmd.extend([inputRaster, outputRaster])
    print("Conversion command:", cmd)
    subprocess.call(cmd)

def convert_images(src_directory, dst_directory):
    '''
    Convert all GeoTIFF images in a directory to PNG format.

    Parameters:
    src_directory (str): Path to the source directory containing GeoTIFF images.
    dst_directory (str): Path to the destination directory to save PNG images.

    Returns:
    None
    '''
    os.makedirs(dst_directory, exist_ok=True)
    files = sorted(os.listdir(src_directory))

    for file in files:
        if file.endswith('.tif'):
            src_path = os.path.join(src_directory, file)
            dst_path = os.path.join(dst_directory, file.replace('.tif', '.png'))
            bit16_to_Bit8(src_path, dst_path)
