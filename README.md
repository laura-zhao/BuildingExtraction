# SpaceNet Building Detection with Mask R-CNN

## Overview
This repository contains a complete workflow for building detection using the Mask R-CNN deep learning model on the SpaceNet dataset. It includes scripts for preprocessing SpaceNet GeoTIFF images, extracting building labels from GeoJSON files, and training the Mask R-CNN model.

## Contents
1. **GeoTIFF to PNG Conversion**: Scripts to convert GeoTIFF images from the SpaceNet dataset to PNG format for easier processing.
2. **Label Extraction**: Code to extract building labels from GeoJSON files and overlay them on corresponding satellite images.
3. **Mask R-CNN Training**: Implementation of the Mask R-CNN model for building detection, including dataset preparation, model configuration, and training routines.

## Prerequisites
- Python 3.x
- Libraries: `numpy`, `matplotlib`, `PIL`, `geopandas`, `rasterio`, `json`, `geoio`, `mrcnn`
- SpaceNet dataset
- Mask R-CNN library implemented by Waleed Abdulla https://github.com/matterport/Mask_RCNN
