# SwinUnet_hls_Cloud

The code is for the work titled 'A global Swin-Unet Sentinel-2 surface reflectance-based cloud and cloud shadow detection algorithm for the NASA Harmonized Landsat Sentinel-2 (HLS) dataset' It is an application of Swin-Unet for cloud and cloud shadow detection. Our paper has been submitted to the journal Science of Remote Sensing.


# Environment
Please prepare an environment with TensorFlow 2.7 and Python 3.7.16. 


# steps

1. Download the pre-trained model and test examples
   ```bash
   git hub https://github.com/Access-Planet-DL/SwinUnet_HLS_CLOUD.git
   ```
  
2. Unzip the pretrained model
   ```bash
   unzip trained_model.zip
   ```
3. Run classification on the test data, modify the "inputdir" and "model_path" to the folders where the test data and the unzipped trained model are saved.
   ```bash
   python hls_swin_cloud_shadow.py
   ```

It will replicate the examples shown in Figure 8 of the paper.

More test data can be found at https://zenodo.org/records/13910150.


# Citation

If you use this work, please cite it as:

```bibtex
@unpublished{swinunet_hls_cloud_shadow_2024,
author = {Haiyan Huang and David P. Roy and Hugo De Lemos and Yuean Qiu and Hankui K. Zhang},
title = {A global Swin-Unet Sentinel-2 surface reflectance-based cloud and cloud shadow detection algorithm for the NASA Harmonized Landsat Sentinel-2 (HLS) dataset},
note = {Manuscript submitted for publication},
year = {2024},
journal = {Science of Remote Sensing}
}


     
   
