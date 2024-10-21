# SwinUnet_hls_Cloud

The code is for the work titled 'A robust global Swin-Unet Sentinel-2 surface reflectance cloud and shadow detection algorithm solution for the NASA Harmonized Landsat Sentinel-2 (HLS) data.' It is an application of Swin-Unet for cloud and cloud shadow detection. Our paper has been submitted to the journal Science of Remote Sensing.

1. Download the pre-trained model and test examples
   https://github.com/Access-Planet-DL/SwinUnet_HLS_CLOUD.git
  
2. Unzip the pretrained model
     trained_model.zip 
3. Run the classification on the test data, modify the "inputdir" and "model_path" to the folders where the test data and the unzipped trained model are saved.
     python hls_swin_cloud_shadow.py

It will replicate the examples shown in Figure 8 of the paper.

More test data can be found at https://zenodo.org/records/13910150.




     
     
   
