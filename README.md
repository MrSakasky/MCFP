MCFP
=======
# Environments
* Ubuntu18.04
* CUDA 11.3
* cuDNN8.5
* Pytorch 1.12.1
* Python 3.9
# Experiments
## Visulisation(KITTI)
```Python
python kitti/prepare_data.py --demo  # for single sample
python kitti/kitti_object.py  # for gt boxes and prediction boxes
```
## Prepare data (args need to be set according to the actual situation)
### data
frustum_pointnets_pytorch  
├── dataset  
│   ├── KITTI  
│   │   ├── ImageSets  
│   │   ├── object  
│   │   │   ├──training  
│   │   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes)  
│   │   │   ├──testing  
│   │   │      ├──calib & velodyne & image_2  
├── kitti  
│   │   ├── image_sets  
│   │   ├── rgb_detections  
├── train  

```Python
python kitti/prepare_data.py --gen_train --gen_val --gen_val_rgb_detection --car_only
```
## Train
```Python
CUDA_VISIBLE_DEVICES=0 python train/train_fpointnets.py --log_dir log
python train/train_fpointnets.py --name three  # for 2D ground truth
```
## Test
```Python
CUDA_VISIBLE_DEVICES=0 python train/test_fpointnets.py --model_path <log/.../xx.pth> --output test_results
train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ test_results

python train/test_fpointnets.py --output output/default --model_path <log/.../xx.pth>   # for 2D ground truth
train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ output/default/

python train/test_fpointnets.py --output output/gt2rgb --model_path <log/.../xx.pth> --data_path kitti/frustum_caronly_val_rgb_detection.pickle --from_rgb_det  # for rgb detection results
train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ output/gt2rgb/
```
