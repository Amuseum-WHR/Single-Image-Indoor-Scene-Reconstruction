# Single Image Indoor Scene Reconstruction
## Requirements
#### 运行demo的requirements：
+ vtk
+ jellyfish
+ seaborn
+ pymesh (unnecessary if you use vtk)
#### 运行train的requirements：
+ python 3.7
+ cuda 11.3
+ pytorch 1.12.1
+ torchvision 0.13.1
+ pytorch3d
+ tensorboard
+ chainercv
+ cupy

## Demo
+ 运行demo.py的主要选项

    + --mode: 我们有四种不同的demo模式: normal, replace, add, exchange
    + --read_data: 我们有三种不同的读取数据的方式: 2d-detection, json, dataset. dataset读入的数据只支持normal模式。
    + --k: 我们有三种不同的读取相机内参K的方式: default, ours, txt.
    + --demo_path: 保存demo结果的文件夹 

+ 在normal模式从SunRGB-D数据集抽取图片进行3D重建：

    ```bash
    python demo.py --mode normal --read_data dataset  --k default
    ```

+ 对无2Dbdb信息的图片根据保存好的相机内参K进行replace模式下的3D重建，将所有chair类换成sofa类：

    ```bash
    python demo.py --mode replace --read_data 2d-detection --img_path demo/img.jpg --k txt --k_path demo/K.txt --src_class chair --target_class sofa
    ```

+ 对存好2Dbdb信息的图片img1，根据存在demo.py中的相机内参K进行add模式下的3D重建，将另一张图片img2的物体放入img1划定的方框中：

    ```bash
    python demo.py --mode add --read_data json --img_path demo/img1.jpg --add_img demo/img2.jpg --k ours --add_box [38,304,245,496]
    ```

+ 对无2Dbdb信息的图片根据保存好的相机内参K进行exchange模式下的3D重建，将一个chair类与sofa类互换位置：

    ```bash
    python demo.py --mode replace --read_data 2d-detection --img_path demo/img.jpg --k txt --k_path demo/K.txt --src_class chair --target_class sofa
    ```

+ 以上部分示例展示如何运行demo.py，更多组合可以根据实际调整。

## Demo_pymesh
+ demo_pymesh的接口与demo相同，只是不会采用vtk作为可视化工具，而是使用pymesh将三维重建结果保存为.ply文件。

## Train
+ 进行联合训练：
    ```bash
    python train.py --lr 1e-4
    ```
+ 详情见于train.py文件。LEN, ODN, MGN各自的训练代码分别在train_for_len.py, train_for_odn.py, train_for_mgn.py中

