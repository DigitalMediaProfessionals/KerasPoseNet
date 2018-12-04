# KerasPoseNet
KerasPoseNet inference example

Network used is from: https://github.com/ildoonet/tf-pose-estimation
(git SHA1 ID: b119759e8a41828c633bd39b5c883bf5a56a214f, Apache 2.0 License).

To generate network inference sources and weights files use the [tool](https://github.com/DigitalMediaProfessionals/tool):
```bash
python3 <path_to_tool>/convertor.py keras_pose.ini
```

To build main application on Ubuntu:
```bash
sudo apt-get install python3-opencv python3-psutil
make
```

To build 3rd-party helper library:
```bash
cd pafprocess
sudo apt-get install swig
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

To run application using Web-camera (`sudo` is needed to set linux console to graphics mode before drawing to framebuffer):
```bash
sudo ./run.sh /dev/video0
```

To run application with input image:
```bash
sudo ./run.sh IMAGE_PATH
```

To run application with multiple images (img_01.jpg, img_02.jpg, ...):
```bash
sudo ./run.sh img_%02d.jpg
```
