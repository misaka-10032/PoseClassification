# Head pose classification
This is code for my undergraduate thesis. It can be used to classify whether a person is raising or bow his head.

## Prerequisite
* [Caffe](http://caffe.berkeleyvision.org/) (with pycaffe) [v1.0-rc2](https://github.com/BVLC/caffe/archive/rc2.tar.gz)
* [Opencv](http://opencv.org/) v2.4.12 with python

## Demo
Demo is available on [youtube](https://youtu.be/med1rRB8QuA). Or you may run it after installing all prerequisites.

```
./demo.py
```

Blue box indicates a raising head; Green box indicates a bowing head; Red box indicates not being sure.

## Acknowledgements
* Thank [Köstinger](https://lrs.icg.tugraz.at/members/koestinger) very much for providing me with [AFLW](https://lrs.icg.tugraz.at/research/aflw/) dataset.
* Thank [Dr. Jin](http://homepage.fudan.edu.cn/~chengjin) very much for providing me with GTX-680.