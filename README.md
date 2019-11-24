# touchclass

This repository showcases the classfication of contact between a finger and an inert surface, such as a tabletop, in a video stream produced by a depth camera, through a supervised learning approach.

The targeted interaction can be observed in the video from the [paper](https://dl.acm.org/citation.cfm?id=3135074):

[![video](https://img.youtube.com/vi/Q8hzbU9B_k0/0.jpg)](https://www.youtube.com/watch?v=Q8hzbU9B_k0)

A series of notebooks shows how to:

- [visualise](dataset-exploration.ipynb) the dataset, 
- [extract](feature-extraction.ipynb) the features,
- [train](model-training.ipynb) a neural network.

These produce the model that was used for the experiment.


# Requirements

1. install required python package from requirements.txt:

`conda install --file requirements.txt`

2. install additional dependencies:

`brew install pcl`

`pip install python-pcl`

`conda install -c open3d-admin open3d`

2. compile cython extention

`python helper/deproject/setup.py build_ext --inplace`

3. install the dataset from the  at the root of the repository in the folder called `dataset`.


# Dataset
This repository includes the dataset on which the model was trained and tested. It is accessible in the [release](https://github.com/toinsson/touchclass/releases).


# Related Publication
If you want to learn more about potential applications, please refer to the associated paper [Gesture Typing on Virtual Tabletop](https://dl.acm.org/citation.cfm?id=3135074).

```
@inproceedings{Loriette:2017:GTV:3132272.3135074,
 author = {Loriette, Antoine and Murray-Smith, Roderick and Stein, Sebastian and Williamson, John},
 title = {Gesture Typing on Virtual Tabletop: Effect of Input Dimensions on Performance},
 booktitle = {Proceedings of the 2017 ACM International Conference on Interactive Surfaces and Spaces},
 series = {ISS '17},
 year = {2017},
 url = {http://doi.acm.org/10.1145/3132272.3135074},
}
```
