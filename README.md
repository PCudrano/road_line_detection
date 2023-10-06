  # Road Line Detection

This pipeline extracts road line markings from the image plane and projects them in the world coordinate frame exploiting the camera calibration.
It then allows to accumulate the detections throughout an entire acquisition, in order to obtain a raw point map.

## Usage


### Frame-by-frame processing

Process data stream frame-by-frame and visualize all intermediate results: image-plane features, extracted lines, and projected lines in the BEV plane.

`python main_plot_frame_by_frame.py`

![Frame-by-frame processing](https://github.com/PCudrano/road_line_detection/assets/20643285/02577ef5-05e1-4b1b-af7e-dec5bd9fa808)

### Line points accumulation

Process a data stream frame-by-frame and store each line detection in a CSV file. <br>
This CSV can then be post-processed visualize the line markings detected over the entire data stream, together with the vehicle trajectory.

1. `python main_store_frames_to_csv.py` --> process the data stream and store the detections in a CSV file <br>
2. `python main_plot_all_from_csv.py` --> load the CSV file and visualize the accumulated point map

![Line points accumulation](https://github.com/PCudrano/road_line_detection/assets/20643285/e19ea2be-e372-421b-9a61-e6150c18ad64)

### Repo structure

```
.
├── configs.py
├── data/              # For the dataset specifications, see below
│   ├── calibration/
│   ├── gt/
│   ├── imgs/
│   └── odom/
├── outputs/           # Placeholder for output folder
├── environment.yml    # Conda environment
├── features_cnn/      # CNN used to extract the line markings from the image plane (Simone Mentasti © 2020)
├── includes
│   ├── Camera.py                      # Class for handling the camera calibration
│   ├── Bev.py                         # Given a camera calibration, produces and manipulates Bird's Eye View (BEV)
│   ├── input_manager.py               # Management of input data stream (images/video), ground truth and vehicle odometry
│   ├── output_manager.py              # Management of visualization and storage of video results
│   ├── procedural/                    # Main stages of the extraction pipeline:
│   │   ├── feature_extraction.py        # Line detection in the image plane through CNN and thresholding
│   │   ├── feature_point_selection.py   # Line points extraction through Window-based Line Following (WLF) algorithm
│   │   └── fitting.py                   # Simple line fitting algorithms
|   | # Private utils
│   ├── Line.py
│   ├── MovingWindowLineFollower.py
│   ├── geo.py
│   ├── manage_gt.py
│   ├── plot_utils.py
│   └── utils.py
|
| # Main scripts
├── main_plot_all_from_csv.py
├── main_plot_frame_by_frame.py
└── main_store_frames_to_csv.py

```

All main scripts can be configured through flags at the beginning of the scripts and a global config file `configs.py`, which holds information on the dataset and on the parameters needed by the pipeline.

## Cite us

If you use this work, please cite us!

### Code

Our [frame-by-frame line detection pipeline](#frame-by-frame-processing) was first used in this work:

***Advances in centerline estimation for autonomous lateral control*** [[IEEEXplore](https://ieeexplore.ieee.org/abstract/document/9304729/)] [[arXiv](https://arxiv.org/abs/2002.12685)]<br>
_Paolo Cudrano, Simone Mentasti, Matteo Matteucci, Mattia Bersani, Stefano Arrigoni, and Federico Cheli_<br>
_Presented at 2020 IEEE Intelligent Vehicles Symposium (IV), Oct 2020_

```
@inproceedings{9304729,
  author = {Cudrano, Paolo and Mentasti, Simone and Matteucci, Matteo and Bersani, Mattia and Arrigoni, Stefano and Cheli, Federico},
  booktitle = {2020 IEEE Intelligent Vehicles Symposium&nbsp;(IV)},
  title = {Advances in centerline estimation for autonomous lateral control},
  year = {2020},
  volume = {},
  number = {},
  pages = {1415-1422},
  keywords = {},
  doi = {10.1109/IV47402.2020.9304729},
  issn = {2642-7214},
  month = oct
}
```

Our [line point accumulation pipeline](#line-points-accumulation) was used as a pre-processing step for this work:

***Clothoid-Based Lane-Level High-Definition Maps: Unifying Sensing and Control Models*** [[IEEEXplore](https://ieeexplore.ieee.org/abstract/document/9935693)]<br>
_Paolo Cudrano, Barbara Gallazzi, Matteo Frosi, Simone Mentasti, and Matteo Matteucci_<br>
_In IEEE Vehicular Technology Magazine, Dec 2022_

```
@article{9935693,
  author = {Cudrano, Paolo and Gallazzi, Barbara and Frosi, Matteo and Mentasti, Simone and Matteucci, Matteo},
  journal = {IEEE Vehicular Technology Magazine},
  title = {Clothoid-Based Lane-Level High-Definition Maps: Unifying Sensing and Control Models},
  year = {2022},
  volume = {17},
  number = {4},
  pages = {47-56},
  keywords = {},
  doi = {10.1109/MVT.2022.3209503},
  issn = {1556-6080},
  month = dec
}
```

### Dataset

The dataset used in our work and for this demo is available at this [link](http://polimi365-my.sharepoint.com/:f:/g/personal/10104160_polimi_it/Em5LCwFs8L5OlCcY7Udx9vMBjMn08c66b8uj-JnSjygoCA?e=vVx7Gf). <br>
Please reach out for more info!

***Beyond Image-Plane-Level: A Dataset for Validating End-to-End Line Detection Algorithms for Autonomous Vehicles***<br>
_Simone Mentasti,  Paolo Cudrano, Stefano Arrigoni, Matteo Matteucci, and Federico Cheli_<br>
_Presented at 2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC) Workshop on Building Reliable Datasets for Autonomous Vehicles_


```
@INPROCEEDINGS{mentasti2023beyond,
  author={Mentasti, Simone and Cudrano, Paolo and Arrigoni, Stefano and Matteucci, Matteo and Cheli, Federico},
  booktitle={2023 IEEE 26th International Conference on Intelligent Transportation Systems&nbsp;(ITSC) Workshop on Building Reliable Datasets for Autonomous Vehicles}, 
  title={Beyond Image-Plane-Level: A Dataset for Validating End-to-End Line Detection Algorithms for Autonomous Vehicles}, 
  year={2023},
  pages={1-6},
  month={Sep}
}
```
