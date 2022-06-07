# Vehicle Speed Estimation with Optical Flow and Linear Regression

Shawn's project (kangzx@stanford.edu)

## This Repo
We used pytorch to build and train the model.

`models.py` contains the regression models

`dataset.py` contains the dataset classes for the use of `torch.utils.data.Dataset`

`utils.py` contains utility functions (from other repos)

### Environment Requirements:
pytorch, numpy, opencv, PIL, h5py, matplotlib, cudatoolkit, scipy, and some other necessary packages

### Dataset
Download KITTI 2015 dataset from http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow

Download comma.ai driving dataset from https://research.comma.ai/

(Discarded) A small dataset we used from comma.ai: https://github.com/commaai/calib_challenge

### Fintuning RAFT and Generate Flows and Labels
Use `python fintune_raft.py` to fine-tune RAFT and save the model

Run the cells following the instructions in `process_data.ipynb` to generate flows and labels

(Discarded) Use `python generate_flow.py` and specify datapath, sample rate, and model path (.pt file) to generate .flo files for the small data

### Training the Regression Model
Run the cells and following the instructions in `train_script_new.ipynb` to train the model and save the results

(Discarded) Modify the paths and run `python train_regression.py` to train the model and save the results


### Visualization
`visualization.ipynb` contains cells that visualize the results for the report and poster

`viz_video.ipynb` contains cells that we used to first observe the dataset and try RAFT model
