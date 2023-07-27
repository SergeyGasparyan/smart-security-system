# Real-Time Object Detection and Multi-Object Tracking for Smart Security Systems
![Demo](assets/result.gif)
Reference: youtube [link](https://www.youtube.com/watch?v=ZBIX2cHvizA&ab_channel=XAmbience) 

## Project Overview
This project demonstrates the integration of YOLOv5, a state-of-the-art object detection model, and DeepSORT (Deep Learning for Generic Object Tracking), a real-time object tracking algorithm that incorporates Kalman filtering. The combination of these two powerful techniques allows for robust people detection and tracking in both images and videos.

## Installation
1 Clone this repository to your local machine:
```bash
git clone https://github.com/SergeyGasparyan/smart-security-system
cd smart-security-system
```

2 Create a virtual environment with Python >=3.8
```bash
conda create -n py38 python=3.8    
conda activate py38   
```

3 Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Run
```bash
# on video file
python main.py --input_path [VIDEO_FILE_NAME]

# on webcam 
python main.py --cam 0 --display
```

## Running with Docker
1 Make sure you have Docker installed on your machine and Nvidia GPU drivers set up.

2 Build the Docker image from the project directory:
```bash
docker build -t smart-security-system-gpu .
```

3 Run the project inside the Docker container with GPU support:
```bash
docker run --gpus all --rm -it -v /path/to/your/video:/app/[VIDEO_FILE_NAME] smart-security-system-gpu
```
This will launch the object detection and multi-object tracking application with GPU support in the Docker container.
