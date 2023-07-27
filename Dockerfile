# Use the official PyTorch image as the base image
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy the contents of the current directory to the working directory in the container
COPY . /app/

# Install additional dependencies required for GPU support
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx

# Install Miniconda to manage Python environment
RUN curl -so miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh \
    && chmod +x miniconda.sh \
    && ./miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Add Conda to the system path
ENV PATH="/opt/conda/bin:${PATH}"

# Create and activate a Conda environment
RUN conda create -n py38 python=3.8
RUN echo "conda activate py38" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Install the required Python dependencies
RUN pip install -r requirements.txt

# Set an environment variable to enable GPU support for PyTorch
ENV TORCH_CUDA_ARCH_LIST="5.0 6.0 7.0 7.5 8.0 8.6+PTX"

# Start the main script
CMD ["python", "main.py", "--input_path", "[VIDEO_FILE_NAME]"]