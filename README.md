<p align="center">
    <img src="./assets/CBAS_logo.png" alt="CBAS Logo" style="width: 800px; height: auto;">
</p>

# CBAS v3 (BETA)

> [!WARNING]
> **This is the development branch for the upcoming CBAS v3.** This version is a complete architectural and functional overhaul and should be considered **experimental**. It may contain bugs or undergo significant changes.

> For the current stable and recommended version, please see the [**`v2-stable` branch**](https://github.com/jones-lab-tamu/CBAS/tree/v2-stable).

---

CBAS (Circadian Behavioral Analysis Suite) is a suite of tools for phenotyping complex animal behaviors. It is designed to automate classification of behaviors from video data and provide a simple interface for visualizing and analyzing the results.

*Originally created by Logan Perry and now maintained by the Jones Lab at Texas A&M University.*

CBAS was designed with circadian behavior monitoring in mind. Here's a visualization of what CBAS can do! Note that the behaviors are not set in stone and can be easily changed to fit the user's needs.
<p align="center">
    <img src=".//assets/realtime.gif" alt="CBAS actograms" style="width: 600px; height: auto;">
</p>
<p align="center"> 

###### *(Left) Real-time video recording of an individual mouse over two weeks in a 12h:12h light:dark cycle and in constant darkness. (Right) real-time actogram generation of nine distinct home cage behaviors.* 

</p>

## Major New Features in v3

*   **Advanced Labeling Workflow:** A new "Review & Correct" mode allows you to pre-label videos with a model and quickly correct its predictions, dramatically speeding up the labeling process.
*   **Instance-Based Navigation:** Jump directly between behavioral events using `Tab` and `Shift+Tab` instead of scrubbing through frames manually.
*   **Redesigned User Interface:** A cleaner, more modern UI across all pages for improved usability.
*   **Robust Backend:** The application's backend has been re-engineered with dedicated worker threads for stability and performance.

## Background

CBAS is a user-friendly, GUI-enabled Python package that allows for the automated acquisition, classification, and visualization of behaviors over time. It consists of three modules:

- ### Module 1: Acquisition

<p align="center">
    <img src=".//assets/acquisition_1.png" alt="CBAS Diagram" style="width: 500px; height: auto;">
</p>
<p align="center"> 

The acquisition module is capable of batch processing streaming video data from any number of network-configured real-time streaming protocol (RTSP) IP cameras.

- ### Module 2: Classification and visualization

<p align="center">
    <img src=".//assets/classification_1.png" alt="CBAS Diagram" style="width: 500px; height: auto;">
</p>
<p align="center"> 

CBAS performs classification using a frozen feature extractor "backbone," the [state-of-the-art DINOv2 vision transformer](https://arxiv.org/abs/2304.07193). We have added a joint long short-term memory and linear layer classification model 'head' onto this backbone to classify behaviors of interest.

<p align="center">
    <img src=".//assets/clocklab.png" alt="CBAS Diagram" style="width: 300px; height: auto;">
</p>
<p align="center"> 

The classification and visualization module enables inference on recorded video and displays acquired behavior time series data as actograms that can be readily exported for offline analysis.



- ### Module 3: Training (optional)

<p align="center">
    <img src=".//assets/training_1.png" alt="CBAS Diagram" style="width: 750px; height: auto;">
</p>
<p align="center"> 


The training module allows the user to create balanced training sets of behaviors of interest, train custom model heads, and validate model performance on a naive test set of behavior instances.

-------

## Installation

We have tested the installation instructions to be as straightforward and user-friendly as possible, even for users with limited programming experience.

[Click here for step-by-step instructions on how to install CBAS from source (for Windows users.)](Installation.md)

------

## Setting up cameras

CBAS will work with any real-time streaming protocol (RTSP) IP camera, but we have provided instructions on how to install power-over-ethernet (PoE) cameras and network switches to allow for the parallel, indefinite recording of up to 24 cameras per network switch.

[Click here for step-by-step instructions on how to set up RTSP IP cameras to automatically record video.](Cameras.md)

-------

## Out-of-the-box recording

CBAS comes packaged with the Jones Lab's DINOv2+ joint long short-term memory (LSTM) and linear layer model head to allow users to immediately begin classification of our nine behaviors of interest (eating, drinking, rearing, climbing, grooming, exploring, nesting, digging, and resting). If you want to record these behaviors, you will need to first replicate our recording setup using [these instructions and parts list](Recording_setup.md).

------------------

## Creating a dataset and training a classification model

Since the DINOv2 visual backbone remains static in our training model, users can quickly and easily adapt CBAS for various classification tasks, animal species, and video environments. The training module enables users to create balanced training sets for specific behaviors, train joint LSTM and linear layer models, and validate model performance on a naive test set of behavior instances.

[To learn how to create a dataset and train a classification model, click here](Training.md).

--------------
## Hardware requirements

While not required, we **strongly** recommend using a NVIDIA GPU with high VRAM to allow for GPU optimization with CUDA.

Our PC specs are:
- CPU: 12-core AMD Ryzen 9 5900X (more recent installations use a 7900X)
- RAM: 32 GB DDR5
- SSD: 1TB+ NVMe SSD
- GPU: NVIDIA GeForce RTX 3090 24GB (more recent installations use a 4090)

-----

## Feedback

As this is a beta, feedback and bug reports are highly encouraged! Please open an [Issue](https://github.com/jones-lab-tamu/CBAS/issues) to report any problems you find.

-----

###### MIT License

###### Copyright (c) 2025 Jones Lab, Texas A&M University

###### Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

###### The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

###### THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
