# Self-Driving-Car-Simulation
Minor 1 project based on self driving car simulation with two-factor authentication using python

To run the entire project run the file:
1) Download visual studio and install all C and C++ packages 
2) pip install cmake
3) pip install dlib
4) pip install -r /path/to/requirements.txt
5) python face_authentication_v2.0.py

Dataset:
Approximately 45,500 images, 2.2GB. One of the original datasets I made in 2017. Data was recorded around Rancho Palos Verdes and San Pedro California.

Link: https://drive.google.com/file/d/1Ue4XohCOV5YXy57S_5tDfCVqzLr101M7/view

Data format is as follows: filename.jpg angle

Explanation:
This project is divided into two phases:
1) Two-factor Authnetication which includes face-recognition and entering password (face_authentication_v2.0.py)
2) Accessing the simulation if the user is verified ( run.dataset.py is called as a subprocess)

-All the detail;s of code are commented in each file-


