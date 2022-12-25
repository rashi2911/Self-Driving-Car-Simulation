# Self-Driving-Car-Simulation with Face-Authentication
Minor 1 project based on self driving car simulation with two-factor authentication using python.
This project is made with reference to TensorFlow implementation of Nvidia paper with some changes. 

To run the entire project run the file:
>1) Download visual studio and install all C and C++ packages 
>2) pip install cmake
>3) pip install dlib
>4) pip install -r /path/to/requirements.txt
>5) python two-factor-authentication.py

Dataset:
Approximately 45,500 images, 2.2GB. One of the original datasets I made in 2017. Data was recorded around Rancho Palos Verdes and San Pedro California.

[Dataset Link](https://drive.google.com/file/d/1Ue4XohCOV5YXy57S_5tDfCVqzLr101M7/view)

Data format is as follows: filename.jpg angle

Explanation:
This project is divided into two phases:
1) Two-factor Authnetication which includes face-recognition and entering password (two-factor-authentication.py)
2) Accessing the simulation if the user is verified ( run.dataset.py is called as a subprocess)

*All the details of code are commented in each file*

Acknowledgement:
1.	Mogaveera, A., Giri, R., Mahadik, M., & Patil, A. (2018, August). Self-driving robot using neural network. In 2018 International Conference on Information, Communication, Engineering and Technology (ICICET) (pp. 1-6). IEEE.
2.	Pomerleau, D. (1990). Rapidly adapting artificial neural networks for autonomous navigation. Advances in neural information processing systems, 3.
3.	Okuyama, T., Gonsalves, T., & Upadhay, J. (2018, March). Autonomous driving system based on deep q learning. In 2018 International conference on intelligent autonomous systems (ICoIAS) (pp. 201-205). IEEE.
4.	Swaminathan, V., Arora, S., Bansal, R., & Rajalakshmi, R. (2019, February). Autonomous driving system with road sign recognition using convolutional neural networks. In 2019 International Conference on Computational Intelligence in Data Science (ICCIDS) (pp. 1-4). IEEE
5.	L. Yuan, Z. Qu, Y. Zhao, H. Zhang and Q. Nian, "A convolutional neural network based on TensorFlow for face recognition," 2017 IEEE 2nd Advanced Information Technology, Electronic and Automation Control Conference (IAEAC), 2017, pp. 525-529, doi: 10.1109/IAEAC.2017.8054070.
6.	Z. Zhang, "Improved Adam Optimizer for Deep Neural Networks," 2018 IEEE/ACM 26th International Symposium on Quality of Service (IWQoS), 2018, pp. 1-2, doi: 10.1109/IWQoS.2018.8624183.
7.	N. P. Ramaiah, E. P. Ijjina and C. K. Mohan, "Illumination invariant face recognition using convolutional neural networks," 2015 IEEE International Conference on Signal Processing, Informatics, Communication and Energy Systems (SPICES), 2015, pp. 1-4, doi: 10.1109/SPICES.2015.7091490.




