SPGG-with-Learning-And-Propagation
====
Propagation is crucial for acquiring information, and learning involves deep information processing. Imitation dynamics is commonly used as an evolutionary dynamics in spatial public goods games, representing the propagation of strategic information in society. Learning dynamics allows agents to acquire information and self-learn through environmental interactions. In this paper, Q-learning and the Fermi update rule are used to compare differences between learning dynamics and imitation dynamics in simulation experiments.Furthermore, we combine Imitation Dynamics and Learning Dynamics, and the new dynamics integrate the advantages of both.

Requirements
----
It is worth mentioning that because python runs slowly, we use cuda library to improve the speed of code running.

`<
* Python Version 3.12.2
* CUDA Version: 12.4
* torch Version: 2.2.1
* numpy Version: 1.26.4
* pandas Version: 2.2.2
>`

Results
----
Case Ⅰ And Case Ⅱ
![https://github.com/Tychema/Learning-And-Propagation/blob/main/img/pic1.png](https://github.com/Tychema/Learning-And-Propagation/blob/main/img/pic1.png)

Case Ⅲ


![https://github.com/Tychema/Learning-And-Propagation/blob/main/img/pic6.png](https://github.com/Tychema/Learning-And-Propagation/blob/main/img/pic6.png)

Update
----
We are committed to refining the code to enhance its readability and reusability. Upon completion of this process, we will synchronize the full source code to the GitHub repository, ensuring easy access for other researchers.

Citation
=====
`<
@article{SHEN2024115377,
title = {Learning and propagation: Evolutionary dynamics in spatial public goods games through combined Q-learning and Fermi rule},
journal = {Chaos, Solitons & Fractals},
volume = {187},
pages = {115377},
year = {2024},
issn = {0960-0779},
doi = {https://doi.org/10.1016/j.chaos.2024.115377},
url = {https://www.sciencedirect.com/science/article/pii/S0960077924009299},
author = {Yong Shen and Yujie Ma and Hongwei Kang and Xingping Sun and Qingyi Chen}
}
>`
>
copyright
=====
This package is a python source code of SPGG.

Please see the following paper:

Shen, Y.; Ma, Y.; Kang, H.; Sun, X.; Chen, Q. 

Propagation and Learning: Updating Strategies in Spatial Public Goods Games through Combined Fermi Update and Q-Learning

