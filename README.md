# KAUST_CS340_GraphProject
Graph Mining project files for KAUST [CS340 Computational Methods in Data Mining](https://academicaffairs.kaust.edu.sa/Courses/Pages/DownloadSyllabus.aspx?Year=2020&Semester=020&Course=00007210&V=I) course. 

The goal of this assignment is to propose your own graph mining technique possibly based on another famous algorithm. 

## Second Order Graph Convolutional Networks

In paper [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) authors use linear approximation of filtering in order to classify nodes of the given graph. In contrast to them, I propose second-order approximation with two kernel tricks. For more information, check folder ```doc```.

Substantial amount of source code is taken from https://github.com/tkipf/pygcn. 

### Installation and usage 

- Installation: ```python setup.py install```
- Usage: ```python train.py```
- Help: ```python train.py --help```
- Valid paths (datasets): ```cora (cora) ```, ```citeseer (citeseer)```, ```WebKB (texas, wisconsin)```

## References
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- https://github.com/tkipf/pygcn
