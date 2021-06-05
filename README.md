
# RiD (Reinforced dynamics)
## 0. **Content**
<!-- vscode-markdown-toc -->
* 1. [Introduction](#Introduction)
* 2. [Installation](#Installation)
	* 2.1. [Environment installation](#Environmentinstallation)
        * 2.1.1. [Install python and tensorflow](#Installpythonandtensorflow)
        * 2.1.2. [Install tensorflow's C++ interface](#Installpythonandtensorflow)
        * 2.1.3. [Install plumed2.5.2](#Installplumed2.5.2)
        * 2.1.4. [Install gromacs 2019.2](#Installgromacs2019.2)
* 3. [Quick Start](#QuickStart)
	* 3.1. [RiD work_path generation](#RiDwork_pathgeneration)
	* 3.2. [Run RiD](#RunRiD)
* 4. [Main procedure of RiD](#MainprocedureofRiD)
		* 4.1. [a. Biased MD](#a.BiasedMD)
		* 4.2. [b. Restrained MD](#b.RestrainedMD)
		* 4.3. [c. Neural network training](#c.Neuralnetworktraining)
* 5. [RiD settings](#RiDsettings)
	* 5.1. [rid.json](#rid.json)


##  1. <a name='Introduction'></a>**Introduction**

RiD-kit is a python package for enhanced sampling via RiD (Reinforced Dynamics) method.

##  2. <a name='Installation'></a>**Installation**

###  2.1. <a name='Environmentinstallation'></a>**Environment installation**

#### 2.1.1. <a name='Installpythonandtensorflow'></a>**Install python and tensorflow** (version<=1.15)

#### 2.1.2 <a name='Installpythonandtensorflow'></a>**Install tensorflow's C++ interface**
The tensorflow's C++ interface will be compiled from the source code, can be found [here](https://github.com/deepmodeling/deepmd-kit/blob/master/doc/install-tf.1.8.md).

#### 2.1.3 <a name='Installplumed2.5.2'></a>**Install plumed2.5.2**
You need copy compiled `DeePFE.cpp` to the plumed directory. This file locats at `install/DeePFE.cpp`
```bash
tar -xvzf plumed-2.5.2.tgz
cp DeePFE.cpp plumed-2.5.2/src/bias
tf_path=$tensorflow_root
CXXFLAGS="-std=gnu++11 -I $tf_path/include/" LDFLAGS=" -L$tf_path/lib -ltensorflow_framework -ltensorflow_cc -Wl,-rpath,$tf_path/lib/" ./configure --prefix=/software/plumed252 CC=mpicc CXX=mpicxx
```
Set the bashrc
```bash
source /software/plumed-2.5.2/sourceme.sh
export PLUMED2_HOME=/software/plumed252
export PATH=$PLUMED2_HOME/bin:$PATH
export LD_LIBRARY_PATH=$PLUMED2_HOME/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$PLUMED2_HOME/pkgconfig:$PKG_CONFIG_PATH
export PLUMED_VIMPATH=$PLUMED2_HOME/vim:$PLUMED_VIMPATH
export INCLUDE=$PLUMED2_HOME/include:$INCLUDE
export PLUMED_KERNEL=/home/dongdong/software/plumed252/lib/libplumedKernel.so
```

#### 2.1.4 <a name='Installgromacs2019.2'></a>**Install gromacs 2019.2**

```bash
tar -xzvf gromacs-2019.2.tar.gz
cd gromacs-2019.2
plumed patch -p
mkdir build
cd build
/software/cmake312/bin/cmake .. -DCMAKE_INSTALL_PREFIX=/software/GMX20192plumed -DGMX_BUILD_OWN_FFTW=ON -DGMX_GPU=on -DGMX_SIMD=avx_256 -DGMX_PREFER_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=OFF -DGMX_EXTERNAL_BLAS=off
make -j 4
make install
```
Set the bashrc
```bash
source /software/GMX20192plumed/bin/GMXRC.bash
```

##  3. <a name='QuickStart'></a>**Quick Start**

Two steps in this version of RiD.
The first is the generation of RiD work path, and then run RiD in the work path.

###  3.1. <a name='RiD work_path generation'></a>**RiD work_path generation**

Let's begin with a simple example, ala2, which has a sequence (1ACE, 2ALA, 3NME).The `cv.json` file can be set as:
```json
{
    "_comment":	   " dihedral angles: phi, psi ",
    "dih_angles" : [ [ {"name" : ["C"],  "resid_shift" : -1},
		       {"name" : ["N"],  "resid_shift" : 0},
		       {"name" : ["CA"], "resid_shift" : 0},
		       {"name" : ["C"],  "resid_shift" : 0} ], 
		     [ {"name" : ["N"],  "resid_shift" : 0}, 
		       {"name" : ["CA"], "resid_shift" : 0},
		       {"name" : ["C"],  "resid_shift" : 0},
		       {"name" : ["N"],  "resid_shift" : 1} ]
		   ],
    "alpha_idx_fmt":	"%03d",
    "angle_idx_fmt":	"%02d"
}
```
`"dih_angles"` is our defination of dihedral angles($\phi$, $\psi$) by default.   

```bash
python gen.py rid ./jsons/default_gen.json ./jsons/CV.json ./mol/alan/amber99sb/01/ -o ala2.rid
```

###  3.2. <a name='run RiD'></a>**run RiD**

```bash
cd ala2.rid
python rid.py rid.json
```
The parameters in `"rid.json"` are described in the following.

##  4. <a name='MainprocedureofRiD'></a>**Main procedure of RiD**

RiD will run in iterations. Every iteration contains tasks below:

1. Biased MD;
2. Restrained MD;
3. Training neural network.

####  4.1. <a name='a.BiasedMD'></a>a. **Biased MD**

Just like Metadynamics, RiD will sample based on a bias potential given by NN models. A uncertainty indicator will direct the process of adding bias potential.

####  4.2. <a name='b.RestrainedMD'></a>b. **Restrained MD**

This procedure will calculate mean force based on the sampling results, which can generate data set for training. 

####  4.3. <a name='c.Neuralnetworktraining'></a>c. **Neural network training**

A fully connected NN will be trained via sampling data. This network will generate a map from selected CV to free energy.

A more detail description of RiD is published now, please see:

>  [1]  Zhang, L., Wang, H., E, W.. Reinforced dynamics for enhanced sampling in large atomic and molecular systems[J]. The Journal of chemical physics, 2018, 148(12): 124113.
>  
>  [2]  Wang, D., Zhang, L., Wang, H., E, W.. Efficient sampling of high-dimensional free energy landscapes using adaptive reinforced dynamics[J]. arXiv preprint arXiv:2104.01620, 2021.


##  5. <a name='RiDsettings'></a>**RiD settings**


Two necessary json files are required to get start a RiD procedure.

1. rid.json for configuration of simulation.
2. cv.json for specifying CV.

###  5.1. <a name='rid.json'></a>**rid.json**

**General setting**

| Parameters | Type | Description | Default/Example |
| :----: | :----: | :----: | :----: |
| gmx_prep | str | Gromacs preparation command | gmx grompp|
| gmx_run | str | Gromacs md run command | gmx mdrun|
| init_graph | list&str | initial graph files list | [] |
| numb_iter | int | number of iterations | 12 |

**Setting for biased MD**

| Parameters | Type | Description | Default/Example |
| :----: | :----: | :----: | :----: |
| numb_walkers | int | number of walkers | 8 |
| bias_trust_lvl_1 | int | trust lower level | 2 |
| bias_trust_lvl_2 | int | trust upper level | 3 |
| bias_nsteps | int | total number of steps of biased MD | 20000 |
| bias_frame_freq | int | frame frequency for recording | 20 |
| sel_threshold | float/int | initial threshold for selection | 2 |
| cluster_threshold | float | * | 1.5 |
| num_of_cluster_threshhold | int | minimum of cluster number | 15 |
| max_sel | int | maximum of selection of clusters | 30 |

**Setting for restrained MD**

| Parameters | Type | Description | Default/Example |
| :----: | :----: | :----: | :----: |
| res_nsteps | int | total number of steps of restrained MD | 25000 |
| res_frame_freq | int | frame frequency for recording| 50 |
| conf_start | int | the index of the first conformation selected | 0 |
| conf_every | int | the stride of conformation selection | 1 |

**Setting for training and neural network**

| Parameters | Type | Description | Default/Example |
| :----: | :----: | :----: | :----: |
| numb_model | int | number of nn models | 4 |
| neurons | list&int | number of nodes for each layer | [200, 200, 200, 200] |
| resnet | bool | whether to use Resnet | True |
| batch_size | int | batch size | 128 |
| numb_epoches | int | total number of epochs for every training | 12000 |
| starter_lr | float | initial learning rate | 0.0008 |
| decay_steps | int | decay steps of lr | 120 |
| decay_rate | float | decay rate of lr | 0.96 |
| res_iter | int | after this iteration, old data will be reduced | 13 |
| res_numb_epoches | int | restrat setting | 2000 |
| res_starter_lr | float | restrat setting | 0.0008 |
| res_olddata_ratio | int/float | restrat setting | 7 |
| res_decay_steps | int | restrat setting | 120 |
| res_decay_rate | float | restrat setting | 0.96 |

