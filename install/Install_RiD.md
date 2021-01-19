1, Install python and tensorflow (version<=1.15)

2, Install tensorflow's C++ interface
The tensorflow's C++ interface will be compiled from the source code, can be found [here](https://github.com/deepmodeling/deepmd-kit/blob/master/doc/install-tf.1.8.md).

3, Install plumed2.5.2

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

4, Install gromacs 2019.2

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
