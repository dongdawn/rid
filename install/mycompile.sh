tf_path=$HOME/SCR//local/tensorflow/r1.8-gpu
CXXFLAGS="-std=gnu++11 -I $tf_path/include/" LDFLAGS=" -L$tf_path/lib -ltensorflow_framework -ltensorflow_cc -Wl,-rpath,$tf_path/lib/" ./configure --prefix=$HOME/SCR/wanghan//local/

# CXXFLAGS="-std=gnu++11 -I $HOME/local/include/google/tensorflow/ -I $HOME/local/include/ -I $HOME/local/include/google/protobuf/ -I $HOME/local/include/eigen/eigen-eigen-f3a22f35b044/" LDFLAGS=" -L$HOME/local/lib/ -lprotobuf -ltensorflow_all -Wl,-rpath,$HOME/local/lib/"  ./configure

# CXXFLAGS="-I ~/local/include/google/tensorflow/ -I ~/local/include/ -I ~/local/include/google/protobuf/ -I ~/local/include/eigen/eigen-eigen-f3a22f35b044/" LDFLAGS=" -L$HOME/local/lib/ -lprotobuf -ltensorflow_all -Wl,-rpath,$HOME/local/lib/"  ./configure
