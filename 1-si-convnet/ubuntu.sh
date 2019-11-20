sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libgflags-dev python-pip

sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

sudo apt install bc 

wget https://github.com/google/protobuf/releases/download/v2.5.0/protobuf-2.5.0.tar.gz

tar zxvf protobuf-2.5.0.tar.gz


cd protobuf-2.5.0/
./configure --prefix=/home/local_install/
make
make install
export PATH=/home/local_install/bin/:$PATH
cd ../


wget http://sourceforge.net/projects/boost/files/boost/1.56.0/boost_1_56_0.tar.bz2
tar jxvf boost_1_56_0.tar.bz2
cd boost_1_56_0/
./bootstrap.sh --with-libraries=system, thread, python
./b2

cp -r boost/ /home/local_install/include
cp stage/lib/* /home/local_install/lib
cd ../



wget https://github.com/gflags/gflags/archive/v2.1.1.tar.gz
tar zxvf v2.1.1.tar.gz 
cd gflags-2.1.1

mkdir build
cd build/
cmake ..
sudo apt-get install cmake-curses-gui
ccmake ..

                                                     Page 1 of 1
 BUILD_PACKAGING                  OFF
 BUILD_SHARED_LIBS                ON
 BUILD_TESTING                    OFF
 BUILD_gflags_LIB                 ON
 BUILD_gflags_nothreads_LIB       ON
 CMAKE_BUILD_TYPE
 CMAKE_INSTALL_PREFIX             /home/local_install


