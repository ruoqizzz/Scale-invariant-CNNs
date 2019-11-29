#!/usr/bin/env sh
# This scripts downloads the mnist data and unzips it.

# DIR="$( cd "$(dirname "$0")" ; pwd -P )"
# cd $DIR

echo "Downloading..."

wget http://angjookanazawa.com/sicnn/mnist-sc-table1.tar.gz

echo "Unzipping..."

tar vxzf mnist-sc-table1.tar.gz

# Creation is split out because leveldb sometimes causes segfault
# and needs to be re-created.

echo "Done."
