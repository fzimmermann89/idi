#!/bin/bash -ex
#https://github.com/singularityhub/travis-ci/blob/master/.travis/setup.sh
sudo sed -i -e 's/^Defaults\tsecure_path.*$//' /etc/sudoers

# Install Singularity

SINGULARITY_BASE="${GOPATH}/src/github.com/sylabs/singularity"
export PATH="${GOPATH}/bin:${PATH}"

mkdir -p "${GOPATH}/src/github.com/sylabs"
cd "${GOPATH}/src/github.com/sylabs"

export VERSION=3.7.0 
wget https://github.com/hpcng/singularity/releases/download/v${VERSION}/singularity-${VERSION}.tar.gz
tar -xzf singularity-${VERSION}.tar.gz
cd singularity
./mconfig -v -p /usr/local
make -j `nproc 2>/dev/null || echo 1` -C ./builddir all
sudo make -C ./builddir install
