bootstrap: docker
from: nvidia/cuda:10.2-devel-centos7




%help
Centos7 with conda python, jupyter, latex, cuda and mkl




%environment
CUDA_ROOT=/usr/local/cuda
export CUDA_ROOT
SINGULARITY_SHELL=/bin/zsh
export SINGULARITY_SHELL
PIP_TARGET=/opt/pip-packages
export PIP_TARGET
PYTHONPATH=/opt/pip-packages:$PYTHONPATH
export PYTHONPATH
export PATH
MKLROOT=/opt/intel/compilers_and_libraries/linux/mkl
export MKLROOT
TBBROOT=/opt/intel/compilers_and_libraries/linux/tbb
export TBBROOT
INTEL_LICENSE_FILE=/opt/intel/compilers_and_libraries/linux/licenses
export INTEL_LICENSE_FILE
LIBRARY_PATH=$LIBRARY_PATH:/opt/intel/compilers_and_libraries/linux/compiler/lib/intel64_lin:/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64_lin:/opt/intel/compilers_and_libraries/linux/tbb/lib/intel64/gcc4.8:/opt/intel/compilers_and_libraries/linux/tbb/lib/intel64/gcc4.8
export LIBRARY_PATH
CPATH=$CPATH:/opt/intel/compilers_and_libraries/linux/mkl/include:/opt/intel/compilers_and_libraries/linux/tbb/include:/opt/intel/compilers_and_libraries/linux/tbb/include
export CPATH
PATH=$PATH:/opt/anaconda3/bin:/usr/local/texlive/bin/x86_64-linux:/usr/local/cuda/bin
PATH=$PATH:/opt/intel/compilers_and_libraries/linux/bin/intel64:/opt/intel/compilers_and_libraries/linux/bin
export PATH
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/opt/intel/compilers_and_libraries/linux/mkl/bin/pkgconfig
export PKG_CONFIG_PATH




%post
##yum
yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
yum -y install epel-release https://repo.ius.io/ius-release-el7.rpm
yum update -y
yum upgrade -y
yum install -y mc unzip libtiff  make cmake binutils mosh less vim aria2  rsync openssh  screen  tmux htop curl  wget zsh  git224  perl-Digest-MD5 zip  binutils gcc gcc-c++ gettext  man man-pages libtool make patch elfutils  patchutils gdb diffutils  openmpi-devel environment-modules  fontconfig freetype freetype-devel fontconfig-devel libstdc++ strace ltrace ghostscript intel-mkl
yum clean all
rm -rf /var/cache/yum


##texlive
echo "installing texlive"
cat > /tmp/texlive.profile << "EOF0"
# texlive.profile
selected_scheme scheme-minimal
TEXDIR /usr/local/texlive/
TEXMFCONFIG ~/.texlive/texmf-config
TEXMFHOME ~/.texlive/texmf
TEXMFLOCAL /usr/local/texlive/texmf-local
TEXMFSYSCONFIG /usr/local/texlive/texmf-config
TEXMFSYSVAR /usr/local/texlive/texmf-var
TEXMFVAR ~/.texlive/texmf-var
binary_x86_64-linux 1
instopt_adjustpath 0
instopt_adjustrepo 1
instopt_letter 0
instopt_portable 0
instopt_write18_restricted 1
tlpdbopt_autobackup 1
tlpdbopt_backupdir tlpkg/backups
tlpdbopt_create_formats 1
tlpdbopt_desktop_integration 1
tlpdbopt_file_assocs 1
tlpdbopt_generate_updmap 0
tlpdbopt_install_docfiles 1
tlpdbopt_install_srcfiles 1
tlpdbopt_post_code 1
tlpdbopt_sys_bin /usr/local/bin
tlpdbopt_sys_info /usr/local/share/info
tlpdbopt_sys_man /usr/local/share/man
tlpdbopt_w32_multi_user 1
EOF0

wget -q -O /tmp/install-tl-unx.tar.gz http://ftp.acc.umu.se/mirror/CTAN/systems/texlive/tlnet/install-tl-unx.tar.gz && \
    cd /tmp/ && tar xzf install-tl-unx.tar.gz
cd /tmp/install-tl-* && ./install-tl -profile /tmp/texlive.profile -no-verify-downloads -persistent-downloads -q
rm -rf /tmp/install-tl*


#miniconda
echo "installing miniconda"
wget -q -O /tmp/Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh
cd /tmp/&& bash /tmp/Miniconda3.sh -b -p /opt/anaconda3
rm /tmp/Miniconda3.sh
ln -s /opt/anaconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh
PATH=$PATH:/usr/local/cuda/bin:/opt/anaconda3/bin


#conda env
. /etc/profile.d/conda.sh
echo "installing conda extensions"
conda activate
conda install -q -y "numpy<1.17" hdf5 scipy numba numexpr mkl cython  "jupyterlab=2"  jupyter scikit-image appdirs mako scikit-learn  cupy "python>3.6" seaborn pandas line_profiler black ninja colorama
conda install -q -y -c  conda-forge lmfit ipympl "nodejs>=12"  ptvsd xeus-python pytools nbdime "pip>=20.1"
conda install -q -y -c conda-forge -c plotly jupyter-dash

#jupyterlab extensions
echo "installing jlab extensions"
jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build
jupyter labextension install ipyvolume --no-build
jupyter labextension install jupyter-threejs --no-build
jupyter labextension install @krassowski/jupyterlab_go_to_definition --no-build
jupyter labextension install @aquirdturtle/collapsible_headings --no-build
jupyter labextension install @jupyterlab/google-drive --no-build
pip install -q --upgrade jupyterlab-git jupyterlab_code_formatter  jupyterlab_latex
jupyter labextension install @jupyterlab/latex --no-build
jupyter labextension install @ryantam626/jupyterlab_code_formatter --no-build
jupyter labextension install @jupyterlab/git --no-build
jupyter labextension install @jupyterlab/debugger --no-build
jupyter labextension install  jupyterlab-dash --no-build
jupyter serverextension enable --py jupyterlab_code_formatter --sys-prefix
jupyter serverextension enable --sys-prefix jupyterlab_latex
jupyter serverextension enable --py jupyterlab_git --sys-prefix
jupyter lab build

#disable extensions not working with sdf hub
#jupyter labextension disable @jupyterlab/git
#jupyter labextension disable @ryantam626/jupyterlab_code_formatter
#jupyter labextension disable @jupyterlab/latex

cat > /opt/anaconda3/etc/jupyter/jupyter_config.json << "EOF3"
{
  "NotebookApp": {
    "nbserver_extensions": {
      "jupyterlab_git": true,
      "jupyterlab": true,
      "jupyterlab_latex": true,
      "nbdime": true,
      "jupyterlab_code_formatter": true,
    }
  }
}
EOF3

rm -rf /tmp/npm*


#pycuda
echo "installing pycuda from pip"
LIBRARY_PATH=/usr/local/cuda/lib64/stubs CPATH=/usr/local/cuda/include CUDA_ROOT=/usr/local/cuda pip install pycuda

#idi
echo "installing idi from github"
pip install git+https://github.com/fzimmermann89/idi

#pip-packages for overlay
mkdir /opt/pip-packages
chmod 777 /opt/pip-packages


#conda config
conda config  --file /opt/anaconda3/.condarc --set changeps1 False
conda config  --file /opt/anaconda3/.condarc --set auto_activate_base true

cat > /etc/profile.d/zconda.sh << "EOF1"
#. /etc/profile.d/conda.sh
conda activate
EOF1

cat > /opt/runscript.sh << "EOF2"
#!/bin/bash
. /etc/profile.d/conda.sh
conda activate
exec "$@"
EOF2
chmod +x /opt/runscript.sh


#cleanup
conda clean -a
pip cache purge
rm -rf /root/.conda
rm -rf /root/.npm
rm -rf /root/.cache




%labels
    Maintainer zimmf
    Version 0.1-200910
    
    
    
    
%runscript
exec /opt/runscript.sh python "$@"

%startscript
exec /opt/runscript.sh jupyter lab "$@"

%apprun jupyter
exec /opt/runscript.sh jupyter "$@"

%apprun ipython
exec /opt/runscript.sh ipython "$@"

%apprun pdftex
exec pdftex "$@"
