Build Ascend customized kernel.
More details please refer to https://gitee.com/ascend/samples.git.

## build

mkdir build
cd build
cmake ../
make

## install

./ms_ascend_custom_kernel_installer.run

After install, you can use converter tools to convert model with customized kernel on Ascend developing env.
