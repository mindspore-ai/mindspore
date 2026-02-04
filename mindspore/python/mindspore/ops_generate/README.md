The operator definitions for MindSpore are implemented as yaml files, in the directory "mindspore/ops/op_def/yaml".

There are two ways to generate python and c++ operator definitions from yaml files:

1. Execute the shell: python mindspore/python/mindspore/ops_generate/gen_ops.py.
2. Run the compilation script: bash build.sh xxx.

Python files generated in the directory "mindspore/python/mindspore/ops/auto_generate/".

C++ source files (.cc) generated in the directory "mindspore/ops/op_def/auto_generate/".

C++ header files (.h) generated in the directory "mindspore/ops/include/primitive/auto_generate/".
