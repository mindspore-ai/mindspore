The operator definitions for MindSpore are implemented as yaml files, in the directory "mindspore/core/ops/ops_def/".

There are two ways to generate python and c++ operator definitions from yaml files:

1. Execute the shell: python mindspore/python/mindspore/ops_generate/gen_ops.py.
2. Run the compilation script: bash build.sh xxx.

Python files generated in the directory "mindspore/python/mindspore/ops/auto_generate/".

C++ files generated in the directory "mindspore/core/ops/auto_generate/".
