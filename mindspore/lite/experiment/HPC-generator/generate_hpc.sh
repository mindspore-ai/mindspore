#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# generate gemm fma asm code
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=12 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_12x8_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=11 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_11x8_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=10 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_10x8_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=9 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_9x8_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=8 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_8x8_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=7 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_7x8_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=6 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_6x8_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=5 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_5x8_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=4 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_4x8_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=3 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_3x8_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=2 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_2x8_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=1 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_1x8_kernel_nc8hw8_fp32_asm.c
#
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=6 col_block=16 -O ./gemm_fma/nnacl_gemm_fma_6x16_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=5 col_block=16 -O ./gemm_fma/nnacl_gemm_fma_5x16_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=4 col_block=16 -O ./gemm_fma/nnacl_gemm_fma_4x16_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=3 col_block=16 -O ./gemm_fma/nnacl_gemm_fma_3x16_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=2 col_block=16 -O ./gemm_fma/nnacl_gemm_fma_2x16_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=1 col_block=16 -O ./gemm_fma/nnacl_gemm_fma_1x16_kernel_nc8hw8_fp32_asm.c
#
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=4 col_block=24 -O ./gemm_fma/nnacl_gemm_fma_4x24_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=3 col_block=24 -O ./gemm_fma/nnacl_gemm_fma_3x24_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=2 col_block=24 -O ./gemm_fma/nnacl_gemm_fma_2x24_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=1 col_block=24 -O ./gemm_fma/nnacl_gemm_fma_1x24_kernel_nc8hw8_fp32_asm.c
#
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=3 col_block=32 -O ./gemm_fma/nnacl_gemm_fma_3x32_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=2 col_block=32 -O ./gemm_fma/nnacl_gemm_fma_2x32_kernel_nc8hw8_fp32_asm.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8_asm.c.in -A row_block=1 col_block=32 -O ./gemm_fma/nnacl_gemm_fma_1x32_kernel_nc8hw8_fp32_asm.c
#
## generate gemm fma intrinics code
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=12 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_12x8_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=11 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_11x8_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=10 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_10x8_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=9 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_9x8_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=8 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_8x8_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=7 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_7x8_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=6 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_6x8_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=5 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_5x8_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=4 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_4x8_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=3 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_3x8_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=2 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_2x8_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=1 col_block=8 -O ./gemm_fma/nnacl_gemm_fma_1x8_kernel_nc8hw8_fp32.c
#
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=6 col_block=16 -O ./gemm_fma/nnacl_gemm_fma_6x16_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=5 col_block=16 -O ./gemm_fma/nnacl_gemm_fma_5x16_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=4 col_block=16 -O ./gemm_fma/nnacl_gemm_fma_4x16_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=3 col_block=16 -O ./gemm_fma/nnacl_gemm_fma_3x16_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=2 col_block=16 -O ./gemm_fma/nnacl_gemm_fma_2x16_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=1 col_block=16 -O ./gemm_fma/nnacl_gemm_fma_1x16_kernel_nc8hw8_fp32.c
#
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=4 col_block=24 -O ./gemm_fma/nnacl_gemm_fma_4x24_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=3 col_block=24 -O ./gemm_fma/nnacl_gemm_fma_3x24_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=2 col_block=24 -O ./gemm_fma/nnacl_gemm_fma_2x24_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=1 col_block=24 -O ./gemm_fma/nnacl_gemm_fma_1x24_kernel_nc8hw8_fp32.c
#
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=3 col_block=32 -O ./gemm_fma/nnacl_gemm_fma_3x32_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=2 col_block=32 -O ./gemm_fma/nnacl_gemm_fma_2x32_kernel_nc8hw8_fp32.c
#python3 generator.py -I ./template_file/gemm_fma_nc8hw8.c.in -A row_block=1 col_block=32 -O ./gemm_fma/nnacl_gemm_fma_1x32_kernel_nc8hw8_fp32.c

# generate gemm avx512 asm code
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=4 col_block=96 -O ./gemm_avx512/nnacl_gemm_avx512_4x96_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=3 col_block=96 -O ./gemm_avx512/nnacl_gemm_avx512_3x96_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=2 col_block=96 -O ./gemm_avx512/nnacl_gemm_avx512_2x96_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=1 col_block=96 -O ./gemm_avx512/nnacl_gemm_avx512_1x96_kernel_nhwc_fp32.c

python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=5 col_block=80 -O ./gemm_avx512/nnacl_gemm_avx512_5x80_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=4 col_block=80 -O ./gemm_avx512/nnacl_gemm_avx512_4x80_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=3 col_block=80 -O ./gemm_avx512/nnacl_gemm_avx512_3x80_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=2 col_block=80 -O ./gemm_avx512/nnacl_gemm_avx512_2x80_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=1 col_block=80 -O ./gemm_avx512/nnacl_gemm_avx512_1x80_kernel_nhwc_fp32.c

python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=6 col_block=64 -O ./gemm_avx512/nnacl_gemm_avx512_6x64_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=5 col_block=64 -O ./gemm_avx512/nnacl_gemm_avx512_5x64_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=4 col_block=64 -O ./gemm_avx512/nnacl_gemm_avx512_4x64_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=3 col_block=64 -O ./gemm_avx512/nnacl_gemm_avx512_3x64_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=2 col_block=64 -O ./gemm_avx512/nnacl_gemm_avx512_2x64_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=1 col_block=64 -O ./gemm_avx512/nnacl_gemm_avx512_1x64_kernel_nhwc_fp32.c

python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=8 col_block=32 -O ./gemm_avx512/nnacl_gemm_avx512_8x32_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=7 col_block=32 -O ./gemm_avx512/nnacl_gemm_avx512_7x32_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=6 col_block=32 -O ./gemm_avx512/nnacl_gemm_avx512_6x32_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=5 col_block=32 -O ./gemm_avx512/nnacl_gemm_avx512_5x32_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=4 col_block=32 -O ./gemm_avx512/nnacl_gemm_avx512_4x32_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=3 col_block=32 -O ./gemm_avx512/nnacl_gemm_avx512_3x32_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=2 col_block=32 -O ./gemm_avx512/nnacl_gemm_avx512_2x32_kernel_nhwc_fp32.c
python3 generator.py -I ./template_file/gemm_avx512_nhwc_asm.c.in -A row_block=1 col_block=32 -O ./gemm_avx512/nnacl_gemm_avx512_1x32_kernel_nhwc_fp32.c
