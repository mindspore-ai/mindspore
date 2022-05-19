/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDSPORE_NNACL_EXPERIMENT_CONV_FP32_AVX512_H_
#define MINDSPORE_NNACL_EXPERIMENT_CONV_FP32_AVX512_H_
#include "nnacl/kernel.h"

int conv2d_prepare_fp32_nchwx_avx512(struct KernelBase *self);
int conv2d_release_fp32_nchwx_avx512(struct KernelBase *self);
int conv2d_compute_fp32_nchwx_avx512(struct KernelBase *self);
int conv2d_infershape_fp32_nchwx_avx512(struct KernelBase *self);
int conv2d_resize_fp32_nchwx_avx512(struct KernelBase *self);
#endif
