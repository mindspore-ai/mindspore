/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef NNACL_KERNEL_DECONVOLUTION_WINOGRAD_H_
#define NNACL_KERNEL_DECONVOLUTION_WINOGRAD_H_

#ifndef _WIN32
#ifndef ENABLE_MCU
#include <pthread.h>
#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"
#include "nnacl/kernel/convolution_base.h"

#define kDeconvWinogradMaxPixel 3145728
#define WINOGRAD_DEFAULT_UNIT 3
#define WINOGRAD_DEFAULT_TILE 8
#define WINOGRAD_MAX_COUNT 8

typedef struct DeConvWinogradStruct {
  ConvolutionBaseStruct conv_;
  DeConvParam param_;
  pthread_mutex_t lock_;
  int thread_num_hw_;
  int thread_stride_hw_;
  float *nhwc_input_;
  float *nhwc_output_;
  float *tile_input_;
  float *tile_output_;
  float *origin_input_;
  float *nc4hw4_output_;
  bool valid_weight_shape_;
} DeConvWinogradStruct;

#define NNACL_DECONV_WINOGRAD_HW_MAX 2000

ConvolutionBaseStruct *CreateDeConvWinograd(ConvParameter *param);
#endif
#endif
#endif  // NNACL_KERNEL_DECONVOLUTION_WINOGRAD_H_
