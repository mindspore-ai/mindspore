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

#ifndef NNACL_KERNEL_CONVOLLUTION_BASE_H_
#define NNACL_KERNEL_CONVOLLUTION_BASE_H_

#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/kernel.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/tensor_c_utils.h"

#define ConvMinBlock 1

typedef struct ConvolutionBaseStruct {
  KernelBase base_;
  ConvComputeParam compute_;
  bool weight_is_packed_;
  bool is_repack_;
  bool infershape_done_;
  bool use_batch_cut_flag_;
  FormatC out_format_;

  void *packed_weight_;
  void *bias_data_;
  void *origin_weight_;  // do not Free
  void *origin_bias_;    // do not Free

  void (*init_global_variable_)(struct ConvolutionBaseStruct *conv_im2col);
  int (*malloc_weight_bias_)(struct ConvolutionBaseStruct *conv_base);
  void (*pack_weight_)(struct ConvolutionBaseStruct *conv_base);
  int (*run_impl_)(struct ConvolutionBaseStruct *conv, int task_id);

  bool is_sharing_pack_;
  void *shaing_manager_;
  void (*free_sharing_weight_)(void *manager, void *tensor_data);
  void *(*get_sharing_weight_)(void *manager, const void *tensor_data, const size_t size, bool *is_packed);
} ConvolutionBaseStruct;

int ConvBaseUpdateParamInfo(ConvComputeParam *compute, ConvParameter *conv_param);
int ConvBaseUpdateComputeInfo(ConvolutionBaseStruct *conv);
void ConvBaseRelease(ConvolutionBaseStruct *conv);
int ConvBaseCheckResizeValid(ConvolutionBaseStruct *conv);
int ConvBasePrepare(ConvolutionBaseStruct *conv);
int ConvBaseInitConvWeightBias(ConvolutionBaseStruct *conv);
int ConvBaseRepackWeight(ConvolutionBaseStruct *conv);
void ConvBaseUpdateOriginWeightAndBias(ConvolutionBaseStruct *conv);
void *ConvBaseGetConvPackWeightData(ConvolutionBaseStruct *conv, int data_size);

#endif  // NNACL_KERNEL_CONVOLLUTION_BASE_H_
