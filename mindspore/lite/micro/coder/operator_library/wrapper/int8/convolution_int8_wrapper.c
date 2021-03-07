/*
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "wrapper/int8/convolution_int8_wrapper.h"

void CheckSupportOptimize(const ConvolutionInt8Args *args) {
  int tile_num = 8;
#ifdef ENABLE_ARM32
  tile_num = 4;
  args->is_optimize_ = false;
#endif
#ifdef ENABLE_ARM64
  if (mindspore::lite::IsSupportSDot()) {
    matmul_func_ = MatMulRInt8_optimize_handler;
    args->is_optimize_ = true;
  } else {
    tile_num = 4;
    args->is_optimize_ = false;
  }
#endif
  args->conv_param_->tile_num_ = tile_num;
}

int ConvolutionInt8Run(void *cdata, int task_id) {
  ConvolutionInt8Args *args = (ConvolutionInt8Args *)cdata;
  ConvInt8(args->input_data_, args->packed_input_, args->matmul_input_, args->packed_weight_, args->bias_data_,
           args->output_data_, args->filter_zp_, args->input_sum_, task_id, args->conv_param_, args->matmul_func_,
           args->is_optimize_);
  return NNACL_OK;
}
