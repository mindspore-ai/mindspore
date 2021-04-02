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

#include "wrapper/int8/convolution_depthwise_int8_wrapper.h"

int ConvDepthwiseInt8Run(void *cdata, int task_id) {
  ConvDepthwiseInt8Args *args = (ConvDepthwiseInt8Args *)cdata;
  int32_t *buffer = args->row_buffer_ + args->conv_param_->output_w_ * args->conv_param_->output_channel_ * task_id;
  ConvDwInt8(args->output_data_, buffer, args->input_data_, args->weight_data_, args->bias_data_, args->conv_param_,
             task_id);
  return NNACL_OK;
}
