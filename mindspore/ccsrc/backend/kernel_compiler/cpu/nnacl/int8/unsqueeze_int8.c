/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "nnacl/int8/unsqueeze_int8.h"
#include "nnacl/unsqueeze_parameter.h"

int Int8Unsqueeze(int8_t *input_ptr, int8_t *output_ptr, UnSqueezeParameter *para_, size_t data_size, int task_id) {
  float output_scale = para_->quant_arg.out_quant_args_.scale_;
  int8_t output_zp = para_->quant_arg.out_quant_args_.zp_;
  float input_scale = para_->quant_arg.in_quant_args_.scale_;
  int8_t input_zp = para_->quant_arg.in_quant_args_.zp_;

  for (int i = task_id; i < data_size; i += para_->thread_count_) {
    output_ptr[i] = output_zp + round(1 / output_scale * input_scale * (input_ptr[i] - input_zp));
  }
  return 0;
}
