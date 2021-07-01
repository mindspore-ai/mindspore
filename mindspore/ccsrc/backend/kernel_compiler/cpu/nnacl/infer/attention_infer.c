/**
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

#include "nnacl/infer/attention_infer.h"
#include "nnacl/infer/infer_register.h"

int AttentionInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                        OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 7, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  const TensorC *q_input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, q_input);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  const TensorC *q_weight = inputs[3];
  if (q_input->shape_size_ != 2 && q_input->shape_size_ != 3) {
    return NNACL_ERR;
  }
  if (q_weight->shape_size_ != 2) {
    return NNACL_ERR;
  }
  int batch = (q_input->shape_size_ == 2) ? 1 : q_input->shape_[0];
  int f_seq = (q_input->shape_size_ == 2) ? q_input->shape_[0] : q_input->shape_[1];
  int d_model = q_weight->shape_[1];

  output->shape_[0] = batch;
  output->shape_[1] = f_seq;
  output->shape_[2] = d_model;
  output->shape_size_ = 3;
  return NNACL_OK;
}

REG_INFER(Attention, PrimType_Attention, AttentionInferShape)
