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
#include "nnacl/attention_parameter.h"

int AttentionInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                        OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 7, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  AttentionParameter *param = (AttentionParameter *)parameter;
  const TensorC *q_input = inputs[FIRST_INPUT];
  const TensorC *k_input = inputs[SECOND_INPUT];
  TensorC *output0 = outputs[FIRST_INPUT];
  SetDataTypeFormat(output0, q_input);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  const TensorC *q_weight = inputs[FOURTH_INPUT];
  if (q_input->shape_size_ != C2NUM && q_input->shape_size_ != C3NUM) {
    return NNACL_ERR;
  }
  if (q_weight->shape_size_ != C2NUM) {
    return NNACL_ERR;
  }
  int batch = (q_input->shape_size_ == C2NUM) ? 1 : q_input->shape_[0];
  int f_seq = (q_input->shape_size_ == C2NUM) ? q_input->shape_[0] : q_input->shape_[1];
  int t_seq_len = k_input->shape_[1];
  if (q_input->shape_size_ == C2NUM) {
    output0->shape_[FIRST_INPUT] = batch * f_seq;
    output0->shape_[SECOND_INPUT] = param->head_num_ * param->head_size_;
    output0->shape_size_ = C2NUM;
  } else {
    output0->shape_[FIRST_INPUT] = batch;
    output0->shape_[SECOND_INPUT] = f_seq;
    output0->shape_[THIRD_INPUT] = param->head_num_ * param->head_size_;
    output0->shape_size_ = C3NUM;
  }
  if (outputs_size >= C3NUM) {
    TensorC *output1 = outputs[SECOND_INPUT];
    SetDataTypeFormat(output1, q_input);
    output1->shape_[FIRST_INPUT] = batch;
    output1->shape_[SECOND_INPUT] = param->head_num_;
    output1->shape_[THIRD_INPUT] = param->head_size_;
    output1->shape_[FOURTH_INPUT] = t_seq_len;
    output1->shape_size_ = C4NUM;
    TensorC *output2 = outputs[THIRD_INPUT];
    SetDataTypeFormat(output2, q_input);
    output2->shape_[FIRST_INPUT] = batch;
    output2->shape_[SECOND_INPUT] = param->head_num_;
    output2->shape_[THIRD_INPUT] = t_seq_len;
    output2->shape_[FOURTH_INPUT] = param->head_size_;
    output2->shape_size_ = C4NUM;
  }
  return NNACL_OK;
}

REG_INFER(Attention, PrimType_Attention, AttentionInferShape)
