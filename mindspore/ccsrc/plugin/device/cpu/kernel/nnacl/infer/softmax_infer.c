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

#include "nnacl/infer/softmax_infer.h"
#include "nnacl/infer/infer_register.h"

int SoftMaxInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                      OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];

  output->data_type_ = input->data_type_;
  output->format_ = input->format_;
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  // there is a model with an 8-dim input, which runs on ascend910.
  if (input->shape_size_ > DIMENSION_8D) {
    return NNACL_ERR;
  }

  SoftmaxParameter *param = (SoftmaxParameter *)parameter;
  NNACL_CHECK_NULL_RETURN_ERR(param);
  if (param->axis_ < (-1 * (int)(input->shape_size_)) || param->axis_ > (int)(input->shape_size_)) {
    return NNACL_PARAM_INVALID;
  }
  SetShapeTensor(output, input);
  return NNACL_OK;
}

REG_INFER(Softmax, PrimType_Softmax, SoftMaxInferShape)
