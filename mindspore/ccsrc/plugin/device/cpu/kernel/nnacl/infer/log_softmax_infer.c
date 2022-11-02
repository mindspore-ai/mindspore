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

#include "nnacl/infer/log_softmax_infer.h"
#include "nnacl/infer/infer_register.h"

int LogSoftmaxInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                         OpParameter *parameter) {
  const int input_size_limit = 1;
  const int output_size_limit = 1;
  if (inputs_size != input_size_limit || outputs_size != output_size_limit) {
    return NNACL_ERR;
  }
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  if (input->shape_size_ > 5) {
    return NNACL_ERR;
  }
  SetShapeTensor(output, input);
  SoftmaxParameter *param = (SoftmaxParameter *)parameter;
  NNACL_CHECK_NULL_RETURN_ERR(param);
  if (param->axis_ < (-1 * (int)(input->shape_size_)) || param->axis_ >= (int)(input->shape_size_)) {
    return NNACL_PARAM_INVALID;
  }
  return NNACL_OK;
}

REG_INFER(LogSoftmax, PrimType_LogSoftmax, LogSoftmaxInferShape)
