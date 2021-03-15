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

#include "nnacl/infer/where_infer.h"
#include "nnacl/infer/infer_register.h"

int WhereInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                    OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];

  // Need to dynamically allocate at runtime.
  if (inputs_size == 1) {
    return NNACL_INFER_INVALID;
  }

  if (inputs_size < 3 || outputs_size != 1) {
    return NNACL_INPUT_TENSOR_ERROR;
  }

  SetDataTypeFormat(output, input);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  const TensorC *input0 = inputs[0];
  const TensorC *input1 = inputs[1];
  const TensorC *input2 = inputs[2];
  int num = GetElementNum(input0);
  int num1 = GetElementNum(input1);
  int num2 = GetElementNum(input2);
  int nummax = num > num1 ? num : (num1 > num2 ? num1 : num2);
  int axisout = 0;
  size_t temp = 0;
  for (size_t j = 0; j < input0->shape_size_; j++) {
    if (input0->shape_[j] == input1->shape_[j] && input0->shape_[j] != input2->shape_[j]) {
      axisout = j;
      break;
    }
    if (input0->shape_[j] == input2->shape_[j] && input0->shape_[j] != input1->shape_[j]) {
      axisout = j;
      break;
    }
    if (input1->shape_[j] == input2->shape_[j] && input0->shape_[j] != input1->shape_[j]) {
      axisout = j;
      break;
    }
    temp += 1;
    if (temp == input0->shape_size_) {
      SetShapeTensor(output, input);
      output->data_type_ = input->data_type_;
      return NNACL_OK;
    }
  }
  ShapeSet(output->shape_, &output->shape_size_, input0->shape_, input0->shape_size_);
  output->shape_[axisout] = nummax;
  return NNACL_OK;
}

REG_INFER(Where, PrimType_Where, WhereInferShape)
