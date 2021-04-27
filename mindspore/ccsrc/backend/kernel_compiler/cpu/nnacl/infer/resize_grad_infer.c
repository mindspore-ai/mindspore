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

#include "nnacl/infer/resize_grad_infer.h"
#include "nnacl/infer/infer_register.h"

int ResizeGradInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                         OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  if (input->shape_size_ != 4) {
    return NNACL_ERR;
  }
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  const TensorC *input_1 = inputs[1];
  if (input_1->shape_size_ == 4) {
    ShapeSet(output->shape_, &output->shape_size_, input_1->shape_, input_1->shape_size_);
  } else if (input_1->shape_size_ == 1 && input_1->shape_[0] == 2 && input_1->data_type_ == kNumberTypeInt32) {
    int output_shape[MAX_SHAPE_SIZE] = {0};
    size_t output_shape_size = 0;
    int32_t *data = (int32_t *)(input_1->data_);

    ShapePush(output_shape, &output_shape_size, GetBatch(input));
    ShapePush(output_shape, &output_shape_size, data[0]);
    ShapePush(output_shape, &output_shape_size, data[1]);
    ShapePush(output_shape, &output_shape_size, GetChannel(input));
    SetShapeArray(output, output_shape, output_shape_size);
  } else {
    return NNACL_ERR;
  }
  return NNACL_OK;
}

REG_INFER(ResizeGrad, PrimType_ResizeGrad, ResizeGradInferShape)
