/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "nnacl/infer/crop_and_resize_infer.h"
#include "nnacl/infer/infer_register.h"

int CropAndResizeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                            OpParameter *parameter) {
  int check_ret = CheckAugmentNullInputSize(inputs, inputs_size, outputs, outputs_size, parameter, 4);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  if (outputs_size < 1) {
    return NNACL_INPUT_TENSOR_ERROR;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  if (input->shape_size_ != 0 && input->shape_size_ != 4) {
    return NNACL_ERR;
  }
  int output_shape[MAX_SHAPE_SIZE] = {0};
  size_t output_shape_size = 0;
  if (GetBatch(input) == 0) {
    ShapePush(output_shape, &output_shape_size, 0);
  } else if (inputs[1]->data_ != NULL) {
    const TensorC *boxes_tensor = inputs[1];
    if (boxes_tensor->shape_size_ < 1) {
      return NNACL_INPUT_TENSOR_ERROR;
    }
    ShapePush(output_shape, &output_shape_size, boxes_tensor->shape_[0]);
  } else {
    return NNACL_INFER_INVALID;
  }

  const TensorC *shape_tensor = inputs[3];
  int32_t *data = (int32_t *)(shape_tensor->data_);
  if (data == NULL) {
    return NNACL_INFER_INVALID;
  }
  if (GetElementNum(shape_tensor) < 2) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  ShapePush(output_shape, &output_shape_size, data[0]);
  ShapePush(output_shape, &output_shape_size, data[1]);
  ShapePush(output_shape, &output_shape_size, GetChannel(input));
  SetShapeArray(output, output_shape, output_shape_size);
  return NNACL_OK;
}

REG_INFER(CropAndResize, PrimType_CropAndResize, CropAndResizeInferShape)
