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

#include "nnacl/infer/resize_infer.h"

int ResizeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  const TensorC *input = inputs[0];
  if (input->shape_size_ != 0 && input->shape_size_ != 4) {
    return NNACL_ERR;
  }
  TensorC *output = outputs[0];

  SetDataTypeFormat(output, input);
  ResizeParameter *param = (ResizeParameter *)parameter;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  int output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = 0;
  ShapePush(output_shape, &output_shape_size, GetBatch(input));
  if (inputs_size == 2) {
    const TensorC *shape_tensor = inputs[1];
    if (shape_tensor->data_ == NULL) {
      return NNACL_INFER_INVALID;
    }
    size_t shape_size = GetElementNum(shape_tensor);
    switch (shape_size) {
      case 4: {
        if (shape_tensor->data_type_ == kNumberTypeInt32) {
          int32_t *data = (int32_t *)(shape_tensor->data_);
          if (data == NULL) {
            return NNACL_INFER_INVALID;
          }
          switch (shape_tensor->format_) {
            case Format_NCHW:
              ShapePush(output_shape, &output_shape_size, data[2] * GetHeight(input));
              ShapePush(output_shape, &output_shape_size, data[3] * GetWidth(input));
              break;
            case Format_NHWC:
              ShapePush(output_shape, &output_shape_size, data[1] * GetHeight(input));
              ShapePush(output_shape, &output_shape_size, data[2] * GetWidth(input));
              break;
            default:
              return NNACL_INFER_INVALID;
          }
        } else if (shape_tensor->data_type_ == kNumberTypeFloat32) {
          float *data = (float *)(shape_tensor->data_);
          if (data == NULL) {
            return NNACL_INFER_INVALID;
          }
          switch (shape_tensor->format_) {
            case Format_NCHW:
              ShapePush(output_shape, &output_shape_size, data[2] * GetHeight(input));
              ShapePush(output_shape, &output_shape_size, data[3] * GetWidth(input));
              break;
            case Format_NHWC:
              ShapePush(output_shape, &output_shape_size, data[1] * GetHeight(input));
              ShapePush(output_shape, &output_shape_size, data[2] * GetWidth(input));
              break;
            default:
              return NNACL_INFER_INVALID;
          }
        }
        break;
      }
      default: {
        int32_t *data = (int32_t *)(shape_tensor->data_);
        if (data == NULL) {
          return NNACL_INFER_INVALID;
        }
        for (size_t i = 0; i < shape_size; i++) {
          ShapePush(output_shape, &output_shape_size, data[i]);
        }
        break;
      }
    }
  } else if (inputs_size == 1) {
    int new_height = param->new_height_;
    int new_width = param->new_width_;
    ShapePush(output_shape, &output_shape_size, new_height);
    ShapePush(output_shape, &output_shape_size, new_width);
  } else {
    return NNACL_ERR;
  }
  ShapePush(output_shape, &output_shape_size, GetChannel(input));
  SetShapeArray(output, output_shape, output_shape_size);
  return NNACL_OK;
}
