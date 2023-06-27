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
#include <math.h>
#include <limits.h>
#include "nnacl/infer/infer_register.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/tensor_c_utils.h"

int HandleTwoInputs(const TensorC *const *inputs, ResizeParameter *param) {
  const TensorC *input = inputs[0];
  const TensorC *shape_tensor = inputs[1];
  if (shape_tensor->data_ == NULL) {
    return NNACL_INFER_INVALID;
  }
  int shape_size = GetElementNum(shape_tensor);
  void *origin_data = shape_tensor->data_;
  if (origin_data == NULL) {
    return NNACL_INFER_INVALID;
  }
  switch (shape_size) {
    case 2:
    case 4: {
      int height_index = 0;
      int width_index = 1;
      if (shape_size == 4) {
        height_index = kNHWC_H;
        width_index = kNHWC_W;
      }
      if (shape_tensor->data_type_ == kNumberTypeInt32) {
        int32_t *data = (int32_t *)(origin_data);
        param->new_height_ = data[height_index];
        param->new_width_ = data[width_index];
      } else if (shape_tensor->data_type_ == kNumberTypeFloat32) {
        float *data = (float *)(origin_data);
        NNACL_CHECK_INT_MUL_NOT_OVERFLOW((int)(data[height_index]), GetHeight(input), NNACL_ERRCODE_MUL_OVERFLOW);
        NNACL_CHECK_INT_MUL_NOT_OVERFLOW((int)(data[width_index]), GetWidth(input), NNACL_ERRCODE_MUL_OVERFLOW);
        param->new_height_ = round(data[height_index] * GetHeight(input));
        param->new_width_ = round(data[width_index] * GetWidth(input));
      } else if (shape_tensor->data_type_ == kNumberTypeFloat16) {
        uint16_t *data = (uint16_t *)(shape_tensor->data_);
        float scale_height = ShortToFloat32(data[height_index]);
        float scale_width = ShortToFloat32(data[width_index]);
        param->new_height_ = round(scale_height * GetHeight(input));
        param->new_width_ = round(scale_width * GetWidth(input));
      }
      break;
    }
    case 1: {
      // caffe zoom_factor
      int scale;
      if (shape_tensor->data_type_ == kNumberTypeInt32) {
        int *data = (int *)(origin_data);
        scale = data[0];
      } else {
        return NNACL_ERR;
      }
      NNACL_CHECK_INT_MUL_NOT_OVERFLOW(GetHeight(input) - 1, scale - 1, NNACL_ERRCODE_MUL_OVERFLOW);
      NNACL_CHECK_INT_MUL_NOT_OVERFLOW(GetWidth(input) - 1, scale - 1, NNACL_ERRCODE_MUL_OVERFLOW);
      param->new_height_ = GetHeight(input) + (GetHeight(input) - 1) * (scale - 1);
      param->new_width_ = GetWidth(input) + (GetWidth(input) - 1) * (scale - 1);
      break;
    }
    default: {
      return NNACL_ERR;
    }
  }
  return NNACL_OK;
}

int CalculateNewHeightAndWidth(const TensorC *const *inputs, size_t inputs_size, ResizeParameter *param) {
  if (inputs_size == 2) {
    return HandleTwoInputs(inputs, param);
  } else if (inputs_size == 1) {
  } else {
    return NNACL_ERR;
  }
  return NNACL_OK;
}

int ResizeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  if (input->format_ != Format_NHWC) {
    return NNACL_FORMAT_ERROR;
  }
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  if (input->shape_size_ != 0 && input->shape_size_ != 4) {
    return NNACL_ERR;
  }
  ResizeParameter *param = (ResizeParameter *)parameter;
  NNACL_CHECK_NULL_RETURN_ERR(param);
  int output_shape[MAX_SHAPE_SIZE] = {0};
  size_t output_shape_size = 0;
  ShapePush(output_shape, &output_shape_size, GetBatch(input));
  int ret = CalculateNewHeightAndWidth(inputs, inputs_size, param);
  if (ret == NNACL_OK) {
    ShapePush(output_shape, &output_shape_size, param->new_height_);
    ShapePush(output_shape, &output_shape_size, param->new_width_);
    ShapePush(output_shape, &output_shape_size, GetChannel(input));
    SetShapeArray(output, output_shape, output_shape_size);
  }
  return ret;
}

REG_INFER(Resize, PrimType_Resize, ResizeInferShape)
