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

int HandleTwoInputs(const TensorC *const *inputs, ResizeParameter *param) {
  const TensorC *input = inputs[0];
  const TensorC *shape_tensor = inputs[1];
  if (shape_tensor->data_ == NULL) {
    return NNACL_INFER_INVALID;
  }
  int shape_size = GetElementNum(shape_tensor);
  switch (shape_size) {
    case 4: {
      if (shape_tensor->data_type_ == kNumberTypeInt32) {
        int32_t *data = (int32_t *)(shape_tensor->data_);
        if (data == NULL) {
          return NNACL_INFER_INVALID;
        }
        if (GetElementNum(shape_tensor) < 4) {
          return NNACL_ERR;
        }
        param->new_height_ = data[1];
        param->new_width_ = data[2];
      } else if (shape_tensor->data_type_ == kNumberTypeFloat32) {
        float *data = (float *)(shape_tensor->data_);
        if (data == NULL) {
          return NNACL_INFER_INVALID;
        }

        MS_CHECK_INT_MUL_NOT_OVERFLOW((int)(data[1]), GetHeight(input), NNACL_ERRCODE_MUL_OVERFLOW);
        MS_CHECK_INT_MUL_NOT_OVERFLOW((int)(data[2]), GetWidth(input), NNACL_ERRCODE_MUL_OVERFLOW);
        param->new_height_ = round(data[1] * GetHeight(input));
        param->new_width_ = round(data[2] * GetWidth(input));
      } else if (shape_tensor->data_type_ == kNumberTypeFloat16) {
        uint16_t *data = (uint16_t *)(shape_tensor->data_);
        if (data == NULL) {
          return NNACL_INFER_INVALID;
        }

        float scale_height = ShortToFloat32(data[1]);
        float scale_width = ShortToFloat32(data[2]);

        MS_CHECK_INT_MUL_NOT_OVERFLOW(scale_height, GetHeight(input), NNACL_ERRCODE_MUL_OVERFLOW);
        MS_CHECK_INT_MUL_NOT_OVERFLOW(scale_width, GetWidth(input), NNACL_ERRCODE_MUL_OVERFLOW);
        param->new_height_ = round(scale_height * GetHeight(input));
        param->new_width_ = round(scale_width * GetWidth(input));
      }
      break;
    }
    case 2: {
      int32_t *data = (int32_t *)(shape_tensor->data_);
      if (data == NULL) {
        return NNACL_INFER_INVALID;
      }
      param->new_height_ = data[0];
      param->new_width_ = data[1];
      break;
    }
    case 1: {
      // caffe zoom_factor
      int scale;
      if (shape_tensor->data_type_ == kNumberTypeInt32) {
        int *data = (int *)(shape_tensor->data_);
        if (data == NULL) {
          return NNACL_INFER_INVALID;
        }
        scale = data[0];
      } else {
        return NNACL_ERR;
      }
      MS_CHECK_INT_MUL_NOT_OVERFLOW(GetHeight(input) - 1, scale - 1, NNACL_ERRCODE_MUL_OVERFLOW);
      MS_CHECK_INT_MUL_NOT_OVERFLOW(GetWidth(input) - 1, scale - 1, NNACL_ERRCODE_MUL_OVERFLOW);
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
