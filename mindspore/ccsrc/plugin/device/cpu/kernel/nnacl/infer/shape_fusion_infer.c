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

#include "nnacl/infer/shape_fusion_infer.h"
#include "nnacl/infer/infer_register.h"

int ShapeFusionInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                          OpParameter *parameter) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  MS_CHECK_TRUE_RET(inputs_size == outputs_size + 1, NNACL_INPUT_TENSOR_ERROR);
  const TensorC *in_tensor = inputs[0];
  size_t input_len = in_tensor->shape_size_ + 1;
  for (size_t out_idx = 0; out_idx < outputs_size; out_idx++) {
    TensorC *out_tensor = outputs[out_idx];
    size_t origin_out_size =
      out_tensor->data_ == NULL ? 0 : (out_tensor->shape_size_ == 0 ? 1 : (size_t)out_tensor->shape_[0]);
    out_tensor->data_type_ = kNumberTypeInt32;
    out_tensor->format_ = in_tensor->format_;
    if (!InferFlag(inputs, inputs_size)) {
      return NNACL_INFER_INVALID;
    }

    // calculate output tensor data.
    const TensorC *matrix_tensor = inputs[out_idx + 1];
    if (matrix_tensor->shape_size_ == 1) {
      out_tensor->shape_size_ = 0;
      out_tensor->shape_[0] = 0;
    } else {
      out_tensor->shape_size_ = 1;
      out_tensor->shape_[0] = (int)(matrix_tensor->shape_[0]);
    }
    size_t out_size = out_tensor->shape_size_ == 0 ? 1 : (size_t)(out_tensor->shape_[0]);
    if (out_size != origin_out_size && out_tensor->data_ != NULL) {
      free(out_tensor->data_);
      out_tensor->data_ = NULL;
    }
    size_t matrix_data_size = input_len * out_size * sizeof(float);
    float *matrix_data = (float *)(malloc(matrix_data_size));
    NNACL_CHECK_NULL_RETURN_ERR(matrix_data);
    if (matrix_tensor->data_type_ == kNumberTypeFloat32 || matrix_tensor->data_type_ == kNumberTypeFloat) {
      memcpy(matrix_data, matrix_tensor->data_, matrix_data_size);
#ifdef ENABLE_FP16
    } else if (matrix_tensor->data_type_ == kNumberTypeFloat16) {
      for (size_t i = 0; i < input_len * out_size; i++) {
        matrix_data[i] = (float)(((float16_t *)(matrix_tensor->data_))[i]);
      }
#endif
    } else {
      free(matrix_data);
      return NNACL_ERR;
    }
    if (out_tensor->data_ == NULL) {
      out_tensor->data_ = malloc(out_size * sizeof(int));
    }
    int *data = (int *)out_tensor->data_;
    if (data == NULL) {
      free(matrix_data);
      return NNACL_ERR;
    }
    memset(data, 0, out_size * sizeof(int));
    for (size_t i = 0; i < out_size; i++) {
      for (size_t j = 0; j < input_len - 1; j++) {
        data[i] += (int)(in_tensor->shape_[j] * matrix_data[i * input_len + j]);
      }
      data[i] += (int)(matrix_data[i * input_len + input_len - 1]);
    }
    free(matrix_data);
  }
  return NNACL_OK;
}

REG_INFER(ShapeFusion, PrimType_Inner_ShapeFusion, ShapeFusionInferShape)
