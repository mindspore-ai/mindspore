/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "nnacl/infer/transpose_infer.h"
#include "nnacl/infer/infer_register.h"

bool CheckPermTransFormat(const int *perm, const int *perm_transformat, const int size) {
  for (int i = 0; i < size; ++i) {
    if (perm[i] != perm_transformat[i]) {
      return false;
    }
  }
  return true;
}

int SetOutputShape(int perms_num, const TensorC *input, TensorC *output, const int *perm, size_t perm_size,
                   int *out_shape) {
  // set output shape
  size_t in_shape_size = input->shape_size_;
  output->shape_size_ = in_shape_size;
  if (perm_size == 0) {
    for (size_t i = 0; i < in_shape_size; ++i) {
      out_shape[in_shape_size - i - 1] = input->shape_[i];
    }
  } else if (perm_size != in_shape_size) {
    for (size_t i = 0; i < in_shape_size; ++i) {
      out_shape[i] = input->shape_[i];
    }
  } else {
    output->shape_size_ = perm_size;
    for (size_t i = 0; i < perm_size; ++i) {
      if (perm[i] >= input->shape_size_) {
        return NNACL_ERR;
      } else {
        out_shape[i] = input->shape_[perm[i]];
      }
    }
  }
  return NNACL_OK;
}

int TransposeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                        OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];

  SetDataTypeFormat(output, input);
  const TensorC *perm_tensor = inputs[1];
  const int32_t *perm_data = (int32_t *)perm_tensor->data_;
  const int perms_num = perm_tensor->shape_[0];
  MS_CHECK_TRUE_RET(perm_tensor->shape_size_ != 0, NNACL_INFER_INVALID);
  if (perms_num != 0 && perm_data == NULL) {
    return NNACL_INFER_INVALID;
  }
  int perm[MAX_TRANSPOSE_DIM_SIZE] = {0};
  size_t perm_size = 0;
  for (int i = 0; i < perms_num; i++) {
    MS_CHECK_TRUE_RET(perm_data[i] < perms_num, NNACL_ERR);
    ShapePush(perm, &perm_size, perm_data[i]);
  }
  if (perms_num == PERM_NUM_FOUR) {
    const int nchw2nhwc[4] = {0, 2, 3, 1};
    const int nhwc2nchw[4] = {0, 3, 1, 2};
    const int trans3d[3] = {0, 2, 1};
    if (input->format_ == Format_NCHW && CheckPermTransFormat(perm, nchw2nhwc, perms_num)) {
      output->format_ = Format_NHWC;
    } else if ((input->format_ == Format_NHWC || input->format_ == Format_KHWC) &&
               CheckPermTransFormat(perm, nhwc2nchw, perms_num)) {
      output->format_ = Format_NCHW;
    }
    // though the perm is 4d in default, the input can be a 3d tensor. The op implementation must be adapted to this.
    if (input->shape_size_ == 3) {
      ShapeSet(perm, &perm_size, trans3d, 3);
    }
  }
  if (perms_num == PERM_NUM_THREE && perm[0] == 0 && perm[1] == 2) {
    output->format_ = input->format_ == Format_NCHW ? Format_NHWC : Format_NCHW;
  }
  if (parameter->quant_type_ == QuantType_QUANT_WEIGHT) {
    output->data_type_ = kNumberTypeFloat32;
  }
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  // set output shape
  int out_shape[MAX_TRANSPOSE_DIM_SIZE] = {0};
  SetOutputShape(perms_num, input, output, perm, perm_size, out_shape);
  SetShapeArray(output, out_shape, output->shape_size_);
  return NNACL_OK;
}

REG_INFER(Transpose, PrimType_Transpose, TransposeInferShape)
