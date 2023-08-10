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

int GetAndCheckPerm(const TensorC *perm_tensor, const int perms_num, int *perm, size_t *perm_size) {
  if (perms_num >= MAX_TRANSPOSE_DIM_SIZE) {
    return NNACL_TRANSPOSE_PERM_DIMS_INVALID;
  }

  int ret = GetInt32DataFromTensor(perm_tensor, perm, perm_size);
  if (ret != NNACL_OK) {
    return ret;
  }
  for (size_t i = 0; i < *perm_size; i++) {
    NNACL_CHECK_TRUE_RET(perm[i] < perms_num, NNACL_ERR);
  }
  return NNACL_OK;
}

void Handle4DPerm(const TensorC *input, TensorC *output, int *perm, size_t *perm_size) {
  const int nchw2nhwc[4] = {Index0, Index2, Index3, Index1};
  const int nhwc2nchw[4] = {Index0, Index3, Index1, Index2};
  const int trans3d[3] = {Index0, Index2, Index1};
  if (input->format_ == Format_NCHW && CheckPermTransFormat(perm, nchw2nhwc, PERM_NUM_FOUR)) {
    output->format_ = Format_NHWC;
  } else if ((input->format_ == Format_NHWC || input->format_ == Format_KHWC) &&
             CheckPermTransFormat(perm, nhwc2nchw, PERM_NUM_FOUR)) {
    output->format_ = Format_NCHW;
  }
  // though the perm is 4d in default, the input can be a 3d tensor. The op implementation must be adapted to this.
  if (input->shape_size_ == DIMENSION_3D) {
    ShapeSet(perm, perm_size, trans3d, DIMENSION_3D);
  }
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
  if (perm_tensor == NULL) {
    return NNACL_INFER_INVALID;
  }
  NNACL_CHECK_TRUE_RET(perm_tensor->shape_size_ == 1, NNACL_INFER_INVALID);
  const int perms_num = perm_tensor->shape_[0];
  if (perms_num != 0 && perm_tensor->data_ == NULL) {
    return NNACL_INFER_INVALID;
  }
  TransposeParameter *transpose_param = (TransposeParameter *)parameter;
  transpose_param->perm_size_ = perms_num;
  int perm[MAX_TRANSPOSE_DIM_SIZE] = {0};
  size_t perm_size = 0;
  int ret = GetAndCheckPerm(perm_tensor, perms_num, perm, &perm_size);
  if (ret != NNACL_OK) {
    return ret;
  }

  if (perms_num == PERM_NUM_FOUR) {
    Handle4DPerm(input, output, perm, &perm_size);
  }
  int kPermIndex0 = 0;
  int kPermIndex2 = 2;
  if (perms_num == PERM_NUM_THREE && perm[0] == kPermIndex0 && perm[1] == kPermIndex2) {
    output->format_ = input->format_ == Format_NCHW ? Format_NHWC : Format_NCHW;
  }
  if (parameter->quant_type_ == Quant_QuantWeight) {
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
