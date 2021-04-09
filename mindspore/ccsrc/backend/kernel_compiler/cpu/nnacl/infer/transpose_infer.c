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

#include "nnacl/infer/transpose_infer.h"
#include "nnacl/infer/infer_register.h"

bool CheckPermTransFormat(const int *perm, const int *perm_transformat, const size_t size) {
  for (size_t i = 0; i < size; ++i) {
    if (perm[i] != perm_transformat[i]) {
      return false;
    }
  }
  return true;
}

int TransposeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                        OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];

  SetDataTypeFormat(output, input);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  const TensorC *perm_tensor = inputs[1];
  const int32_t *perm_data = (int32_t *)perm_tensor->data_;
  const size_t perms_num = (size_t)perm_tensor->shape_[0];
  if (perm_tensor->shape_size_ == 0) {
    return NNACL_INFER_INVALID;
  }
  if (perms_num != 0 && perm_data == NULL) {
    return NNACL_INFER_INVALID;
  }
  int perm[MAX_SHAPE_SIZE];
  size_t perm_size = 0;
  for (size_t i = 0; i < perms_num; i++) {
    ShapePush(perm, &perm_size, perm_data[i]);
  }
  int out_shape[MAX_SHAPE_SIZE];
  if (input->shape_size_ != 4 && perms_num == 4) {
    for (size_t i = 0; i < input->shape_size_; ++i) {
      out_shape[i] = input->shape_[i];
    }
    SetShapeArray(output, out_shape, input->shape_size_);
    return NNACL_OK;
  }
  const int nchw2nhwc[4] = {0, 2, 3, 1};
  const int nhwc2nchw[4] = {0, 3, 1, 2};
  if (perms_num == 4) {
    if (input->format_ == Format_NCHW && CheckPermTransFormat(perm, nchw2nhwc, perms_num)) {
      output->format_ = Format_NHWC;
    } else if (input->format_ == Format_NHWC && CheckPermTransFormat(perm, nhwc2nchw, perms_num)) {
      output->format_ = Format_NCHW;
    }
  }
  output->shape_size_ = perm_size;
  for (size_t i = 0; i < perm_size; ++i) {
    out_shape[i] = input->shape_[perm[i]];
  }
  if (perm_size == 0) {
    size_t shape_size = input->shape_size_;
    output->shape_size_ = shape_size;
    for (size_t i = 0; i < shape_size; ++i) {
      out_shape[shape_size - i - 1] = input->shape_[i];
    }
  }
  SetShapeArray(output, out_shape, output->shape_size_);
  return NNACL_OK;
}

REG_INFER(Transpose, PrimType_Transpose, TransposeInferShape)
