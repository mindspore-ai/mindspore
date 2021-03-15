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

#include "nnacl/infer/gather_nd_infer.h"
#include "nnacl/infer/infer_register.h"

int GatherNdInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                       OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  const TensorC *indices = inputs[1];
  TensorC *output = outputs[0];

  SetDataTypeFormat(output, input);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  int in_rank = input->shape_size_;
  int indices_rank = indices->shape_size_;
  if (indices->shape_[indices_rank - 1] > in_rank) {
    return NNACL_OK;
  }
  int i = 0;
  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = 0;
  for (i = 0; i < indices_rank - 1; ++i) {
    ShapePush(out_shape, &out_shape_size, indices->shape_[i]);
  }
  for (i = indices->shape_[indices_rank - 1]; i < in_rank; ++i) {
    ShapePush(out_shape, &out_shape_size, input->shape_[i]);
  }
  SetShapeArray(output, out_shape, out_shape_size);
  return NNACL_OK;
}

REG_INFER(GatherNd, PrimType_GatherNd, GatherNdInferShape)
