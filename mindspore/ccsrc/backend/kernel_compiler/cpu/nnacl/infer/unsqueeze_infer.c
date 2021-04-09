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

#include "nnacl/infer/unsqueeze_infer.h"
#include "nnacl/infer/infer_register.h"

int UnsqueezeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                        OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
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

  UnSqueezeParameter *param = (UnSqueezeParameter *)parameter;
  int in_rank = input->shape_size_;
  int dim_rank = param->num_dim_;
  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = 0;
  if (dim_rank == 0) {
    for (size_t i = 0; i < input->shape_size_; i++) {
      if (input->shape_[i] != 1) {
        ShapePush(out_shape, &out_shape_size, input->shape_[i]);
      }
    }
  } else {
    int sz = in_rank + dim_rank;
    size_t in_itr = 0;
    size_t ax_itr = 0;
    for (size_t i = 0; i < sz; i++) {
      if (ax_itr < dim_rank && param->dims_[ax_itr] == (int)(i)) {
        ShapePush(out_shape, &out_shape_size, 1);
        ax_itr++;
      } else if (ax_itr < dim_rank && param->dims_[ax_itr] + sz == i) {
        ShapePush(out_shape, &out_shape_size, 1);
        ax_itr++;
      } else {
        ShapePush(out_shape, &out_shape_size, input->shape_[in_itr]);
        in_itr++;
      }
    }
  }
  SetShapeArray(output, out_shape, out_shape_size);
  return NNACL_OK;
}

REG_INFER(Unsqueeze, PrimType_Unsqueeze, UnsqueezeInferShape)
