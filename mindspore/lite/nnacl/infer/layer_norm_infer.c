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

#include "nnacl/infer/layer_norm_infer.h"
#include "nnacl/infer/infer_register.h"

int LayerNormInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                        OpParameter *parameter) {
#ifdef Debug
  if ((inputs_size != 1 && inputs_size != 3) || (outputs_size != 1 && outputs_size != 3)) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);

  LayerNormParameter *param = (LayerNormParameter *)parameter;
  if (!param->op_parameter_.infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  param->begin_norm_axis_ =
    param->begin_norm_axis_ < 0 ? param->begin_norm_axis_ + input->shape_size_ : param->begin_norm_axis_;
  SetShapeTensor(output, input);
  // take care of other outputs
  if (outputs_size == 3) {
    TensorC *output_mean = outputs[1];
    TensorC *output_var = outputs[2];
    SetDataTypeFormat(output_mean, input);
    SetDataTypeFormat(output_var, input);
    int size = 0;
    for (; size < param->begin_norm_axis_; size++) {
      output_mean->shape_[size] = input->shape_[size];
      output_var->shape_[size] = input->shape_[size];
    }
    output_mean->shape_size_ = size;
    output_var->shape_size_ = size;
  }

  return NNACL_OK;
}

REG_INFER(LayerNormFusion, PrimType_LayerNormFusion, LayerNormInferShape)
