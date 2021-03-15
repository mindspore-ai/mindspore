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
#include "nnacl/infer/layer_norm_grad_infer.h"
#include "nnacl/infer/common_infer.h"
#include "nnacl/fp32_grad/layernormgrad_parameter.h"
#include "nnacl/infer/infer_register.h"

int LayerNormGradInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                            OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 5, 3);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  LayerNormGradParameter *param = (LayerNormGradParameter *)parameter;
  const TensorC *input_x = inputs[0];
  TensorC *output_dx = outputs[0];
  TensorC *output_dg = outputs[1];
  TensorC *output_db = outputs[2];
  SetDataTypeFormat(output_dx, input_x);
  SetDataTypeFormat(output_dg, input_x);
  SetDataTypeFormat(output_db, input_x);
  SetShapeTensor(output_dx, input_x);
  int begin_params_axis = param->begin_params_axis_;
  if (param->begin_params_axis_ < 0) {
    begin_params_axis += input_x->shape_size_;
  }
  int size = 0;
  for (int i = begin_params_axis; i < input_x->shape_size_; i++) {
    output_dg->shape_[size] = input_x->shape_[i];
    output_db->shape_[size] = input_x->shape_[i];
    size++;
  }
  output_db->shape_size_ = size;
  output_dg->shape_size_ = size;
  return NNACL_OK;
}

REG_INFER(LayerNormGrad, PrimType_LayerNormGrad, LayerNormGradInferShape)
