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

int LayerNormInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                        OpParameter *parameter) {
  int check_ret = CheckAugmentNullSizeInputTwo(inputs, inputs_size, outputs, outputs_size, parameter, 1, 3, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  LayerNormParameter *param = (LayerNormParameter *)parameter;
  if (!param->op_parameter_.infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  SetShapeTensor(output, input);
  return NNACL_OK;
}
