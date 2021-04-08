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

#include "nnacl/infer/fused_batchnorm_infer.h"
#include "nnacl/infer/infer_register.h"

int FusedBatchNormInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                             OpParameter *parameter) {
  for (size_t i = 0; i < inputs_size; i++) {
    if (outputs_size <= i) {
      break;
    }
    SetShapeTensor(outputs[i], inputs[i]);
    SetDataTypeFormat(outputs[i], inputs[i]);
  }
  if (outputs_size > 5) {
    SetDataTypeFormat(outputs[5], inputs[0]);
    outputs[5]->shape_size_ = 1;
    outputs[5]->shape_[0] = 1;
  }
  return 0;
}

REG_INFER(FusedBatchNorm, PrimType_FusedBatchNorm, FusedBatchNormInferShape)
