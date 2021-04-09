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

#include "nnacl/infer/bias_grad_infer.h"
#include "nnacl/infer/infer_register.h"

int BiasGradInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                       OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  const TensorC *in0 = inputs[0];
  TensorC *out = outputs[0];

  int inshape[MAX_SHAPE_SIZE];
  size_t inshape_size = 0;
  ShapeSet(inshape, &inshape_size, in0->shape_, in0->shape_size_);
  int ndim = inshape_size;
  for (int i = 0; i < ndim - 1; i++) {
    inshape[i] = 1;
  }
  SetDataTypeFormat(out, in0);
  SetShapeArray(out, inshape, inshape_size);

  return NNACL_OK;
}

REG_INFER(BiasAddGrad, PrimType_BiasAddGrad, BiasGradInferShape)
