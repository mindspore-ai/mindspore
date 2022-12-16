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

#include "nnacl/infer/scatter_nd_infer.h"
#include "nnacl/infer/infer_register.h"

int ScatterNdInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                        OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 3, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *shape = inputs[THIRD_INPUT];
  if (shape->data_ == NULL) {
    return NNACL_INFER_INVALID;
  }
  const TensorC *update = inputs[SECOND_INPUT];
  TensorC *output = outputs[0];

  SetDataTypeFormat(output, update);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  int *shape_data = (int *)(shape->data_);
  MS_CHECK_TRUE_RET(GetElementNum(shape) <= MAX_SHAPE_SIZE, NNACL_ERR);
  SetShapeArray(output, shape_data, (size_t)GetElementNum(shape));
  return NNACL_OK;
}

REG_INFER(ScatterNd, PrimType_ScatterNd, ScatterNdInferShape)
