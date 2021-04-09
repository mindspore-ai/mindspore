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

#include "nnacl/infer/merge_infer.h"
#include <string.h>
#include "nnacl/infer/infer_register.h"

bool MergeAbleToInfer(const TensorC *const *inputs, size_t inputs_size) {
  for (size_t i = 0; i < inputs_size; i++) {
    if (!inputs[i]->is_ready_) {
      return false;
    }
  }
  return true;
}

int MergeInfer(TensorC **inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size) {
  for (size_t i = 0; i < inputs_size; i++) {
    outputs[i] = inputs[i];
    inputs[i] = NULL;
  }
  return NNACL_OK;
}

int MergeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                    OpParameter *parameter) {
#ifdef Debug
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i] == NULL) {
      return NNACL_NULL_PTR;
    }
  }
  if (inputs_size != 2 * outputs_size) {
    return NNACL_ERR;
  }
#endif

  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  const TensorC *const *left_part_inputs = inputs;
  size_t left_part_inputs_size = inputs_size / 2;

  const TensorC *const *right_part_inputs = inputs + left_part_inputs_size;
  size_t right_part_inputs_size = inputs_size / 2;

  if (MergeAbleToInfer(left_part_inputs, left_part_inputs_size)) {
    return MergeInfer((TensorC **)left_part_inputs, left_part_inputs_size, outputs, outputs_size);
  }

  if (MergeAbleToInfer(right_part_inputs, right_part_inputs_size)) {
    return MergeInfer((TensorC **)right_part_inputs, right_part_inputs_size, outputs, outputs_size);
  }

  return NNACL_INFER_INVALID;
}

REG_INFER(Merge, PrimType_Merge, MergeInferShape)
