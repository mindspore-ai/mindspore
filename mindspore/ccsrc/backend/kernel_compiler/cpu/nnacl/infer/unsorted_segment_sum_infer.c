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

#include "nnacl/infer/unsorted_segment_sum_infer.h"
#include "nnacl/infer/infer_register.h"

int UnsortedSegmentSumInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                 size_t outputs_size, OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 3, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
#endif

  TensorC *out = outputs[0];
  const TensorC *x = inputs[0];
  const TensorC *segment_id = inputs[1];
  int num_segments = *(int *)(inputs[2]->data_);
  int output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = 0;
  ShapePush(output_shape, &output_shape_size, num_segments);
  for (int index = segment_id->shape_size_; index < (int)(x->shape_size_); index++) {
    ShapePush(output_shape, &output_shape_size, x->shape_[index]);
  }
  SetShapeArray(out, output_shape, output_shape_size);
  SetDataTypeFormat(out, x);
  return NNACL_OK;
}

REG_INFER(UnsortedSegmentSum, PrimType_UnsortedSegmentSum, UnsortedSegmentSumInferShape)
