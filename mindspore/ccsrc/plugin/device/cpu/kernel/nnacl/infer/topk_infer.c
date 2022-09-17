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

#include "nnacl/infer/topk_infer.h"
#include "nnacl/infer/infer_register.h"

int TopKInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                   OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 2);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output0 = outputs[0];
  TensorC *output1 = outputs[1];
  SetDataTypeFormat(output0, input);
  output1->data_type_ = kNumberTypeInt32;
  output1->format_ = input->format_;
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  const TensorC *input_k_tensor = inputs[1];
  if (input_k_tensor->data_ == NULL) {
    return NNACL_INFER_INVALID;
  }

  TopkParameter *param = (TopkParameter *)parameter;
  param->k_ = ((int32_t *)input_k_tensor->data_)[0];

  if (input->shape_size_ > MAX_SHAPE_SIZE) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = 0;
  ShapeSet(out_shape, &out_shape_size, input->shape_, input->shape_size_);
  if (out_shape_size < 1) {
    return NNACL_ERR;
  }
  if (param->axis_ < 0) {
    param->axis_ += (int)out_shape_size;
  }
  if (param->axis_ < 0 || (size_t)param->axis_ >= out_shape_size) {
    return NNACL_ERR;
  }
  out_shape[(size_t)param->axis_] = param->k_;

  SetShapeArray(output0, out_shape, out_shape_size);
  SetShapeArray(output1, out_shape, out_shape_size);
  return NNACL_OK;
}

REG_INFER(TopK, PrimType_TopKFusion, TopKInferShape)
