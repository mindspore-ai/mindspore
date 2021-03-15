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

#include "nnacl/infer/gather_infer.h"
#include "nnacl/infer/infer_register.h"

int GatherInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  if (inputs_size < 2 || outputs_size != 1) {
    return NNACL_ERR;
  }
  const TensorC *input = inputs[0];
  const TensorC *indices = inputs[1];
  TensorC *output = outputs[0];
  output->data_type_ = input->data_type_;
  if (parameter->quant_type_ == QuantType_WeightQuant) {
    output->data_type_ = kNumberTypeFloat32;
  }
  output->format_ = input->format_;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  int axis = *((int *)inputs[2]->data_);
  if (axis < 0) {
    axis += input->shape_size_;
  }
  int indices_shape[MAX_SHAPE_SIZE];
  size_t indices_shape_size = 0;
  ShapeSet(indices_shape, &indices_shape_size, indices->shape_, indices->shape_size_);
  int indices_rank = indices_shape_size;
  int in_shape[MAX_SHAPE_SIZE];
  size_t in_shape_size = 0;
  ShapeSet(in_shape, &in_shape_size, input->shape_, input->shape_size_);
  int in_rank = in_shape_size;
  if (in_rank < axis + 1) {
    return NNACL_ERR;
  }
  int out_shape[MAX_SHAPE_SIZE];
  size_t out_shape_size = 0;
  ShapeSet(out_shape, &out_shape_size, in_shape, in_shape_size);
  ShapeErase(out_shape, &out_shape_size, axis);
  for (int i = indices_rank - 1; i >= 0; --i) {
    ShapeInsert(out_shape, &out_shape_size, axis, indices_shape[i]);
  }
  SetShapeArray(output, out_shape, out_shape_size);
  return NNACL_OK;
}

REG_INFER(Gather, PrimType_Gather, GatherInferShape)
