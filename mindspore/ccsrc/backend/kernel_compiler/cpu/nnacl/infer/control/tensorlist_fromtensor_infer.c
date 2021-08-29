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

#include "nnacl/infer/control/tensorlist_fromtensor_infer.h"
#include "nnacl/infer/infer_register.h"

int TensorListFromTensorInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                   size_t outputs_size, OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  TensorListC *output = (TensorListC *)(outputs[0]);
  const TensorC *input0 = inputs[0];
  output->data_type_ = kObjectTypeTensorType;
  output->format_ = Format_NHWC;
  output->tensors_data_type_ = input0->data_type_;

  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  if (input0->shape_size_ < 1) {
    return NNACL_ERR;
  }
  int dim0 = input0->shape_[0];
  if (dim0 < 0) {
    return NNACL_ERR;
  }
  const TensorC *input1 = inputs[1];
  if (input1->data_ == NULL) {
    return NNACL_NULL_PTR;
  }
  int *ele_shape_ptr = (int *)(input1->data_);
  NNACL_CHECK_NULL_RETURN_ERR(ele_shape_ptr);
  vvector tensor_shape;
  tensor_shape.size_ = (size_t)(dim0);
  tensor_shape.shape_ = (int **)malloc(tensor_shape.size_ * sizeof(int *));
  if (tensor_shape.shape_ == NULL) {
    return NNACL_NULL_PTR;
  }
  tensor_shape.shape_size_ = (int *)malloc(tensor_shape.size_ * sizeof(int));
  if (tensor_shape.shape_size_ == NULL) {
    free(tensor_shape.shape_);
    return NNACL_NULL_PTR;
  }
  for (int i = 0; i < dim0; i++) {
    tensor_shape.shape_[i] = (int *)(input0->shape_ + 1);
    tensor_shape.shape_size_[i] = (int)(input0->shape_size_) - 1;
  }

  ShapeSet(output->element_shape_, &(output->element_shape_size_), ele_shape_ptr, GetElementNum(input1));
  output->element_num_ = (size_t)(dim0);
  int ret = MallocTensorListData(output, input0->data_type_, &tensor_shape);
  if (ret != NNACL_OK) {
    free(tensor_shape.shape_);
    free(tensor_shape.shape_size_);
    return NNACL_ERR;
  }
  free(tensor_shape.shape_);
  free(tensor_shape.shape_size_);
  return NNACL_OK;
}

REG_INFER(TensorListFromTensor, PrimType_TensorListFromTensor, TensorListFromTensorInferShape)
