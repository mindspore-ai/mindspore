/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use tensor file except in compliance with the License.
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
#include "nnacl/infer/common_infer.h"
#include <stdlib.h>
#include <string.h>
#include "nnacl/infer/infer_register.h"

int MallocTensorListData(TensorListC *tensor_list, TypeIdC dtype, vvector *tensor_shape) {
  // This function will create a new tensors_
  // Your must to set shape(param2: tensor_shape) and data_type_(tensors_data_type_ = param1: dtype) of each tensor in
  // tensors_. After that, you need to call function:MallocData to malloc data buf of each tensor in tensors_.

  if (tensor_list->element_num_ == 0) {
    return NNACL_OK;
  }
  if (((size_t)(tensor_list->element_num_)) != tensor_shape->size_) {
    return NNACL_ERR;
  }
  tensor_list->tensors_data_type_ = dtype;
  tensor_list->tensors_ = (TensorC *)malloc(tensor_list->element_num_ * sizeof(TensorC));  // free in infer_manager
  if (tensor_list->tensors_ == NULL) {
    return NNACL_NULL_PTR;
  }
  memset(tensor_list->tensors_, 0, tensor_list->element_num_ * sizeof(TensorC));
  for (int i = 0; i < tensor_list->element_num_; ++i) {
    tensor_list->tensors_[i].format_ = Format_NHWC;
    tensor_list->tensors_[i].data_type_ = dtype;
    ShapeSet(tensor_list->tensors_[i].shape_, &(tensor_list->tensors_[i].shape_size_), tensor_shape->shape_[i],
             tensor_shape->shape_size_[i]);
  }
  return NNACL_OK;
}

int TensorListMergeShape(int *element_shape, size_t *element_shape_size, const int *tmp, size_t tmp_size) {
  if (*element_shape_size >= 255 || element_shape[0] == -1) {
    ShapeSet(element_shape, element_shape_size, tmp, tmp_size);
    return NNACL_OK;
  }
  if (*element_shape_size != tmp_size) {
    return NNACL_ERR;
  }
  for (size_t j = 0; j < tmp_size; ++j) {
    if (element_shape[j] >= 0 && tmp[j] >= 0 && element_shape[j] != tmp[j]) {
      return NNACL_ERR;
    }
    element_shape[j] = element_shape[j] >= 0 ? element_shape[j] : tmp[j];
  }
  return NNACL_OK;
}

bool TensorListIsFullyDefined(int *shape, size_t shape_size) {
  for (size_t i = 0; i < shape_size; ++i) {
    if (shape[i] < 0) {
      return false;
    }
  }
  return true;
}

int CheckAugmentNull(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  for (size_t i = 0; i < inputs_size; i++) {
    if (inputs[i] == NULL) {
      return NNACL_NULL_PTR;
    }
  }
  for (size_t i = 0; i < outputs_size; i++) {
    if (outputs[i] == NULL) {
      return NNACL_NULL_PTR;
    }
  }
  if (parameter == NULL) {
    return NNACL_NULL_PTR;
  }
  return NNACL_OK;
}

int CheckAugmentNullSize(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                         OpParameter *parameter, size_t inputs_size_obj, size_t outputs_size_obj) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret == NNACL_NULL_PTR) {
    return NNACL_NULL_PTR;
  }
  if (inputs_size != inputs_size_obj || outputs_size != outputs_size_obj) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  return NNACL_OK;
}

int CheckAugmentNullSizeInputTwo(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                 size_t outputs_size, OpParameter *parameter, size_t inputs_size_obj_0,
                                 size_t inputs_size_obj_1, size_t outputs_size_obj) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret == NNACL_NULL_PTR) {
    return NNACL_NULL_PTR;
  }
  if ((inputs_size != inputs_size_obj_0 && inputs_size != inputs_size_obj_1) || outputs_size != outputs_size_obj) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  return NNACL_OK;
}

int CheckAugmentNullInputSize(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                              OpParameter *parameter, size_t inputs_size_obj) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret == NNACL_NULL_PTR) {
    return NNACL_NULL_PTR;
  }
  if (inputs_size != inputs_size_obj) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  return NNACL_OK;
}

int CheckAugmentNullOutputSize(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                               OpParameter *parameter, size_t outputs_size_obj) {
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret == NNACL_NULL_PTR) {
    return NNACL_NULL_PTR;
  }
  if (outputs_size != outputs_size_obj) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  return NNACL_OK;
}

int SetShapeTensor(TensorC *dst, const TensorC *src) {
  for (size_t i = 0; i < src->shape_size_; i++) {
    dst->shape_[i] = src->shape_[i];
  }
  dst->shape_size_ = src->shape_size_;
  return NNACL_OK;
}

int SetShapeArray(TensorC *dst, int *src, size_t src_size) {
  for (size_t i = 0; i < src_size; i++) {
    dst->shape_[i] = src[i];
  }
  dst->shape_size_ = src_size;
  return NNACL_OK;
}

void SetDataTypeFormat(TensorC *dst, const TensorC *src) {
  dst->format_ = src->format_;
  dst->data_type_ = src->data_type_;
}

int GetBatch(const TensorC *tensor) {
  if (tensor->shape_size_ != 4 && tensor->shape_size_ != 2) {
    return -1;
  }
  switch (tensor->format_) {
    case Format_NHWC:
    case Format_NHWC4:
    case Format_NCHW:
    case Format_NC4HW4:
    case Format_KCHW:
    case Format_KHWC:
    case Format_NC:
    case Format_NC4:
      return tensor->shape_[0];
    case Format_HWCK:
    case Format_CHWK:
      return tensor->shape_[3];
    case Format_HWKC:
      return tensor->shape_[2];
    case Format_CKHW:
      return tensor->shape_[1];
    default:
      return -1;
  }
}
int GetHeight(const TensorC *tensor) {
  if (tensor->shape_size_ != 4 && tensor->shape_size_ != 2) {
    return -1;
  }
  switch (tensor->format_) {
    case Format_NCHW:
    case Format_KCHW:
    case Format_CKHW:
      return tensor->shape_[2];
    case Format_NHWC:
    case Format_NHWC4:
    case Format_NC4HW4:
    case Format_KHWC:
    case Format_CHWK:
      return tensor->shape_[1];
    case Format_HWCK:
    case Format_HWKC:
    case Format_HW:
    case Format_HW4:
      return tensor->shape_[0];
    default:
      return -1;
  }
}
int GetWidth(const TensorC *tensor) {
  if (tensor->shape_size_ != 4 && tensor->shape_size_ != 2) {
    return -1;
  }
  switch (tensor->format_) {
    case Format_NCHW:
    case Format_KCHW:
    case Format_CKHW:
      return tensor->shape_[3];
    case Format_KHWC:
    case Format_NHWC:
    case Format_NHWC4:
    case Format_NC4HW4:
    case Format_CHWK:
      return tensor->shape_[2];
    case Format_HWCK:
    case Format_HWKC:
    case Format_HW:
    case Format_HW4:
      return tensor->shape_[1];
    default:
      return -1;
  }
}
int GetChannel(const TensorC *tensor) {
  if (tensor->shape_size_ != 4 && tensor->shape_size_ != 2) {
    return -1;
  }
  switch (tensor->format_) {
    case Format_NCHW:
    case Format_KCHW:
    case Format_NC:
    case Format_NC4:
      return tensor->shape_[1];
    case Format_HWCK:
      return tensor->shape_[2];
    case Format_HWKC:
    case Format_NHWC:
    case Format_NHWC4:
    case Format_NC4HW4:
    case Format_KHWC:
      return tensor->shape_[3];
    case Format_CKHW:
    case Format_CHWK:
      return tensor->shape_[0];
    default:
      return -1;
  }
}

int GetElementNum(const TensorC *tensor) {
  if (tensor->shape_size_ == 0) {
    return 1;  // scalar mode
  }
  int res = 1;
  for (size_t i = 0; i < tensor->shape_size_; i++) {
    res = res * tensor->shape_[i];
  }
  return res;
}
int GetDimensionSize(const TensorC *tensor, const size_t index) {
  int dim_size = -1;
  if (index < tensor->shape_size_) {
    dim_size = tensor->shape_[index];
  }
  return dim_size;
}

int ShapeSet(int *dst_shape, size_t *dst_shape_size, const int *src_shape, size_t src_shape_size) {
  for (size_t i = 0; i < src_shape_size; i++) {
    dst_shape[i] = src_shape[i];
  }
  *dst_shape_size = src_shape_size;
  return NNACL_OK;
}

int ShapePush(int *shape, size_t *shape_size, int value) {
  shape[*shape_size] = value;
  *shape_size = *shape_size + 1;
  return NNACL_OK;
}

int ShapeInsert(int *shape, size_t *shape_size, int index, int value) {
  if (index < 0 || index > *shape_size) {
    return NNACL_ERR;
  }
  for (int i = *shape_size; i > index; i--) {
    shape[i] = shape[i - 1];
  }
  shape[index] = value;
  *shape_size = *shape_size + 1;
  return NNACL_OK;
}

int ShapeErase(int *shape, size_t *shape_size, int index) {
  if (index < 0 && index >= *shape_size) {
    return NNACL_ERR;
  }

  for (int i = index; i < *shape_size - 1; i++) {
    shape[i] = shape[i + 1];
  }
  *shape_size = *shape_size - 1;
  return NNACL_OK;
}

bool ShapeEqual(const int *shape0, size_t shape0_size, const int *shape1, size_t shape1_size) {
  if (shape0_size != shape1_size) {
    return false;
  }
  for (int i = 0; i < shape0_size; i++) {
    if (shape0[i] != shape1[i]) {
      return false;
    }
  }
  return true;
}

void iswap(int *a, int *b) {
  int tmp = *a;
  *a = *b;
  *b = tmp;
}

int imin(int a, int b) { return a > b ? b : a; }

int imax(int a, int b) { return a < b ? b : a; }

// input == output completely refer to
// 1. zeros_like
int CommonInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  if (parameter == NULL || inputs[0] == NULL || outputs[0] == NULL) {
    return NNACL_NULL_PTR;
  }
  SetDataTypeFormat(outputs[0], inputs[0]);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  SetShapeTensor(outputs[0], inputs[0]);
  return NNACL_OK;
}

int FftInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                  OpParameter *parameter) {
  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  output->data_type_ = kNumberTypeFloat32;
  output->format_ = input->format_;
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }
  int input_shape[MAX_SHAPE_SIZE];
  size_t input_shape_size = 0;
  ShapeSet(input_shape, &input_shape_size, input->shape_, input->shape_size_);
  input_shape_size--;
  SetShapeArray(output, input_shape, input_shape_size);
  return NNACL_OK;
}

int VectorCInit(VectorC *vc, size_t per_malloc_size) {
  if (per_malloc_size == 0) {
    return NNACL_ERR;
  }
  vc->data_ = (int *)malloc(per_malloc_size * sizeof(int));
  if (vc->data_ == NULL) {
    return NNACL_ERR;
  }
  vc->size_ = 0;
  vc->max_size_ = per_malloc_size;
  vc->per_malloc_size_ = per_malloc_size;
  return NNACL_OK;
}

void VectorCSet(VectorC *vc, const int *src_shape, size_t src_shape_size) {
  if (src_shape_size == 0) {
    vc->size_ = 0;
  } else {
    free(vc->data_);
    vc->max_size_ = (src_shape_size / vc->per_malloc_size_ + 1) * vc->per_malloc_size_;
    vc->data_ = (int *)malloc(sizeof(int) * vc->max_size_);
    for (size_t i = 0; i < src_shape_size; i++) {
      vc->data_[i] = src_shape[i];
    }
    vc->size_ = src_shape_size;
  }
}

void VectorCPush(VectorC *vc, int value) {
  if (vc->size_ + 1 > vc->max_size_) {
    int *tmp = (int *)malloc(vc->per_malloc_size_ * sizeof(int) + vc->max_size_ * sizeof(int));
    memcpy(tmp, vc->data_, vc->size_ * sizeof(int));
    free(vc->data_);
    vc->data_ = tmp;
    vc->max_size_ = vc->max_size_ + vc->per_malloc_size_;
  }
  vc->data_[vc->size_] = value;
  vc->size_++;
}

void VectorCInsert(VectorC *vc, int index, int value) {
  if (vc->size_ + 1 > vc->max_size_) {
    int *tmp = (int *)malloc(vc->per_malloc_size_ * sizeof(int) + vc->max_size_ * sizeof(int));
    memcpy(tmp, vc->data_, vc->size_ * sizeof(int));
    free(vc->data_);
    vc->data_ = tmp;
    vc->max_size_ = vc->max_size_ + vc->per_malloc_size_;
  }
  memmove(vc->data_ + index + 1, vc->data_ + index, (vc->size_ - index) * sizeof(int));
  vc->data_[index] = value;
  vc->size_++;
}

void VectorCErase(VectorC *vc, int index) {
  memmove(vc->data_ + index, vc->data_ + index + 1, (vc->size_ - index - 1) * sizeof(int));
  vc->size_--;
}

bool VectorCEqual(VectorC *vc1, VectorC *vc2) {
  if (vc1->size_ != vc2->size_) {
    return false;
  }
  for (size_t i = 0; i < vc1->size_; i++) {
    if (vc1->data_[i] != vc2->data_[i]) {
      return false;
    }
  }
  return true;
}

void VectorCFree(VectorC *vc) {
  free(vc->data_);
  vc->data_ = NULL;
}

REG_INFER(Abs, PrimType_Abs, CommonInferShape)
REG_INFER(AbsGrad, PrimType_AbsGrad, CommonInferShape)
REG_INFER(Activation, PrimType_Activation, CommonInferShape)
REG_INFER(ActivationGrad, PrimType_ActivationGrad, CommonInferShape)
REG_INFER(BatchNorm, PrimType_BatchNorm, CommonInferShape)
REG_INFER(BinaryCrossEntropyGrad, PrimType_BinaryCrossEntropyGrad, CommonInferShape)
REG_INFER(BiasAdd, PrimType_BiasAdd, CommonInferShape)
REG_INFER(Ceil, PrimType_Ceil, CommonInferShape)
REG_INFER(Clip, PrimType_Clip, CommonInferShape)
REG_INFER(ControlDepend, PrimType_ControlDepend, CommonInferShape)
REG_INFER(Cos, PrimType_Cos, CommonInferShape)
REG_INFER(Depend, PrimType_Depend, CommonInferShape)
REG_INFER(Elu, PrimType_Elu, CommonInferShape)
REG_INFER(Erf, PrimType_Erf, CommonInferShape)
REG_INFER(Exp, PrimType_ExpFusion, CommonInferShape)
REG_INFER(FakeQuantWithMinMaxVars, PrimType_FakeQuantWithMinMaxVars, CommonInferShape)
REG_INFER(Floor, PrimType_Floor, CommonInferShape)
REG_INFER(If, PrimType_If, CommonInferShape)
REG_INFER(InstanceNorm, PrimType_InstanceNorm, CommonInferShape)
REG_INFER(IsFinite, PrimType_IsFinite, CommonInferShape)
REG_INFER(LeakyRelu, PrimType_LeakyRelu, CommonInferShape)
REG_INFER(Log, PrimType_Log, CommonInferShape)
REG_INFER(LogGrad, PrimType_LogGrad, CommonInferShape)
REG_INFER(LogicalNot, PrimType_LogicalNot, CommonInferShape)
REG_INFER(LRN, PrimType_LRN, CommonInferShape)
REG_INFER(L2Normalize, PrimType_L2NormalizeFusion, CommonInferShape)
REG_INFER(Neg, PrimType_Neg, CommonInferShape)
REG_INFER(NegGrad, PrimType_NegGrad, CommonInferShape)
REG_INFER(PowerGrad, PrimType_PowerGrad, CommonInferShape)
REG_INFER(PReLU, PrimType_PReLUFusion, CommonInferShape)
REG_INFER(Reciprocal, PrimType_Reciprocal, CommonInferShape)
REG_INFER(ReverseSequence, PrimType_ReverseSequence, CommonInferShape)
REG_INFER(Reverse, PrimType_ReverseV2, CommonInferShape)
REG_INFER(Round, PrimType_Round, CommonInferShape)
REG_INFER(Rsqrt, PrimType_Rsqrt, CommonInferShape)
REG_INFER(Scale, PrimType_ScaleFusion, CommonInferShape)
REG_INFER(SigmoidCrossEntropyWithLogits, PrimType_SigmoidCrossEntropyWithLogits, CommonInferShape)
REG_INFER(SigmoidCrossEntropyWithLogitsGrad, PrimType_SigmoidCrossEntropyWithLogitsGrad, CommonInferShape)
REG_INFER(Sin, PrimType_Sin, CommonInferShape)
REG_INFER(SmoothL1Loss, PrimType_SmoothL1Loss, CommonInferShape)
REG_INFER(SmoothL1LossGrad, PrimType_SmoothL1LossGrad, CommonInferShape)
REG_INFER(Sqrt, PrimType_Sqrt, CommonInferShape)
REG_INFER(Square, PrimType_Square, CommonInferShape)
REG_INFER(ZerosLike, PrimType_ZerosLike, CommonInferShape)
