/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl/kernel/non_max_suppression.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/non_max_suppression_parameter.h"
#include "nnacl/fp32/non_max_suppression_fp32.h"

void NonMaxSuppressioExpandDims(int *dst_shape, int *origin_shape, size_t size) {
  int i = 0;
  for (; i < size; i++) {
    dst_shape[i] = 1;
  }
  for (; i < Num3; i++) {
    dst_shape[i] = origin_shape[i - size];
  }
}

void NonMaxSuppressionGetParams(NonMaxSuppressionStruct *nm_suppression) {
  // optional input order: max_output_per_class, iou_threshold, score_threshold
  nm_suppression->max_output_per_class_ = 0;
  if (nm_suppression->base_.in_size_ >= Num3) {
    TensorC *max_output_tensor = nm_suppression->base_.in_[Index3];
    if (max_output_tensor != NULL && max_output_tensor->data_ != NULL) {
      nm_suppression->max_output_per_class_ = *(int *)(max_output_tensor->data_);
    }
  }

  nm_suppression->iou_threshold_ = 0.0f;
  if (nm_suppression->base_.in_size_ >= Num4) {
    TensorC *iou_threshold_tensor = nm_suppression->base_.in_[Index4];
    if (iou_threshold_tensor != NULL && iou_threshold_tensor->data_ != NULL) {
      nm_suppression->iou_threshold_ = *(float *)(iou_threshold_tensor->data_);
    }
  }

  nm_suppression->score_threshold_ = 0.0f;
  if (nm_suppression->base_.in_size_ >= Num5) {
    TensorC *score_threshold_tensor = nm_suppression->base_.in_[Index5];
    if (score_threshold_tensor != NULL && score_threshold_tensor->data_ != NULL) {
      nm_suppression->score_threshold_ = *(float *)(score_threshold_tensor->data_);
    }
  }
}

int NonMaxSuppressionCompute(KernelBase *self) {
  NonMaxSuppressionStruct *nm_suppression = (NonMaxSuppressionStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(nm_suppression);

  NonMaxSuppressionGetParams(nm_suppression);

  TensorC *box_tensor = self->in_[Index0];
  NNACL_CHECK_NULL_RETURN_ERR(box_tensor);
  int box_dims[Num3] = {0};  // batch, box_num, 4
  bool simple_out = false;
  if (box_tensor->shape_size_ != Num3) {
    NonMaxSuppressioExpandDims(box_dims, box_tensor->shape_, Num3 - box_tensor->shape_size_);
    simple_out = true;
  }
  if (box_dims[Index2] != Num4) {
    return NNACL_NON_MAX_SUPPRESSION_BOX_DIMS_INVALID;
  }

  TensorC *score_tensor = self->in_[Index1];
  NNACL_CHECK_NULL_RETURN_ERR(score_tensor);
  int score_dims[Num3] = {0};  // batch, class, box_num
  if (score_tensor->shape_size_ != Num3) {
    NonMaxSuppressioExpandDims(score_dims, score_tensor->shape_, Num3 - score_tensor->shape_size_);
  }
  if (score_dims[Index0] != box_dims[Index0]) {
    return NNACL_NON_MAX_SUPPRESSION_BOX_DIMS_SCORE_UNMATCH;
  }
  if (score_dims[Index2] != box_dims[Index1]) {
    return NNACL_NON_MAX_SUPPRESSION_DIMENSION_SPATIAL_UNMATCH;
  }
  if (nm_suppression->base_.out_[OUTPUT_INDEX]->data_ != NULL) {
    /* output shape and data set in compute */
    return NNACL_NON_MAX_SUPPRESSION_UNSUPPORT_DEFINE_DATA;
  }
  return NonMaxSuppressionSelecte(nm_suppression, simple_out, score_dims);
}

int NonMaxSuppressionPrepare(KernelBase *self) {
  NonMaxSuppressionStruct *nm_suppression = (NonMaxSuppressionStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(nm_suppression);

  // boxes, scores, max_output_boxes, iou_threshold, score_threshold
  if (self->in_size_ < Num2 || self->in_size_ > Num5 || self->out_size_ != Num1) {
    return NNACL_NON_MAX_SUPPRESSION_TENSOR_SIZE_INVALID;
  }

  NNACL_CHECK_NULL_RETURN_ERR(self->in_[FIRST_INPUT]);
  NNACL_CHECK_NULL_RETURN_ERR(self->out_[OUTPUT_INDEX]);
  NMSParameter *nmparam = (NMSParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(nmparam);
  if (nmparam->center_point_box_ != 0 && nmparam->center_point_box_ != 1) {
    return NNACL_NON_MAX_SUPPRESSION_PARAM_INVALID;
  }
  nm_suppression->center_point_box_ = nmparam->center_point_box_;
  return NNACL_OK;
}

KernelBase *CreateNonMaxSuppression(OpParameter *param, int data_type) {
  NonMaxSuppressionStruct *non_max_suppression = (NonMaxSuppressionStruct *)malloc(sizeof(NonMaxSuppressionStruct));
  NNACL_CHECK_NULL_RETURN_NULL(non_max_suppression);
  non_max_suppression->base_.Release = DefaultRelease;
  non_max_suppression->base_.Resize = DefaultResize;
  non_max_suppression->base_.Prepare = NonMaxSuppressionPrepare;
  non_max_suppression->base_.Compute = NonMaxSuppressionCompute;
  return (KernelBase *)non_max_suppression;
}

REG_KERNEL_CREATOR(PrimType_NonMaxSuppression, kNumberTypeFloat32, CreateNonMaxSuppression)
