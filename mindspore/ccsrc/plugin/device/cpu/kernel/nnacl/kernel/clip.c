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

#include "nnacl/kernel/clip.h"
#include "nnacl/op_base.h"
#include "nnacl/clip_parameter.h"
#include "nnacl/fp32/activation_fp32.h"

int GetClipMinMaxValue(TensorC *tensor, float *data) {
  NNACL_CHECK_NULL_RETURN_ERR(tensor);
  switch (tensor->data_type_) {
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      *data = *((float *)tensor->data_);
      break;
    case kNumberTypeInt:
    case kNumberTypeInt32:
      *data = *((int *)tensor->data_);
      break;
    default:
      return NNACL_CLIP_DATA_TYPE_INVALID;
  }
  return NNACL_OK;
}

int clip_prepare(struct KernelBase *self) {
  ClipStruct *clip = (ClipStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(clip);
  NNACL_CHECK_FALSE(self->in_size_ < ONE_TENSOR || self->out_size_ < ONE_TENSOR, NNACL_TENSOR_SIZE_INVALID);
  return NNACL_OK;
}

int clip_resize(struct KernelBase *self) {
  ClipStruct *clip = (ClipStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(clip);
  clip->base_.thread_nr_ = clip->base_.update_thread_(
    TC_PTYPE(PrimType_Clip), 1, 1, GetElementNum(clip->base_.out_[FIRST_INPUT]), clip->base_.thread_nr_);
  return NNACL_OK;
}

int clip_release(struct KernelBase *self) { return NNACL_OK; }

int clip_do_compute(void *cdata, int task_id, float l, float r) {
  ClipStruct *clip = (ClipStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(clip);

  int length = GetElementNum(clip->base_.in_[FIRST_INPUT]);
  int stride = UP_DIV(length, clip->base_.thread_nr_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(stride, task_id, NNACL_ERR);
  int count = MSMIN(stride, length - stride * task_id);
  if (count <= 0) {
    return NNACL_OK;
  }

  int ret = NNACL_OK;
  switch (clip->base_.in_[FIRST_INPUT]->data_type_) {
    case kNumberTypeFloat:
    case kNumberTypeFloat32: {
      ret = Fp32Clip((float *)clip->base_.in_[FIRST_INPUT]->data_ + stride * task_id, count,
                     (float *)clip->base_.out_[FIRST_INPUT]->data_ + stride * task_id, clip->min_val_, clip->max_val_);
    } break;
    case kNumberTypeInt:
    case kNumberTypeInt32: {
      ret = Int32Clip((int *)clip->base_.in_[FIRST_INPUT]->data_ + stride * task_id, count,
                      (int *)clip->base_.out_[FIRST_INPUT]->data_ + stride * task_id, (int)clip->min_val_,
                      (int)clip->max_val_);
    } break;
    default:
      return NNACL_CLIP_DATA_TYPE_INVALID;
  }
  return ret;
}

int clip_compute(struct KernelBase *self) {
  ClipStruct *clip = (ClipStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(clip);
  ClipParameter *param = (ClipParameter *)clip->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);

  int ret = NNACL_OK;
  if (clip->base_.in_size_ > ONE_TENSOR) {
    TensorC *min_tensor = clip->base_.in_[SECOND_INPUT];
    NNACL_CHECK_NULL_RETURN_ERR(min_tensor);
    NNACL_CHECK_NULL_RETURN_ERR(min_tensor->data_);
    ret = GetClipMinMaxValue(min_tensor, &(clip->min_val_));
  }
  if (clip->base_.in_size_ > TWO_TENSOR) {
    TensorC *max_tensor = clip->base_.in_[THIRD_INPUT];
    NNACL_CHECK_NULL_RETURN_ERR(max_tensor);
    NNACL_CHECK_NULL_RETURN_ERR(max_tensor->data_);
    ret = GetClipMinMaxValue(max_tensor, &(clip->max_val_));
  }
  if (ret != NNACL_OK) {
    return ret;
  }
  if (clip->min_val_ >= clip->max_val_) {
    return NNACL_CLIP_MINMAX_VALUE_INVALID;
  }

  return self->env_->parallel_launch(self->env_->thread_pool_, clip_do_compute, clip, clip->base_.thread_nr_);
}

KernelBase *CreateClip(OpParameter *param, int data_type) {
  ClipStruct *clip = (ClipStruct *)malloc(sizeof(ClipStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(clip);
  clip->base_.prepare = clip_prepare;
  clip->base_.resize = clip_resize;
  clip->base_.release = clip_release;
  clip->base_.compute = clip_compute;
  return (KernelBase *)clip;
}

REG_KERNEL_CREATOR(PrimType_Clip, kNumberTypeFloat, CreateClip);
REG_KERNEL_CREATOR(PrimType_Clip, kNumberTypeFloat32, CreateClip);
REG_KERNEL_CREATOR(PrimType_Clip, kNumberTypeInt, CreateClip);
REG_KERNEL_CREATOR(PrimType_Clip, kNumberTypeInt32, CreateClip);
