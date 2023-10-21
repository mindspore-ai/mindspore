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

#include "nnacl/kernel/crop_and_resize.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/fp32/resize_fp32.h"
#include "nnacl/tensor_c_utils.h"

int CropAndResizeMallocTmpBuffer(CropAndResizeStruct *crop_and_resize) {
  TensorC *input_tensor = crop_and_resize->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  TensorC *output_tensor = crop_and_resize->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);
  ExecEnv *env = crop_and_resize->base_.env_;
  NNACL_CHECK_NULL_RETURN_ERR(env);

  // Malloc buffer to save coordinate.
  // For mode CROP_AND_RESIZE, different output batches require different cache coordinates.
  crop_and_resize->batch_ = GetBatch(output_tensor);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(crop_and_resize->new_height_, crop_and_resize->batch_, NNACL_ERR);
  int height_size = crop_and_resize->new_height_ * crop_and_resize->batch_;
  NNACL_CHECK_MALLOC_SIZE(height_size);
  crop_and_resize->y_bottoms_ = (int *)env->Alloc(env->allocator_, height_size * sizeof(int));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(crop_and_resize->y_bottoms_);
  crop_and_resize->y_tops_ = (int *)env->Alloc(env->allocator_, height_size * sizeof(int));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(crop_and_resize->y_tops_);
  crop_and_resize->y_bottom_weights_ = (float *)env->Alloc(env->allocator_, height_size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(crop_and_resize->y_bottom_weights_);

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(crop_and_resize->new_width_, crop_and_resize->batch_, NNACL_ERR);
  int width_size = crop_and_resize->new_width_ * crop_and_resize->batch_;
  NNACL_CHECK_MALLOC_SIZE(width_size);
  crop_and_resize->x_lefts_ = (int *)env->Alloc(env->allocator_, width_size * sizeof(int));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(crop_and_resize->x_lefts_);
  crop_and_resize->x_rights_ = (int *)env->Alloc(env->allocator_, width_size * sizeof(int));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(crop_and_resize->x_rights_);
  crop_and_resize->x_left_weights_ = (float *)env->Alloc(env->allocator_, width_size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(crop_and_resize->x_left_weights_);

  int c = GetChannel(input_tensor);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(crop_and_resize->new_width_, c, NNACL_ERR);
  int new_wc = crop_and_resize->new_width_ * c;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(new_wc, crop_and_resize->mapped_point_num_, NNACL_ERR);
  int total_point_num = new_wc * crop_and_resize->mapped_point_num_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(total_point_num, crop_and_resize->base_.thread_nr_, NNACL_ERR);
  int line_buffer_size = total_point_num * crop_and_resize->base_.thread_nr_ * sizeof(float);
  crop_and_resize->line_buffer_ = (float *)env->Alloc(env->allocator_, line_buffer_size);
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(crop_and_resize->line_buffer_);
  return NNACL_OK;
}

void CropAndResizeFreeTmpBuffer(CropAndResizeStruct *crop_and_resize) {
  ExecEnv *env = crop_and_resize->base_.env_;
  NNACL_CHECK_NULL_RETURN_VOID(env);
  env->Free(env->allocator_, crop_and_resize->y_bottoms_);
  env->Free(env->allocator_, crop_and_resize->y_tops_);
  env->Free(env->allocator_, crop_and_resize->y_bottom_weights_);
  env->Free(env->allocator_, crop_and_resize->x_lefts_);
  env->Free(env->allocator_, crop_and_resize->x_rights_);
  env->Free(env->allocator_, crop_and_resize->x_left_weights_);
  env->Free(env->allocator_, crop_and_resize->line_buffer_);
  crop_and_resize->y_bottoms_ = NULL;
  crop_and_resize->y_tops_ = NULL;
  crop_and_resize->y_bottom_weights_ = NULL;
  crop_and_resize->x_lefts_ = NULL;
  crop_and_resize->x_rights_ = NULL;
  crop_and_resize->x_left_weights_ = NULL;
  crop_and_resize->line_buffer_ = NULL;
}

int CropAndResizeImpl(void *cdata, int task_id, float l, float r) {
  CropAndResizeStruct *crop_and_resize = (CropAndResizeStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(crop_and_resize);

  TensorC *input = crop_and_resize->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  TensorC *boxes = crop_and_resize->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(boxes);
  TensorC *box_idx = crop_and_resize->base_.in_[THIRD_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(box_idx);
  TensorC *output = crop_and_resize->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output);

  int unit = UP_DIV(crop_and_resize->new_height_, crop_and_resize->base_.thread_nr_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(unit, task_id, NNACL_ERR);
  int h_begin = unit * task_id;
  int h_end = MSMIN(h_begin + unit, crop_and_resize->new_height_);
  if (h_end <= h_begin) {
    return NNACL_OK;
  }

  float extrapolation_value = ((CropAndResizeParameter *)crop_and_resize->base_.param_)->extrapolation_value_;
  int c = input->shape_[kNHWC_C];
  float *line0 = crop_and_resize->line_buffer_ + crop_and_resize->new_width_ * c * 2 * task_id;
  float *line1 = line0 + crop_and_resize->new_width_ * c;

  return CropAndResizeBilinear((float *)input->data_, (float *)output->data_, (int32_t *)box_idx->data_,
                               (float *)boxes->data_, extrapolation_value, input->shape_, output->shape_,
                               crop_and_resize->y_bottoms_, crop_and_resize->y_tops_, crop_and_resize->x_lefts_,
                               crop_and_resize->x_rights_, crop_and_resize->y_bottom_weights_,
                               crop_and_resize->x_left_weights_, line0, line1, h_begin, h_end);
}

int CropAndResizeCompute(struct KernelBase *self) {
  CropAndResizeStruct *crop_and_resize = (CropAndResizeStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(crop_and_resize);

  // In Prepare() stage, in_tensor[0] may be of fp16 data type in fp16 mode, so move type checks here.
  TensorC *input_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  TensorC *boxes_tensor = self->in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(boxes_tensor);
  TensorC *boxidx_tensor = self->in_[THIRD_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(boxidx_tensor);
  TensorC *output_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);

  int ret = CropAndResizeMallocTmpBuffer(crop_and_resize);
  if (ret != NNACL_OK) {
    CropAndResizeFreeTmpBuffer(crop_and_resize);
    return ret;
  }

  float *boxes = (float *)boxes_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(boxes);
  int32_t *box_idx = (int32_t *)boxidx_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(box_idx);

  if (CheckCropAndResizeBoxIdx(box_idx, boxes_tensor->shape_[Index0], GetBatch(input_tensor)) != NNACL_OK) {
    return NNACL_CROP_AND_RESIZE_BOX_IDX_INVALID;
  }

  ret = PrepareCropAndResizeBilinear(input_tensor->shape_, boxes, box_idx, output_tensor->shape_,
                                     crop_and_resize->y_bottoms_, crop_and_resize->y_tops_, crop_and_resize->x_lefts_,
                                     crop_and_resize->x_rights_, crop_and_resize->y_bottom_weights_,
                                     crop_and_resize->x_left_weights_);
  if (ret != NNACL_OK) {
    CropAndResizeFreeTmpBuffer(crop_and_resize);
    return ret;
  }

  int error_code = self->env_->ParallelLaunch(self->env_->thread_pool_, CropAndResizeImpl, self, self->thread_nr_);
  CropAndResizeFreeTmpBuffer(crop_and_resize);
  return error_code;
}

int CropAndResizeResize(KernelBase *self) {
  CropAndResizeStruct *crop_and_resize = (CropAndResizeStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(crop_and_resize);
  TensorC *output = self->out_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(output);
  crop_and_resize->new_height_ = output->shape_[Index1];
  crop_and_resize->new_width_ = output->shape_[Index2];
  return NNACL_OK;
}

int CropAndResizePrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < THREE_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_ERR);
  NNACL_CHECK_NULL_RETURN_ERR(self->in_[FIRST_INPUT]);
  NNACL_CHECK_NULL_RETURN_ERR(self->out_[OUTPUT_INDEX]);
  return NNACL_OK;
}

KernelBase *CreateCropAndResize(OpParameter *param, int data_type) {
  CropAndResizeStruct *crop_and_resize = (CropAndResizeStruct *)malloc(sizeof(CropAndResizeStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(crop_and_resize);
  memset(crop_and_resize, 0, sizeof(CropAndResizeStruct));
  crop_and_resize->mapped_point_num_ = Num2;
  crop_and_resize->base_.Prepare = CropAndResizePrepare;
  crop_and_resize->base_.Resize = CropAndResizeResize;
  crop_and_resize->base_.Compute = CropAndResizeCompute;
  crop_and_resize->base_.Release = DefaultRelease;
  return (KernelBase *)crop_and_resize;
}

REG_KERNEL_CREATOR(PrimType_CropAndResize, kNumberTypeFloat32, CreateCropAndResize)
