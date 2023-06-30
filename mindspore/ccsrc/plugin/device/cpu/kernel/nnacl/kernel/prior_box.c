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

#include "nnacl/kernel/prior_box.h"
#include <math.h>
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/fp32/prior_box_fp32.h"
#include "nnacl/tensor_c_utils.h"

int PriorBoxInitOutput(PriorBoxStruct *prior_box, const PriorBoxParameter *param, const float *different_aspect_ratios,
                       int different_aspect_ratios_size) {
  for (int i = 0; i < prior_box->fmap_h_; i++) {
    float cy = i + param->offset;
    for (int j = 0; j < prior_box->fmap_w_; j++) {
      float cx = j + param->offset;
      for (int32_t k = 0; k < param->min_sizes_size; k++) {
        float min = param->min_sizes[k];
        prior_box->output_[prior_box->output_size_++] = (cx - min / prior_box->step_w_ * 0.5f) / prior_box->fmap_w_;
        prior_box->output_[prior_box->output_size_++] = (cy - min / prior_box->step_h_ * 0.5f) / prior_box->fmap_h_;
        prior_box->output_[prior_box->output_size_++] = (cx + min / prior_box->step_w_ * 0.5f) / prior_box->fmap_w_;
        prior_box->output_[prior_box->output_size_++] = (cy + min / prior_box->step_h_ * 0.5f) / prior_box->fmap_h_;

        if (param->max_sizes_size > 0) {
          float max = param->max_sizes[k];
          NNACL_CHECK_FALSE(min * max <= 0, NNACL_PRIOR_BOX_VALUE_INVALID);
          float prime = sqrt(min * max);
          prior_box->output_[prior_box->output_size_++] = (cx - prime / prior_box->step_w_ * 0.5f) / prior_box->fmap_w_;
          prior_box->output_[prior_box->output_size_++] = (cy - prime / prior_box->step_h_ * 0.5f) / prior_box->fmap_h_;
          prior_box->output_[prior_box->output_size_++] = (cx + prime / prior_box->step_w_ * 0.5f) / prior_box->fmap_w_;
          prior_box->output_[prior_box->output_size_++] = (cy + prime / prior_box->step_h_ * 0.5f) / prior_box->fmap_h_;
        }

        for (int m = 0; m < different_aspect_ratios_size; m++) {
          float v = different_aspect_ratios[m];
          if (fabs(v - 1.0f) < 1e-6) {
            continue;
          }
          NNACL_CHECK_FALSE(v <= 0, NNACL_PRIOR_BOX_VALUE_INVALID);
          float as_square_root = sqrt(v);
          NNACL_CHECK_FALSE(as_square_root <= 0, NNACL_PRIOR_BOX_VALUE_INVALID);
          float box_w = min * as_square_root;
          float box_h = min / as_square_root;
          prior_box->output_[prior_box->output_size_++] = (cx - box_w / prior_box->step_w_ * 0.5f) / prior_box->fmap_w_;
          prior_box->output_[prior_box->output_size_++] = (cy - box_h / prior_box->step_h_ * 0.5f) / prior_box->fmap_h_;
          prior_box->output_[prior_box->output_size_++] = (cx + box_w / prior_box->step_w_ * 0.5f) / prior_box->fmap_w_;
          prior_box->output_[prior_box->output_size_++] = (cy + box_h / prior_box->step_h_ * 0.5f) / prior_box->fmap_h_;
        }
      }
    }
  }
  return NNACL_OK;
}

int RunPriorBox(void *cdata, int task_id, float l, float r) {
  PriorBoxStruct *prior_box = (PriorBoxStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(prior_box);
  TensorC *output_tensor = prior_box->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);
  float *output_data = output_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_data);
  return PriorBox(prior_box->output_, output_data, GetSize(output_tensor), task_id, prior_box->base_.thread_nr_);
}

int PriorBoxRelease(KernelBase *self) {
  PriorBoxStruct *prior_box = (PriorBoxStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(prior_box);
  if (prior_box->output_ != NULL) {
    self->env_->Free(self->env_->allocator_, prior_box->output_);
    prior_box->output_ = NULL;
    prior_box->output_size_ = 0;
  }
  return NNACL_OK;
}

int PriorBoxResize(KernelBase *self) {
  PriorBoxStruct *prior_box = (PriorBoxStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(prior_box);
  PriorBoxParameter *param = (PriorBoxParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);
  TensorC *input0_tensor = prior_box->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input0_tensor);
  TensorC *input1_tensor = prior_box->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input1_tensor);
  TensorC *output_tensor = prior_box->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);

  prior_box->fmap_w_ = GetWidth(input0_tensor);
  NNACL_CHECK_ZERO_RETURN_ERR(prior_box->fmap_w_);
  prior_box->fmap_h_ = GetHeight(input1_tensor);
  NNACL_CHECK_ZERO_RETURN_ERR(prior_box->fmap_h_);
  const int image_w = param->image_size_w > 0 ? param->image_size_w : GetWidth(input1_tensor);
  const int image_h = param->image_size_h > 0 ? param->image_size_h : GetHeight(input1_tensor);

  prior_box->step_w_ = param->step_w > 0.0f ? param->step_w : (float)(image_w) / prior_box->fmap_w_;
  prior_box->step_h_ = param->step_h > 0.0f ? param->step_h : (float)(image_h) / prior_box->fmap_h_;

  float *different_aspect_ratios =
    (float *)self->env_->Alloc(self->env_->allocator_, param->aspect_ratios_size * sizeof(float) * Num2);
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(different_aspect_ratios);
  different_aspect_ratios[Index0] = 1.0f;
  int different_aspect_ratios_size = 1;

  float *aspect_ratios = param->aspect_ratios;
  for (int32_t i = 0; i < param->aspect_ratios_size; i++) {
    float ratio = aspect_ratios[i];

    bool exist = false;
    for (int k = 0; k < different_aspect_ratios_size; k++) {
      if (fabs(ratio - different_aspect_ratios[k]) < 1e-6) {
        exist = true;
      }
    }

    if (!exist) {
      different_aspect_ratios[different_aspect_ratios_size++] = ratio;
      if (param->flip) {
        NNACL_CHECK_FALSE(fabs(ratio) <= 1e-5, NNACL_PRIOR_BOX_RATIO_INVALID);
        different_aspect_ratios[different_aspect_ratios_size++] = 1.0f / ratio;
      }
    }
  }

  PriorBoxRelease(self);
  int size = Num4 + Num4 + different_aspect_ratios_size;
  size = size * prior_box->fmap_h_ * prior_box->fmap_w_ * param->min_sizes_size;
  size = size + UP_ROUND(GetHeight(output_tensor), COMM_SHAPE_SIZE);
  size = size * sizeof(float);
  NNACL_CHECK_MALLOC_SIZE(size);
  prior_box->output_ = (float *)self->env_->Alloc(self->env_->allocator_, size);
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(prior_box->output_);
  prior_box->output_size_ = 0;

  int ret = PriorBoxInitOutput(prior_box, param, different_aspect_ratios, different_aspect_ratios_size);
  if (ret != NNACL_OK) {
    return ret;
  }

  // do clip
  if (param->clip) {
    for (int i = 0; i < prior_box->output_size_; i++) {
      float item = prior_box->output_[i];
      if (item > 1.0f) {
        item = 1.0f;
      }
      if (item < 0.0f) {
        item = 0.0f;
      }
    }
  }

  // variance
  for (int i = 0; i < GetHeight(output_tensor) / COMM_SHAPE_SIZE; i++) {
    for (int j = 0; j < COMM_SHAPE_SIZE; j++) {
      prior_box->output_[prior_box->output_size_++] = param->variances[j];
    }
  }
  return NNACL_OK;
}

int PriorBoxCompute(KernelBase *self) {
  return self->env_->ParallelLaunch(self->env_->thread_pool_, RunPriorBox, self, self->thread_nr_);
}

KernelBase *CreatePriorBox(OpParameter *param, int data_type) {
  PriorBoxStruct *prior_box = (PriorBoxStruct *)malloc(sizeof(PriorBoxStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(prior_box);
  memset(prior_box, 0, sizeof(PriorBoxStruct));

  prior_box->base_.Prepare = DefaultPrepare2In1Out;
  prior_box->base_.Resize = PriorBoxResize;
  prior_box->base_.Release = PriorBoxRelease;
  prior_box->base_.Compute = PriorBoxCompute;
  return (KernelBase *)prior_box;
}

REG_KERNEL_CREATOR(PrimType_PriorBox, kNumberTypeFloat32, CreatePriorBox)
REG_KERNEL_CREATOR(PrimType_PriorBox, kNumberTypeInt8, CreatePriorBox)
