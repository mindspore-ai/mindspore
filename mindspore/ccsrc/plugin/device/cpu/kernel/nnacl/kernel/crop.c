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

#include "nnacl/kernel/crop.h"
#include "nnacl/base/crop_base.h"
#include "nnacl/fp32/crop_fp32.h"
#include "nnacl/kernel/default_kernel_base.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/crop_fp16.h"
#endif

int CropLaunch(void *cdata, int task_id, float l, float r) {
  CropStruct *crop = (CropStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(crop);

  TensorC *in = crop->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in);
  TensorC *out = crop->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out);

#ifdef ENABLE_FP16
  if (in->data_type_ == kNumberTypeFloat16) {
    Fp16Crop((float16_t *)in->data_, (float16_t *)out->data_, in->shape_, out->shape_, crop->in_offset_,
             in->shape_size_, task_id, crop->base_.thread_nr_);
    return NNACL_OK;
  }
#endif

  CropParameter *crop_param = (CropParameter *)crop->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(crop_param);
  Crop4D((float *)in->data_, (float *)out->data_, in->shape_, out->shape_, crop_param, task_id, crop->base_.thread_nr_);
  return NNACL_OK;
}

int CropResize(struct KernelBase *self) {
  TensorC *in_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in_tensor);
  TensorC *out_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor);
  NNACL_CHECK_FALSE(out_tensor->shape_size_ <= Num1, NNACL_OUTPUT_TENSOR_ERROR);

  CropStruct *crop = (CropStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(crop);
  CropParameter *crop_param = (CropParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(crop_param);

  return CropPadOffset(in_tensor->shape_size_, crop_param, crop->in_offset_);
}

int CropCompute(struct KernelBase *self) {
  TensorC *in_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in_tensor);
  TensorC *out_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor);
  CropParameter *crop_param = (CropParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(crop_param);

  if (in_tensor->data_type_ != kNumberTypeFloat16 && out_tensor->shape_[Index1] < self->thread_nr_) {
    float *input_data = (float *)in_tensor->data_;
    NNACL_CHECK_NULL_RETURN_ERR(input_data);
    float *output_data = (float *)out_tensor->data_;
    NNACL_CHECK_NULL_RETURN_ERR(output_data);
    Crop4DNoParallel(input_data, output_data, in_tensor->shape_, out_tensor->shape_, crop_param);
    return NNACL_OK;
  }

  return self->env_->ParallelLaunch(self->env_->thread_pool_, CropLaunch, self, self->thread_nr_);
}

KernelBase *CreateCrop(OpParameter *param, int data_type) {
  CropStruct *crop = (CropStruct *)malloc(sizeof(CropStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(crop);
  memset(crop, 0, sizeof(CropStruct));
  crop->base_.Prepare = DefaultPrepare1In1Out;
  crop->base_.Resize = CropResize;
  crop->base_.Release = DefaultRelease;
  crop->base_.Compute = CropCompute;
  return (KernelBase *)crop;
}

REG_KERNEL_CREATOR(PrimType_Crop, kNumberTypeInt32, CreateCrop)
REG_KERNEL_CREATOR(PrimType_Crop, kNumberTypeFloat32, CreateCrop)
REG_KERNEL_CREATOR(PrimType_Crop, kNumberTypeFloat16, CreateCrop)
