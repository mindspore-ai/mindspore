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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either log_softmaxress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/kernel/local_response_norm.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/fp32/local_response_norm_fp32.h"
#include "nnacl/tensor_c_utils.h"

int LocalResponseNormRun(void *cdata, int task_id, float l, float r) {
  LocalResponseNormStruct *lrn = (LocalResponseNormStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(lrn);
  LocalResponseNormParameter *param = (LocalResponseNormParameter *)lrn->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);

  TensorC *input = lrn->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  TensorC *output = lrn->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output);
  NNACL_CHECK_FALSE(input->shape_size_ != DIMENSION_4D, NNACL_LOCAL_RESPONSE_NORM_SHAPE_INVALID);
  NNACL_CHECK_FALSE(param->depth_radius_ <= 0, NNACL_LOCAL_RESPONSE_NORM_DEPTH_RADIUS_INVALID);

  float *input_ptr = (float *)input->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_ptr);
  float *output_ptr = (float *)output->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_ptr);

  int batch = GetBatch(input);
  int height = GetHeight(input);
  int width = GetWidth(input);
  int channel = GetChannel(input);

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(batch, width, NNACL_ERR);
  int size_bw = batch * width;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(size_bw, height, NNACL_ERR);
  int outer_size = size_bw * height;
  int stride = UP_DIV(outer_size, lrn->base_.thread_nr_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(stride, task_id, NNACL_ERR);
  int start = stride * task_id;
  int count = MSMIN(stride, outer_size - start);

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(start, channel, NNACL_ERR);
  input_ptr += start * channel;
  output_ptr += start * channel;

  return LocalResponseNorm(input_ptr, count, channel, output_ptr, param);
}

int LrnCompute(KernelBase *self) {
  return self->env_->ParallelLaunch(self->env_->thread_pool_, LocalResponseNormRun, self, self->thread_nr_);
}

KernelBase *CreateLocalResponseNorm(OpParameter *param, int data_type) {
  LocalResponseNormStruct *lrn = (LocalResponseNormStruct *)malloc(sizeof(LocalResponseNormStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(lrn);
  memset(lrn, 0, sizeof(LocalResponseNormStruct));

  lrn->base_.Prepare = DefaultPrepare1In1Out;
  lrn->base_.Release = DefaultRelease;
  lrn->base_.Resize = DefaultResize;
  lrn->base_.Compute = LrnCompute;
  return (KernelBase *)lrn;
}

REG_KERNEL_CREATOR(PrimType_LRN, kNumberTypeFloat32, CreateLocalResponseNorm)
