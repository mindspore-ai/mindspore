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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either convolutionress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/kernel/convolution_depthwise.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/base/conv_common_base.h"
#include "nnacl/fp32/conv_depthwise_fp32.h"
#include "nnacl/fp32/pack_fp32.h"
#ifdef ENABLE_AVX512
#include "nnacl/intrinsics/ms_simd_cpu_info.h"
#endif
#include "nnacl/fp32/conv_depthwise_avx_fp32.h"

int ConvDwRun(void *cdata, int task_id, float l, float r) {
  ConvolutionDepthwiseStruct *conv_dw = (ConvolutionDepthwiseStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);

  ConvParameter *conv_param = (ConvParameter *)conv_dw->conv_.base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

#ifdef ENABLE_AVX512
  if (X86_Avx512_Support()) {
    return ConvDwAVX512(conv_dw->output_ptr_, conv_dw->input_ptr_, (float *)conv_dw->conv_.packed_weight_,
                        (float *)conv_dw->conv_.bias_data_, conv_param, task_id, &conv_dw->dw_param_);
  } else {
    return ConvDwAVX(conv_dw->output_ptr_, conv_dw->input_ptr_, (float *)conv_dw->conv_.packed_weight_,
                     (float *)conv_dw->conv_.bias_data_, conv_param, task_id, &conv_dw->dw_param_);
  }
#endif

#ifdef ENABLE_AVX
  return ConvDwAVX(conv_dw->output_ptr_, conv_dw->input_ptr_, (float *)conv_dw->conv_.packed_weight_,
                   (float *)conv_dw->conv_.bias_data_, conv_param, task_id, &conv_dw->dw_param_);
#endif

  return ConvDw(conv_dw->output_ptr_, conv_dw->input_ptr_, (float *)conv_dw->conv_.packed_weight_,
                (float *)conv_dw->conv_.bias_data_, conv_param, task_id);
}

void ConvDwReleaseParam(ConvolutionDepthwiseStruct *conv_dw) {
  ExecEnv *env = conv_dw->conv_.base_.env_;
  NNACL_CHECK_NULL_RETURN_VOID(env);

  if (conv_dw->dw_param_.num_pixels_ != NULL) {
    env->Free(env->allocator_, conv_dw->dw_param_.num_pixels_);
    conv_dw->dw_param_.num_pixels_ = NULL;
  }
  if (conv_dw->dw_param_.out_w_start_ != NULL) {
    env->Free(env->allocator_, conv_dw->dw_param_.out_w_start_);
    conv_dw->dw_param_.out_w_start_ = NULL;
  }
  if (conv_dw->dw_param_.out_w_end_ != NULL) {
    env->Free(env->allocator_, conv_dw->dw_param_.out_w_end_);
    conv_dw->dw_param_.out_w_end_ = NULL;
  }
}

void ConvDwPackWeight(ConvolutionBaseStruct *conv) {
  void *origin_data = conv->base_.in_[SECOND_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_VOID(origin_data);
  PackWeightKHWToHWKFp32(origin_data, conv->packed_weight_, conv->compute_.kernel_hw_, conv->compute_.out_c_);
}

int ConvDwMallocWeightBiasData(ConvolutionBaseStruct *conv) {
  TensorC *weight_tensor = conv->base_.in_[SECOND_INPUT];
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(weight_tensor);

  int pack_weight_size = conv->compute_.kernel_hw_ * conv->compute_.out_c_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(pack_weight_size, sizeof(float), NNACL_ERR);

  if (!conv->base_.train_session_) {
    NNACL_CHECK_MALLOC_SIZE(pack_weight_size * sizeof(float));
    conv->packed_weight_ = ConvBaseGetConvPackWeightData(conv, pack_weight_size * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->packed_weight_);
  }

  NNACL_CHECK_MALLOC_SIZE(conv->compute_.out_c_ * sizeof(float));
  if (conv->bias_data_ == NULL) {
    conv->bias_data_ = conv->base_.env_->Alloc(conv->base_.env_->allocator_, conv->compute_.out_c_ * sizeof(float));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->bias_data_);
  }
  memset(conv->bias_data_, 0, conv->compute_.out_c_ * sizeof(float));
  return NNACL_OK;
}

int ConvDwInitConvDwCalcInfo(ConvolutionDepthwiseStruct *conv_dw) {
  ExecEnv *env = conv_dw->conv_.base_.env_;
  NNACL_CHECK_NULL_RETURN_ERR(env);
  ConvComputeParam *compute = &conv_dw->conv_.compute_;
  NNACL_CHECK_NULL_RETURN_ERR(compute);

  ConvDwReleaseParam(conv_dw);

  conv_dw->dw_param_.num_pixels_ = env->Alloc(env->allocator_, compute->kernel_w_ * sizeof(int));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_dw->dw_param_.num_pixels_);

  conv_dw->dw_param_.out_w_start_ = env->Alloc(env->allocator_, compute->kernel_w_ * sizeof(int));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_dw->dw_param_.out_w_start_);

  conv_dw->dw_param_.out_w_end_ = env->Alloc(env->allocator_, compute->kernel_w_ * sizeof(int));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv_dw->dw_param_.out_w_end_);

  int *num_pixels = (int *)(conv_dw->dw_param_.num_pixels_);
  int *out_w_start = (int *)(conv_dw->dw_param_.out_w_start_);
  int *out_w_end = (int *)(conv_dw->dw_param_.out_w_end_);
  conv_dw->dw_param_.first_calc_kw_ = -1;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->dilation_w_, (compute->kernel_w_ - 1), NNACL_ERR);
  for (int kw = 0; kw < compute->kernel_w_; kw++) {
    out_w_start[kw] =
      NNACL_MAX(0, (compute->pad_l_ - compute->dilation_w_ * kw + compute->stride_w_ - 1) / compute->stride_w_);

    out_w_end[kw] = NNACL_MIN(
      (compute->in_w_ + compute->pad_l_ - compute->dilation_w_ * kw + compute->stride_w_ - 1) / compute->stride_w_,
      compute->out_w_);

    num_pixels[kw] = out_w_end[kw] - out_w_start[kw];
    if (conv_dw->dw_param_.first_calc_kw_ == -1 && out_w_start[kw] == 0 && num_pixels[kw] == compute->out_w_) {
      conv_dw->dw_param_.first_calc_kw_ = kw;
    }
  }
  return NNACL_OK;
}

int ConvolutionDepthwisePrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);

  ConvolutionDepthwiseStruct *conv_dw = (ConvolutionDepthwiseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);

  ConvBaseUpdateOriginWeightAndBias(&conv_dw->conv_);

  if (self->train_session_) {
    TensorC *weight_tensor = self->in_[SECOND_INPUT];
    NNACL_CHECK_NULL_RETURN_ERR(weight_tensor);
    NNACL_CHECK_TRUE_RET(weight_tensor->shape_size_ == DIMENSION_4D, NNACL_CONVOLUTION_WEIGHT_SHAPE_INVALID);

    int weight_size_hw = GetHeight(weight_tensor) * GetWidth(weight_tensor);
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(GetBatch(weight_tensor), weight_size_hw, NNACL_ERR);
    int pack_weight_size = GetBatch(weight_tensor) * weight_size_hw;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(pack_weight_size, sizeof(float), NNACL_ERR);
    self->work_size_ = pack_weight_size * sizeof(float);
  }

  return ConvBaseInitConvWeightBias(&conv_dw->conv_);
}

int ConvolutionDepthwiseCompute(KernelBase *self) {
  ConvolutionDepthwiseStruct *conv_dw = (ConvolutionDepthwiseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);

  int ret = ConvBaseRepackWeight(&conv_dw->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  TensorC *input_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  conv_dw->input_ptr_ = (float *)input_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw->input_ptr_);

  TensorC *output_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);
  conv_dw->output_ptr_ = (float *)output_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw->output_ptr_);

  NNACL_CHECK_NULL_RETURN_ERR(conv_dw->dw_param_.num_pixels_);
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw->dw_param_.out_w_start_);
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw->dw_param_.out_w_end_);

  return self->env_->ParallelLaunch(self->env_->thread_pool_, ConvDwRun, self, self->thread_nr_);
}

int ConvolutionDepthwiseResize(KernelBase *self) {
  ConvolutionDepthwiseStruct *conv_dw = (ConvolutionDepthwiseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);

  int ret = ConvBasePrepare(&conv_dw->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  self->thread_nr_ = NNACL_MIN(self->thread_nr_, conv_dw->conv_.compute_.out_h_);
  NNACL_CHECK_ZERO_RETURN_ERR(self->thread_nr_);

  ret = ConvDwInitConvDwCalcInfo(conv_dw);
  if (ret != NNACL_OK) {
    return ret;
  }

  return NNACL_OK;
}

int ConvolutionDepthwiseRelease(KernelBase *self) {
  ConvolutionDepthwiseStruct *conv_dw = (ConvolutionDepthwiseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv_dw);

  ConvDwReleaseParam(conv_dw);

  ConvBaseRelease(&conv_dw->conv_);
  return NNACL_OK;
}

KernelBase *CreateConvDw(ConvParameter *conv) {
  ConvolutionDepthwiseStruct *conv_dw = (ConvolutionDepthwiseStruct *)malloc(sizeof(ConvolutionDepthwiseStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(conv_dw);
  memset(conv_dw, 0, sizeof(ConvolutionDepthwiseStruct));

  conv_dw->conv_.pack_weight_ = ConvDwPackWeight;
  conv_dw->conv_.malloc_weight_bias_ = ConvDwMallocWeightBiasData;
  conv_dw->conv_.base_.Prepare = ConvolutionDepthwisePrepare;
  conv_dw->conv_.base_.Compute = ConvolutionDepthwiseCompute;
  conv_dw->conv_.base_.Resize = ConvolutionDepthwiseResize;
  conv_dw->conv_.base_.Release = ConvolutionDepthwiseRelease;
  return (KernelBase *)conv_dw;
}
