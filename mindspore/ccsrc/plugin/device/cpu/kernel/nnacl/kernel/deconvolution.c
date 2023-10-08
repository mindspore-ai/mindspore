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

#include "nnacl/kernel/deconvolution.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/kernel/deconvolution_winograd.h"
#include "nnacl/kernel/deconvolution_depthwise.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/deconv_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/matmul_avx_fp32.h"
#include "nnacl/kernel/default_kernel_base.h"

int DeConvMallocWeightBiasData(ConvolutionBaseStruct *conv) {
  int output_aligned_size = UP_ROUND(conv->compute_.out_c_, C8NUM) * sizeof(float);
  size_t pack_weight_size = conv->compute_.in_c_ * conv->compute_.kernel_hw_ * output_aligned_size;
  if (!conv->base_.train_session_) {
    conv->packed_weight_ = conv->base_.env_->Alloc(conv->base_.env_->allocator_, pack_weight_size);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->packed_weight_);
  }
  if (conv->bias_data_ == NULL) {
    conv->bias_data_ = conv->base_.env_->Alloc(conv->base_.env_->allocator_, output_aligned_size);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->bias_data_);
  }
  memset(conv->bias_data_, 0, output_aligned_size);
  return NNACL_OK;
}

void DeConvPackWeight(ConvolutionBaseStruct *conv) {
  TensorC *weight_tensor = conv->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_VOID(weight_tensor);
  void *weight_data = weight_tensor->data_ == NULL ? conv->origin_weight_ : weight_tensor->data_;
  NNACL_CHECK_NULL_RETURN_VOID(weight_data);

#ifdef ENABLE_AVX
  PackNHWCToCXHWNXFp32((float *)weight_data, (float *)conv->packed_weight_, conv->compute_.in_c_,
                       conv->compute_.kernel_hw_, conv->compute_.out_c_);
#else
  PackNHWCToC8HWN8Fp32((float *)weight_data, (float *)conv->packed_weight_, conv->compute_.in_c_,
                       conv->compute_.kernel_hw_, conv->compute_.out_c_);
#endif
}

int DeConvInitParam(DeConvStruct *deconv) {
  ConvComputeParam *compute = &deconv->conv_.compute_;
  deconv->matmul_.row_ = compute->in_hw_;
  deconv->matmul_.deep_ = compute->in_c_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(compute->out_c_, compute->kernel_hw_, NNACL_ERR);
  deconv->matmul_.col_ = compute->out_c_ * compute->kernel_hw_;
  deconv->matmul_.row_align_ = UP_ROUND(deconv->matmul_.row_, deconv->matmul_.row_tile_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(UP_ROUND(compute->out_c_, C8NUM), compute->kernel_hw_, NNACL_ERR);
  deconv->matmul_.col_align_ = UP_ROUND(compute->out_c_, C8NUM) * compute->kernel_hw_;

  deconv->conv_.base_.thread_nr_ = NNACL_MIN(deconv->conv_.base_.thread_nr_, UP_DIV(compute->out_c_, C8NUM));
  NNACL_CHECK_ZERO_RETURN_ERR(deconv->conv_.base_.thread_nr_);
#ifdef ENABLE_AVX
  deconv->thread_stride_ = UP_DIV(UP_DIV(compute->out_c_, C8NUM * C3NUM), deconv->conv_.base_.thread_nr_) * C3NUM;
#else
  deconv->thread_stride_ = UP_DIV(UP_DIV(compute->out_c_, C8NUM), deconv->conv_.base_.thread_nr_);
#endif
  return NNACL_OK;
}

int DeConvRun(void *cdata, int task_id, float l, float r) {
  DeConvStruct *deconv = (DeConvStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(deconv);

  int total_thead_stride_ = task_id * deconv->thread_stride_;
  int res_stride = UP_DIV(deconv->conv_.compute_.out_c_, C8NUM) - total_thead_stride_;
  int oc = NNACL_MIN(deconv->thread_stride_, res_stride);
  int cur_stride = deconv->thread_stride_ * C8NUM;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(total_thead_stride_, C8NUM, NNACL_ERR);
  int total_thead_stride_c8 = total_thead_stride_ * C8NUM;
  res_stride = deconv->conv_.compute_.out_c_ - total_thead_stride_c8;
  int oc_res = NNACL_MIN(cur_stride, res_stride);
  if (oc <= 0 || oc_res <= 0) {
    return NNACL_OK;
  }
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(total_thead_stride_c8, deconv->conv_.compute_.kernel_hw_, NNACL_ERR);
  int plane_thead_stride_c8 = total_thead_stride_c8 * deconv->conv_.compute_.kernel_hw_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(plane_thead_stride_c8, deconv->matmul_.row_align_, NNACL_ERR);
  int row_c8 = plane_thead_stride_c8 * deconv->matmul_.row_align_;
  float *tmp_buffer = deconv->tmp_buffer_ + row_c8;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(plane_thead_stride_c8, deconv->matmul_.deep_, NNACL_ERR);
  int deep_c8 = plane_thead_stride_c8 * deconv->matmul_.deep_;

#ifdef ENABLE_AVX
  DeconvMatmulAvx(deconv->pack_input_, (float *)deconv->conv_.packed_weight_ + deep_c8, tmp_buffer,
                  deconv->matmul_.deep_, deconv->matmul_.row_align_, oc * C8NUM * deconv->conv_.compute_.kernel_hw_,
                  deconv->conv_.compute_.kernel_hw_);
#elif ENABLE_SSE
  DeconvMatmulFloatSse(deconv->pack_input_, (float *)deconv->conv_.packed_weight_ + deep_c8, tmp_buffer,
                       deconv->matmul_.deep_, deconv->matmul_.row_align_,
                       oc * C8NUM * deconv->conv_.compute_.kernel_hw_);
#else
  MatMulOpt(deconv->pack_input_, (float *)deconv->conv_.packed_weight_ + deep_c8, tmp_buffer, NULL, ActType_No,
            deconv->matmul_.deep_, deconv->matmul_.row_align_, oc * C8NUM * deconv->conv_.compute_.kernel_hw_,
            deconv->matmul_.col_, OutType_C8);
#endif

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(total_thead_stride_c8, deconv->conv_.compute_.out_hw_, NNACL_OK);
  DeConvPostFp32C8(tmp_buffer, deconv->pack_output_ + total_thead_stride_c8 * deconv->conv_.compute_.out_hw_,
                   (float *)deconv->conv_.bias_data_ + total_thead_stride_c8,
                   deconv->output_ptr_ + total_thead_stride_c8, oc_res, (ConvParameter *)deconv->conv_.base_.param_);
  return NNACL_OK;
}

void DeConvFreeRunBuf(DeConvStruct *deconv) {
  ExecEnv *env = deconv->conv_.base_.env_;
  NNACL_CHECK_NULL_RETURN_VOID(env);

  if (deconv->pack_output_ != NULL) {
    env->Free(env->allocator_, deconv->pack_output_);
    deconv->pack_output_ = NULL;
  }
  if (deconv->tmp_buffer_ != NULL) {
    env->Free(env->allocator_, deconv->tmp_buffer_);
    deconv->tmp_buffer_ = NULL;
  }
  if (deconv->pack_input_ != NULL) {
    env->Free(env->allocator_, deconv->pack_input_);
    deconv->pack_input_ = NULL;
  }
}

int DeConvInitRunBuf(DeConvStruct *deconv) {
  ExecEnv *env = deconv->conv_.base_.env_;
  NNACL_CHECK_NULL_RETURN_ERR(env);

  int pack_output_size = UP_ROUND(deconv->conv_.compute_.out_c_, C8NUM) * deconv->conv_.compute_.out_hw_;
  deconv->pack_output_ = (float *)env->Alloc(env->allocator_, pack_output_size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(deconv->pack_output_);

  int tmp_buffer_size = deconv->matmul_.row_align_ * deconv->matmul_.col_align_;
  deconv->tmp_buffer_ = (float *)env->Alloc(env->allocator_, tmp_buffer_size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(deconv->tmp_buffer_);

  int pack_input_size = deconv->matmul_.row_align_ * deconv->matmul_.deep_;
  deconv->pack_input_ = (float *)env->Alloc(env->allocator_, pack_input_size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(deconv->pack_input_);

  return NNACL_OK;
}

int DeConvCheckvResizeValid(ConvolutionBaseStruct *conv) {
  // ===============check in channel================= //
  TensorC *input_tensor = conv->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  TensorC *filter_tensor = conv->base_.in_[SECOND_INPUT];

  int resize_out_channel = GetChannel(input_tensor);
  int filter_out_channel = GetBatch(filter_tensor);
  if (filter_out_channel != resize_out_channel) {
    return NNACL_DECONV_RESIZE_OC_INVALID;
  }
  return NNACL_OK;
}

int DeConvResize(KernelBase *self) {
  DeConvStruct *deconv = (DeConvStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(deconv);

  (void)ConvBaseUpdateComputeInfo(&deconv->conv_);

  int ret = DeConvCheckvResizeValid(&deconv->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = ConvBasePrepare(&deconv->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = DeConvInitParam(deconv);
  if (ret != NNACL_OK) {
    return ret;
  }

  return NNACL_OK;
}

int DeConvCompute(KernelBase *self) {
  DeConvStruct *deconv = (DeConvStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(deconv);

  int error_code = ConvBaseRepackWeight(&deconv->conv_);
  if (error_code != NNACL_OK) {
    return error_code;
  }

  error_code = DeConvInitRunBuf(deconv);
  if (error_code != NNACL_OK) {
    DeConvFreeRunBuf(deconv);
    return error_code;
  }

  float *src_in = (float *)self->in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(src_in);
  float *src_out = (float *)self->out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(src_out);

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(deconv->conv_.compute_.in_n_ - 1, deconv->conv_.compute_.in_c_, NNACL_ERR);
  int input_bc = (deconv->conv_.compute_.in_n_ - 1) * deconv->conv_.compute_.in_c_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(deconv->conv_.compute_.in_hw_, input_bc, NNACL_ERR);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(deconv->conv_.compute_.out_hw_, input_bc, NNACL_ERR);
  for (int batch_index = 0; batch_index < deconv->conv_.compute_.in_n_; batch_index++) {
    deconv->input_ptr_ = src_in + batch_index * deconv->conv_.compute_.in_hw_ * deconv->conv_.compute_.in_c_;
    deconv->output_ptr_ = src_out + batch_index * deconv->conv_.compute_.out_hw_ * deconv->conv_.compute_.out_c_;

#if defined(ENABLE_ARM32) || defined(ENABLE_SSE)
    RowMajor2Col4Major(deconv->input_ptr_, deconv->pack_input_, deconv->matmul_.row_, deconv->matmul_.deep_);
#else
    RowMajor2Col12Major(deconv->input_ptr_, deconv->pack_input_, deconv->matmul_.row_, deconv->matmul_.deep_);
#endif

    error_code = self->env_->ParallelLaunch(self->env_->thread_pool_, DeConvRun, self, self->thread_nr_);
    if (error_code != NNACL_OK) {
      DeConvFreeRunBuf(deconv);
      return error_code;
    }
  }

  DeConvFreeRunBuf(deconv);
  return NNACL_OK;
}

int DeConvPrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_ERR);
  DeConvStruct *deconv = (DeConvStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(deconv);
  ConvParameter *param = (ConvParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(param);

  // There could be weight dataType casting before Prepare, thus weight update is required.
  ConvBaseUpdateOriginWeightAndBias(&deconv->conv_);

#if defined(ENABLE_ARM32) || defined(ENABLE_AVX) || defined(ENABLE_SSE)
  deconv->matmul_.row_tile_ = C4NUM;
#else
  deconv->matmul_.row_tile_ = C12NUM;
#endif

  if (self->train_session_) {
    int output_aligned_size = UP_ROUND(deconv->conv_.compute_.out_c_, C8NUM);
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(deconv->conv_.compute_.in_c_, deconv->conv_.compute_.kernel_hw_, NNACL_ERR);
    int kernel_chw = deconv->conv_.compute_.in_c_ * deconv->conv_.compute_.kernel_hw_;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(kernel_chw, output_aligned_size, NNACL_ERR);
    size_t pack_weight_size = kernel_chw * output_aligned_size * sizeof(float);
    self->work_size_ = pack_weight_size;
  }

  if (self->in_[SECOND_INPUT]->data_ != NULL) {
    int error_code = ConvBaseInitConvWeightBias(&deconv->conv_);
    if (error_code != NNACL_OK) {
      return error_code;
    }
  } else {
    deconv->conv_.is_repack_ = true;
  }

  return NNACL_OK;
}

ConvolutionBaseStruct *CreateDeConv(ConvParameter *param) {
  DeConvStruct *deconv = (DeConvStruct *)malloc(sizeof(DeConvStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(deconv);
  memset(deconv, 0, sizeof(DeConvStruct));
  deconv->conv_.malloc_weight_bias_ = DeConvMallocWeightBiasData;
  deconv->conv_.pack_weight_ = DeConvPackWeight;
  deconv->conv_.base_.Prepare = DeConvPrepare;
  deconv->conv_.base_.Resize = DeConvResize;
  deconv->conv_.base_.Release = DefaultRelease;
  deconv->conv_.base_.Compute = DeConvCompute;
  return &deconv->conv_;
}

ConvolutionBaseStruct *SelectDeConv(ConvParameter *conv_param) {
#ifndef _WIN32
#ifndef ENABLE_MCU
  bool param_winograd_fit = (conv_param->stride_h_ > 1 || conv_param->stride_w_ > 1) &&
                            (conv_param->dilation_w_ == 1 && conv_param->dilation_h_ == 1);

#ifdef ENABLE_AVX
  bool in_size_winograd_fit = conv_param->input_w_ * conv_param->input_h_ >= NNACL_DECONV_WINOGRAD_HW_MAX;
  bool size_winograd_fit = (conv_param->kernel_w_ / conv_param->stride_w_ >= C2NUM ||
                            conv_param->kernel_h_ / conv_param->stride_h_ >= C2NUM || conv_param->output_channel_ == 1);
#else
  bool in_size_winograd_fit = true;
  bool size_winograd_fit =
    (conv_param->kernel_w_ / conv_param->stride_w_ > C2NUM || conv_param->kernel_h_ / conv_param->stride_h_ > C2NUM);
#endif

  if (param_winograd_fit && size_winograd_fit && in_size_winograd_fit) {
    ConvolutionBaseStruct *kernel = CreateDeConvWinograd(conv_param);
    if (kernel != NULL) {
      return kernel;
    }
  }
#endif
#endif

  return CreateDeConv(conv_param);
}

KernelBase *CreateConvolutionTranspose(OpParameter *param, int data_type) {
  ConvParameter *conv_param = (ConvParameter *)param;
  NNACL_CHECK_NULL_RETURN_NULL(conv_param);

  ConvolutionBaseStruct *conv = NULL;
  if (conv_param->group_ == 1 && conv_param->input_channel_ == 1 && conv_param->output_channel_ == 1) {
    conv = CreateDeConvDw(conv_param);
  } else if (conv_param->group_ == 1) {
    conv = SelectDeConv(conv_param);
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    conv = CreateDeConvDw(conv_param);
  }
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(conv);
  ConvBaseUpdateParamInfo(&conv->compute_, conv_param);
  return &conv->base_;
}

REG_KERNEL_CREATOR(PrimType_Conv2dTransposeFusion, kNumberTypeFloat32, CreateConvolutionTranspose)
