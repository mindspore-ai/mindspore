/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/deconvolution_fp32.h"
#include "src/litert/kernel/cpu/fp32/deconvolution_winograd_fp32.h"
#include "src/litert/kernel/cpu/fp32/deconvolution_depthwise_fp32.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2dTransposeFusion;

namespace mindspore::kernel {
DeConvolutionCPUKernel::~DeConvolutionCPUKernel() {
  if (matmul_param_ != nullptr) {
    delete matmul_param_;
    matmul_param_ = nullptr;
  }
}

int DeConvolutionCPUKernel::ReSize() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(conv_param_);
  CHECK_NULL_RETURN(matmul_param_);

  auto ret = ConvolutionBaseCPUKernel::CheckDeconvResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }

  auto weight_tensor = in_tensors_.at(kWeightIndex);
  CHECK_NULL_RETURN(weight_tensor);
  CHECK_NOT_EQUAL_RETURN(conv_param_->kernel_h_, weight_tensor->Height());
  CHECK_NOT_EQUAL_RETURN(conv_param_->kernel_w_, weight_tensor->Width());

  ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBaseCPUKernel init error!";
    return ret;
  }

  int error_code = InitParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv InitParam error!ret: " << error_code;
    return error_code;
  }
  return RET_OK;
}

int DeConvolutionCPUKernel::MallocWeightBiasData() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = weight_tensor->Batch();
  auto output_channel = weight_tensor->Channel();
  auto kernel_h_ = weight_tensor->Height();
  auto kernel_w_ = weight_tensor->Width();
  int output_aligned_size = UP_ROUND(output_channel, C8NUM);
  size_t pack_weight_size = input_channel * kernel_w_ * kernel_h_ * output_aligned_size * sizeof(float);
  if (!op_parameter_->is_train_session_) {
    packed_weight_ = MallocAlignedData(C32NUM, pack_weight_size);
    if (packed_weight_ == nullptr) {
      MS_LOG(ERROR) << "deconv malloc packed_weight_ error!";
      return RET_ERROR;
    }
  }
  if (bias_data_ == nullptr) {
    bias_data_ = MallocAlignedData(C32NUM, output_aligned_size * sizeof(float));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "deconv malloc bias_data_ error!";
      return RET_ERROR;
    }
  }
  memset(bias_data_, 0, output_aligned_size * sizeof(float));
  return RET_OK;
}

void DeConvolutionCPUKernel::PackWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = weight_tensor->Batch();
  auto output_channel = weight_tensor->Channel();
  auto kernel_h = weight_tensor->Height();
  auto kernel_w = weight_tensor->Width();
  void *origin_weight = IsTrainable() ? weight_tensor->data() : origin_weight_;
  MS_ASSERT(origin_weight != nullptr);
#ifdef ENABLE_AVX
  PackNHWCToCXHWNXFp32(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_),
                       input_channel, kernel_w * kernel_h, output_channel);
#else
  PackNHWCToC8HWN8Fp32(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_),
                       input_channel, kernel_w * kernel_h, output_channel);
#endif
}

int DeConvolutionCPUKernel::InitParam() {
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->input_h_, conv_param_->input_w_, RET_ERROR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->kernel_w_, conv_param_->kernel_h_, RET_ERROR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_h_, conv_param_->output_w_, RET_ERROR);
  input_plane_ = conv_param_->input_h_ * conv_param_->input_w_;
  kernel_plane_ = conv_param_->kernel_w_ * conv_param_->kernel_h_;
  output_plane_ = conv_param_->output_h_ * conv_param_->output_w_;

  matmul_param_->row_ = input_plane_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_channel_, kernel_plane_, RET_ERROR);
  matmul_param_->col_ = conv_param_->output_channel_ * kernel_plane_;
  matmul_param_->row_align_ = UP_ROUND(matmul_param_->row_, row_tile_);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(UP_ROUND(conv_param_->output_channel_, C8NUM), kernel_plane_, RET_ERROR);
  matmul_param_->col_8_ = UP_ROUND(conv_param_->output_channel_, C8NUM) * kernel_plane_;

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(conv_param_->output_channel_, C8NUM));
  NNACL_CHECK_ZERO_RETURN_ERR(thread_count_);

#ifdef ENABLE_AVX
  thread_stride_ = UP_DIV(UP_DIV(conv_param_->output_channel_, C8NUM * C3NUM), thread_count_) * C3NUM;
#else
  thread_stride_ = UP_DIV(UP_DIV(conv_param_->output_channel_, C8NUM), thread_count_);
#endif
  return RET_OK;
}

int DeConvFp32Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto deconv = reinterpret_cast<DeConvolutionCPUKernel *>(cdata);
  auto error_code = deconv->DoDeconv(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DeConvFp32Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeConvolutionCPUKernel::DoDeconv(int task_id) {
  MS_CHECK_INT_MUL_NOT_OVERFLOW(task_id, thread_stride_, RET_ERROR);
  int total_thead_stride_ = task_id * thread_stride_;
  int res_stride = UP_DIV(conv_param_->output_channel_, C8NUM) - total_thead_stride_;
  int oc = MSMIN(thread_stride_, res_stride);
  int cur_stride = thread_stride_ * C8NUM;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(total_thead_stride_, C8NUM, RET_ERROR);
  int total_thead_stride_c8 = total_thead_stride_ * C8NUM;
  res_stride = conv_param_->output_channel_ - total_thead_stride_c8;
  int oc_res = MSMIN(cur_stride, res_stride);
  if (oc <= 0 || oc_res <= 0) {
    return RET_OK;
  }
  MS_CHECK_INT_MUL_NOT_OVERFLOW(total_thead_stride_c8, kernel_plane_, RET_ERROR);
  int plane_thead_stride_c8 = total_thead_stride_c8 * kernel_plane_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(plane_thead_stride_c8, matmul_param_->row_align_, RET_ERROR);
  int row_c8 = plane_thead_stride_c8 * matmul_param_->row_align_;
  auto tmp_buffer = tmp_buffer_ + row_c8;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(plane_thead_stride_c8, matmul_param_->deep_, RET_ERROR);
  int deep_c8 = plane_thead_stride_c8 * matmul_param_->deep_;

#ifdef ENABLE_AVX
  DeconvMatmulAvx(pack_input_, reinterpret_cast<float *>(packed_weight_) + deep_c8, tmp_buffer, matmul_param_->deep_,
                  matmul_param_->row_align_, oc * C8NUM * kernel_plane_, kernel_plane_);
#elif ENABLE_SSE
  DeconvMatmulFloatSse(pack_input_, reinterpret_cast<float *>(packed_weight_) + deep_c8, tmp_buffer,
                       matmul_param_->deep_, matmul_param_->row_align_, oc * C8NUM * kernel_plane_);
#else
  MatMulOpt(pack_input_, reinterpret_cast<float *>(packed_weight_) + deep_c8, tmp_buffer, nullptr, ActType_No,
            matmul_param_->deep_, matmul_param_->row_align_, oc * C8NUM * kernel_plane_, matmul_param_->col_,
            OutType_C8);
#endif

  MS_CHECK_INT_MUL_NOT_OVERFLOW(total_thead_stride_c8, output_plane_, RET_ERROR);
  DeConvPostFp32C8(tmp_buffer, pack_output_ + total_thead_stride_c8 * output_plane_,
                   reinterpret_cast<float *>(bias_data_) + total_thead_stride_c8, output_ptr_ + total_thead_stride_c8,
                   oc_res, conv_param_);
  return RET_OK;
}

int DeConvolutionCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(conv_param_);
  CHECK_NULL_RETURN(in_tensors_.at(kInputIndex));
  CHECK_NULL_RETURN(in_tensors_.at(kWeightIndex));
  // There could be weight dataType casting before Prepare, thus weight update is required.
  UpdateOriginWeightAndBias();

#if defined(ENABLE_ARM32) || defined(ENABLE_AVX) || defined(ENABLE_SSE)
  row_tile_ = C4NUM;
#else
  row_tile_ = C12NUM;
#endif
  if (op_parameter_->is_train_session_) {
    auto weight_tensor = in_tensors_.at(kWeightIndex);
    auto input_channel = weight_tensor->Batch();
    auto output_channel = weight_tensor->Channel();
    auto kernel_h_ = weight_tensor->Height();
    auto kernel_w_ = weight_tensor->Width();
    int output_aligned_size = UP_ROUND(output_channel, C8NUM);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(kernel_w_, kernel_h_, RET_ERROR);
    int kernel_hw = kernel_w_ * kernel_h_;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(input_channel, kernel_hw, RET_ERROR);
    int kernel_chw = input_channel * kernel_hw;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(kernel_chw, output_aligned_size, RET_ERROR);
    size_t pack_weight_size = static_cast<size_t>(kernel_chw * output_aligned_size) * sizeof(float);
    set_workspace_size(pack_weight_size);
  }
  if (matmul_param_ == nullptr) {
    matmul_param_ = new (std::nothrow) MatMulParameter();
    if (matmul_param_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      return RET_ERROR;
    }
  }
  if (in_tensors_.at(kWeightIndex)->data() != nullptr) {
    int error_code = InitConvWeightBias();
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "deconv InitConvWeightBias error!ret: " << error_code;
      return error_code;
    }
  } else {
    is_repack_ = true;
    MS_LOG(WARNING) << "The weight is nullptr, will pack in runtime.";
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

void DeConvolutionCPUKernel::FreeRunBuf() {
  if (pack_output_ != nullptr) {
    ctx_->allocator->Free(pack_output_);
    pack_output_ = nullptr;
  }
  if (tmp_buffer_ != nullptr) {
    ctx_->allocator->Free(tmp_buffer_);
    tmp_buffer_ = nullptr;
  }
  if (pack_input_ != nullptr) {
    ctx_->allocator->Free(pack_input_);
    pack_input_ = nullptr;
  }
}

int DeConvolutionCPUKernel::InitRunBuf() {
  MS_CHECK_INT_MUL_NOT_OVERFLOW(UP_ROUND(conv_param_->output_channel_, C8NUM), output_plane_, RET_ERROR);
  pack_output_ = reinterpret_cast<float *>(
    ctx_->allocator->Malloc(UP_ROUND(conv_param_->output_channel_, C8NUM) * output_plane_ * sizeof(float)));
  if (pack_output_ == nullptr) {
    MS_LOG(ERROR) << "deconv Malloc pack_output_ error!";
    return RET_NULL_PTR;
  }

  MS_CHECK_INT_MUL_NOT_OVERFLOW(matmul_param_->row_align_, matmul_param_->col_8_, RET_ERROR);
  tmp_buffer_ = reinterpret_cast<float *>(
    ctx_->allocator->Malloc(matmul_param_->row_align_ * matmul_param_->col_8_ * sizeof(float)));
  if (tmp_buffer_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc tmp_buffer_ error!";
    return RET_NULL_PTR;
  }

  MS_CHECK_INT_MUL_NOT_OVERFLOW(matmul_param_->row_align_, matmul_param_->deep_, RET_ERROR);
  pack_input_ = reinterpret_cast<float *>(
    ctx_->allocator->Malloc(matmul_param_->row_align_ * matmul_param_->deep_ * sizeof(float)));
  if (pack_input_ == nullptr) {
    MS_LOG(ERROR) << "deconv Malloc pack_input_ error!";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeConvolutionCPUKernel::Run() {
  if (RepackWeight() != RET_OK) {
    MS_LOG(ERROR) << "Repack weight failed.";
    return RET_ERROR;
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  auto output_tensor = out_tensors_.at(kOutputIndex);
  float *src_in = reinterpret_cast<float *>(input_tensor->data());
  float *src_out = reinterpret_cast<float *>(output_tensor->data());
  CHECK_NULL_RETURN(src_in);
  CHECK_NULL_RETURN(src_out);

  int error_code = InitRunBuf();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv fp32 InitRunBuf error! error_code[" << error_code << "]";
    FreeRunBuf();
    return error_code;
  }

  MS_CHECK_INT_MUL_NOT_OVERFLOW((conv_param_->input_batch_ - 1), conv_param_->input_channel_, RET_ERROR);
  int input_bc = (conv_param_->input_batch_ - 1) * conv_param_->input_channel_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(input_plane_, input_bc, RET_ERROR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(output_plane_, input_bc, RET_ERROR);
  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    input_ptr_ = src_in + batch_index * input_plane_ * conv_param_->input_channel_;
    output_ptr_ = src_out + batch_index * output_plane_ * conv_param_->output_channel_;

#if defined(ENABLE_ARM32) || defined(ENABLE_SSE)
    RowMajor2Col4Major(input_ptr_, pack_input_, matmul_param_->row_, matmul_param_->deep_);
#else
    RowMajor2Col12Major(input_ptr_, pack_input_, matmul_param_->row_, matmul_param_->deep_);
#endif

    error_code = ParallelLaunch(this->ms_context_, DeConvFp32Run, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "deconv fp32 run error! error_code[" << error_code << "]";
      FreeRunBuf();
      return error_code;
    }
  }

  FreeRunBuf();
  return RET_OK;
}

kernel::LiteKernel *CpuNormDeconvFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                   const std::vector<lite::Tensor *> &outputs,
                                                   OpParameter *op_parameter, const InnerContext *ctx) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  kernel::LiteKernel *kernel = nullptr;
#ifdef ENABLE_AVX
  if ((conv_param->stride_h_ > 1 || conv_param->stride_w_ > 1) &&
      (conv_param->dilation_w_ == 1 && conv_param->dilation_h_ == 1) &&
      (conv_param->kernel_w_ / conv_param->stride_w_ >= C2NUM ||
       conv_param->kernel_h_ / conv_param->stride_h_ >= C2NUM || conv_param->output_channel_ == 1) &&
      conv_param->input_w_ * conv_param->input_h_ >= DECONV_WINOGRAD_MAX) {
    // output_channel_ = 1 is not appropriate in gemm deconv in x86
    kernel = new (std::nothrow) kernel::DeConvolutionWinogradCPUKernel(op_parameter, inputs, outputs,
                                                                       static_cast<const lite::InnerContext *>(ctx));
  } else {
    kernel = new (std::nothrow)
      kernel::DeConvolutionCPUKernel(op_parameter, inputs, outputs, static_cast<const lite::InnerContext *>(ctx));
  }
#else
  if ((conv_param->stride_h_ != 1 || conv_param->stride_w_ != 1) &&
      (conv_param->dilation_w_ == 1 && conv_param->dilation_h_ == 1) &&
      (conv_param->kernel_h_ / conv_param->stride_h_ > C2NUM ||
       conv_param->kernel_w_ / conv_param->stride_w_ > C2NUM)) {
    kernel = new (std::nothrow) kernel::DeConvolutionWinogradCPUKernel(op_parameter, inputs, outputs,
                                                                       static_cast<const lite::InnerContext *>(ctx));
  } else {
    kernel = new (std::nothrow)
      kernel::DeConvolutionCPUKernel(op_parameter, inputs, outputs, static_cast<const lite::InnerContext *>(ctx));
  }
#endif
  return kernel;
}

kernel::LiteKernel *CpuDeConvFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                               const InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_CHECK_TRUE_RET(op_parameter != nullptr, nullptr);
  MS_CHECK_TRUE_RET(ctx != nullptr, nullptr);

  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2dTransposeFusion);

  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  kernel::LiteKernel *kernel = nullptr;
  if (conv_param->group_ == 1 && conv_param->input_channel_ == 1 && conv_param->output_channel_ == 1) {
    kernel = new (std::nothrow) kernel::DeconvolutionDepthwiseCPUKernel(op_parameter, inputs, outputs,
                                                                        static_cast<const lite::InnerContext *>(ctx));
  } else if (conv_param->group_ == 1) {
    kernel =
      CpuNormDeconvFp32KernelCreator(inputs, outputs, op_parameter, static_cast<const lite::InnerContext *>(ctx));
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    kernel = new (std::nothrow) kernel::DeconvolutionDepthwiseCPUKernel(op_parameter, inputs, outputs,
                                                                        static_cast<const lite::InnerContext *>(ctx));
  } else {
    MS_LOG(ERROR) << "deconv do not support group deconv!";
    kernel = nullptr;
  }

  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    free(op_parameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Conv2dTransposeFusion, CpuDeConvFp32KernelCreator)
}  // namespace mindspore::kernel
