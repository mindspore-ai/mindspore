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

#include "src/runtime/kernel/arm/fp32/deconvolution_fp32.h"
#include "src/runtime/kernel/arm/fp32/deconvolution_winograd_fp32.h"
#include "src/runtime/kernel/arm/fp32/deconvolution_depthwise_fp32.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
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
  if (weight_ptr_ != nullptr) {
    free(weight_ptr_);
    weight_ptr_ = nullptr;
  }
}

int DeConvolutionCPUKernel::ReSize() {
  ConvolutionBaseCPUKernel::Init();

  int error_code = InitParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv InitParam error!ret: " << error_code;
    return error_code;
  }
  return RET_OK;
}

int DeConvolutionCPUKernel::InitWeightBias() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = weight_tensor->Batch();
  auto output_channel = weight_tensor->Channel();
  auto kernel_h_ = weight_tensor->Height();
  auto kernel_w_ = weight_tensor->Width();

  bias_data_ = malloc(UP_ROUND(output_channel, C4NUM) * sizeof(float));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "deconv malloc bias_data_ error!";
    return RET_ERROR;
  }
  memset(bias_data_, 0, UP_ROUND(output_channel, C4NUM) * sizeof(float));
  if (in_tensors_.size() == 3 && in_tensors_.at(kBiasIndex)->shape().size() == 1 &&
      in_tensors_.at(kBiasIndex)->DimensionSize(0) == output_channel) {
    memcpy(bias_data_, in_tensors_.at(2)->MutableData(), output_channel * sizeof(float));
  }

  size_t weight_pack_size = input_channel * kernel_w_ * kernel_h_ * UP_ROUND(output_channel, C8NUM) * sizeof(float);
  weight_ptr_ = reinterpret_cast<float *>(malloc(weight_pack_size));
  if (weight_ptr_ == nullptr) {
    MS_LOG(ERROR) << "deconv malloc weight_ptr_ error!";
    return RET_ERROR;
  }
  memset(weight_ptr_, 0, weight_pack_size);
  PackNHWCToC8HWN8Fp32(reinterpret_cast<float *>(in_tensors_.at(1)->MutableData()), weight_ptr_, input_channel,
                       kernel_w_ * kernel_h_, output_channel);
  return RET_OK;
}

int DeConvolutionCPUKernel::InitParam() {
  input_plane_ = conv_param_->input_h_ * conv_param_->input_w_;
  kernel_plane_ = conv_param_->kernel_w_ * conv_param_->kernel_h_;
  output_plane_ = conv_param_->output_h_ * conv_param_->output_w_;

  matmul_param_->row_ = input_plane_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->col_ = conv_param_->output_channel_ * kernel_plane_;
  matmul_param_->row_12_ = UP_ROUND(matmul_param_->row_, C12NUM);
  matmul_param_->row_4_ = UP_ROUND(matmul_param_->row_, C4NUM);
  matmul_param_->col_8_ = UP_ROUND(conv_param_->output_channel_, C8NUM) * kernel_plane_;

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(conv_param_->output_channel_, C8NUM));
  thread_stride_ = UP_DIV(UP_DIV(conv_param_->output_channel_, C8NUM), thread_count_);
  return RET_OK;
}

int DeConvFp32Run(void *cdata, int task_id) {
  auto deconv = reinterpret_cast<DeConvolutionCPUKernel *>(cdata);
  auto error_code = deconv->DoDeconv(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DeConvFp32Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeConvolutionCPUKernel::DoDeconv(int task_id) {
  int res_stride = UP_DIV(conv_param_->output_channel_, C8NUM) - task_id * thread_stride_;
  int oc = MSMIN(thread_stride_, res_stride);
  int cur_stride = thread_stride_ * C8NUM;
  res_stride = conv_param_->output_channel_ - task_id * thread_stride_ * C8NUM;
  int oc_res = MSMIN(cur_stride, res_stride);
  if (oc <= 0 || oc_res <= 0) {
    return RET_OK;
  }

#if defined(ENABLE_ARM32) || defined(ENABLE_SSE)
  auto tmp_buffer = tmp_buffer_ + task_id * thread_stride_ * C8NUM * kernel_plane_ * matmul_param_->row_4_;
  MatMulOpt(pack_input_, weight_ptr_ + task_id * thread_stride_ * C8NUM * kernel_plane_ * matmul_param_->deep_,
            tmp_buffer, nullptr, ActType_No, matmul_param_->deep_, matmul_param_->row_4_, oc * C8NUM * kernel_plane_,
            matmul_param_->col_, OutType_C8);
#else
  auto tmp_buffer = tmp_buffer_ + task_id * thread_stride_ * C8NUM * kernel_plane_ * matmul_param_->row_12_;
  MatMulOpt(pack_input_, weight_ptr_ + task_id * thread_stride_ * C8NUM * kernel_plane_ * matmul_param_->deep_,
            tmp_buffer, nullptr, ActType_No, matmul_param_->deep_, matmul_param_->row_12_, oc * C8NUM * kernel_plane_,
            matmul_param_->col_, OutType_C8);
#endif

  DeConvPostFp32C8(tmp_buffer, pack_output_ + task_id * thread_stride_ * C8NUM * output_plane_,
                   reinterpret_cast<float *>(bias_data_) + thread_stride_ * task_id * C8NUM,
                   output_ptr_ + task_id * thread_stride_ * C8NUM, oc_res, conv_param_);
  return RET_OK;
}

int DeConvolutionCPUKernel::Init() {
  matmul_param_ = new (std::nothrow) MatMulParameter();
  if (matmul_param_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  int error_code = InitWeightBias();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv InitWeightBias error!ret: " << error_code;
    return error_code;
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
  return;
}

int DeConvolutionCPUKernel::InitRunBuf() {
  pack_output_ = reinterpret_cast<float *>(
    ctx_->allocator->Malloc(UP_ROUND(conv_param_->output_channel_, C8NUM) * output_plane_ * sizeof(float)));
  if (pack_output_ == nullptr) {
    MS_LOG(ERROR) << "deconv Malloc pack_output_ error!";
    return RET_NULL_PTR;
  }

#if defined(ENABLE_ARM32) || defined(ENABLE_SSE)
  tmp_buffer_ =
    reinterpret_cast<float *>(ctx_->allocator->Malloc(matmul_param_->row_4_ * matmul_param_->col_8_ * sizeof(float)));
#else
  tmp_buffer_ =
    reinterpret_cast<float *>(ctx_->allocator->Malloc(matmul_param_->row_12_ * matmul_param_->col_8_ * sizeof(float)));
#endif
  if (tmp_buffer_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc tmp_buffer_ error!";
    return RET_NULL_PTR;
  }

#if defined(ENABLE_ARM32) || defined(ENABLE_SSE)
  pack_input_ =
    reinterpret_cast<float *>(ctx_->allocator->Malloc(matmul_param_->row_4_ * matmul_param_->deep_ * sizeof(float)));
#else
  pack_input_ =
    reinterpret_cast<float *>(ctx_->allocator->Malloc(matmul_param_->row_12_ * matmul_param_->deep_ * sizeof(float)));
#endif
  if (pack_input_ == nullptr) {
    MS_LOG(ERROR) << "deconv Malloc pack_input_ error!";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeConvolutionCPUKernel::Run() {
  float *src_in = reinterpret_cast<float *>(in_tensors_[0]->MutableData());
  float *src_out = reinterpret_cast<float *>(out_tensors_[0]->MutableData());

  int error_code = InitRunBuf();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv fp32 InitRunBuf error! error_code[" << error_code << "]";
    FreeRunBuf();
    return error_code;
  }

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    input_ptr_ = src_in + batch_index * input_plane_ * conv_param_->input_channel_;
    output_ptr_ = src_out + batch_index * output_plane_ * conv_param_->output_channel_;

#if defined(ENABLE_ARM32) || defined(ENABLE_SSE)
    RowMajor2Col4Major(input_ptr_, pack_input_, matmul_param_->row_, matmul_param_->deep_);
#else
    RowMajor2Col12Major(input_ptr_, pack_input_, matmul_param_->row_, matmul_param_->deep_);
#endif

    error_code = ParallelLaunch(this->context_->thread_pool_, DeConvFp32Run, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "deconv fp32 run error! error_code[" << error_code << "]";
      FreeRunBuf();
      return error_code;
    }
  }

  FreeRunBuf();
  return RET_OK;
}

kernel::LiteKernel *CpuDeConvFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(op_parameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2dTransposeFusion);

  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  kernel::LiteKernel *kernel = nullptr;
  if (conv_param->group_ == 1) {
    if ((conv_param->stride_h_ != 1 || conv_param->stride_w_ != 1) &&
        (conv_param->dilation_w_ == 1 && conv_param->dilation_h_ == 1)) {
      kernel = new (std::nothrow) kernel::DeConvolutionWinogradCPUKernel(op_parameter, inputs, outputs, ctx);
    } else {
      kernel = new (std::nothrow) kernel::DeConvolutionCPUKernel(op_parameter, inputs, outputs, ctx);
    }
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    kernel = new (std::nothrow) kernel::DeconvolutionDepthwiseCPUKernel(op_parameter, inputs, outputs, ctx);
  } else {
    MS_LOG(ERROR) << "deconv do not support group deconv!";
    kernel = nullptr;
  }

  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    free(op_parameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Conv2dTransposeFusion, CpuDeConvFp32KernelCreator)
}  // namespace mindspore::kernel
