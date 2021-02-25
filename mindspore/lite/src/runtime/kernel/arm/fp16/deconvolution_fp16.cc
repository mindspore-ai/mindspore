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

#include "src/runtime/kernel/arm/fp16/deconvolution_fp16.h"
#include "src/runtime/kernel/arm/fp16/deconvolution_winograd_fp16.h"
#include "src/runtime/kernel/arm/fp16/deconvolution_depthwise_fp16.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2dTransposeFusion;

namespace mindspore::kernel {
DeConvolutionFp16CPUKernel::~DeConvolutionFp16CPUKernel() {
  if (matmul_param_ != nullptr) {
    delete matmul_param_;
    matmul_param_ = nullptr;
  }
  if (execute_weight_ != nullptr) {
    free(execute_weight_);
    execute_weight_ = nullptr;
  }
  return;
}

int DeConvolutionFp16CPUKernel::ReSize() {
  ConvolutionBaseCPUKernel::Init();

  int error_code = InitParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv InitParam error!";
    return error_code;
  }
  return RET_OK;
}

int DeConvolutionFp16CPUKernel::InitWeightBias() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = weight_tensor->Batch();
  auto output_channel = weight_tensor->Channel();
  auto kernel_h = weight_tensor->Height();
  auto kernel_w = weight_tensor->Width();

  bias_data_ = malloc(UP_ROUND(output_channel, C4NUM) * sizeof(float16_t));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "deconv malloc bias_data_ error!";
    return RET_ERROR;
  }
  memset(bias_data_, 0, UP_ROUND(output_channel, C4NUM) * sizeof(float16_t));
  if (in_tensors_.size() == 3 && in_tensors_.at(kBiasIndex)->shape().size() == 1 &&
      in_tensors_.at(kBiasIndex)->DimensionSize(0) == output_channel) {
    Float32ToFloat16(reinterpret_cast<float *>(in_tensors_.at(2)->MutableData()),
                     reinterpret_cast<float16_t *>(bias_data_), output_channel);
  }

  size_t weight_pack_size = input_channel * kernel_w * kernel_h * UP_ROUND(output_channel, C8NUM) * sizeof(float16_t);
  execute_weight_ = reinterpret_cast<float16_t *>(malloc(weight_pack_size));
  if (execute_weight_ == nullptr) {
    MS_LOG(ERROR) << "deconv malloc execute_weight_ error!";
    return RET_ERROR;
  }
  memset(execute_weight_, 0, weight_pack_size);
  PackNHWCFp32ToC8HWN8Fp16(reinterpret_cast<float *>(in_tensors_.at(1)->MutableData()), execute_weight_, input_channel,
                           kernel_w * kernel_h, output_channel);
  return RET_OK;
}

int DeConvolutionFp16CPUKernel::InitParam() {
  input_plane_ = conv_param_->input_h_ * conv_param_->input_w_;
  kernel_plane_ = conv_param_->kernel_w_ * conv_param_->kernel_h_;
  output_plane_ = conv_param_->output_h_ * conv_param_->output_w_;

  matmul_param_->row_ = input_plane_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->col_ = conv_param_->output_channel_ * kernel_plane_;
  matmul_param_->row_16_ = UP_ROUND(matmul_param_->row_, C16NUM);
  matmul_param_->col_8_ = UP_ROUND(conv_param_->output_channel_, C8NUM) * kernel_plane_;

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(conv_param_->output_channel_, C8NUM));
  thread_stride_ = UP_DIV(UP_DIV(conv_param_->output_channel_, C8NUM), thread_count_);
  return RET_OK;
}

int DeConvolutionFp16CPUKernel::InitRunBuf() {
  pack_output_ = reinterpret_cast<float16_t *>(
    ctx_->allocator->Malloc(UP_ROUND(conv_param_->output_channel_, C8NUM) * output_plane_ * sizeof(float16_t)));
  if (pack_output_ == nullptr) {
    MS_LOG(ERROR) << "deconv Malloc pack_output_ error!";
    return RET_NULL_PTR;
  }

  tmp_buffer_ = reinterpret_cast<float16_t *>(
    ctx_->allocator->Malloc(matmul_param_->row_16_ * matmul_param_->col_8_ * sizeof(float16_t)));
  if (tmp_buffer_ == nullptr) {
    MS_LOG(ERROR) << "deconv Malloc tmp_buffer_ error!";
    return RET_ERROR;
  }

  pack_input_ =
    reinterpret_cast<float16_t *>(malloc(matmul_param_->row_16_ * matmul_param_->deep_ * sizeof(float16_t)));
  if (pack_input_ == nullptr) {
    MS_LOG(ERROR) << "deconv Malloc pack_input_ error!";
    return RET_ERROR;
  }
  return RET_OK;
}

void DeConvolutionFp16CPUKernel::FreeRunBuf() {
  if (tmp_buffer_ != nullptr) {
    ctx_->allocator->Free(tmp_buffer_);
    tmp_buffer_ = nullptr;
  }
  if (pack_output_ != nullptr) {
    ctx_->allocator->Free(pack_output_);
    pack_output_ = nullptr;
  }
  if (pack_input_ != nullptr) {
    ctx_->allocator->Free(pack_input_);
    pack_input_ = nullptr;
  }
  return;
}

static int DeConvFp16Run(void *cdata, int task_id) {
  auto deconv = reinterpret_cast<DeConvolutionFp16CPUKernel *>(cdata);
  auto error_code = deconv->DoDeconv(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DeConvFp16Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeConvolutionFp16CPUKernel::DoDeconv(int task_id) {
  int cur_stride = UP_DIV(conv_param_->output_channel_, C8NUM) - task_id * thread_stride_;
  int oc = MSMIN(thread_stride_, cur_stride);
  cur_stride = conv_param_->output_channel_ - task_id * thread_stride_ * C8NUM;
  int oc_res = MSMIN(thread_stride_ * C8NUM, cur_stride);
  if (oc <= 0) {
    return RET_OK;
  }

  auto tmp_buf = tmp_buffer_ + task_id * thread_stride_ * C8NUM * kernel_plane_ * matmul_param_->row_16_;
  MatMulFp16(pack_input_, execute_weight_ + task_id * thread_stride_ * C8NUM * kernel_plane_ * matmul_param_->deep_,
             tmp_buf, nullptr, ActType_No, matmul_param_->deep_, matmul_param_->row_, oc * C8NUM * kernel_plane_, 0,
             OutType_C8);

  DeConvPostFp16(tmp_buf, pack_output_ + task_id * thread_stride_ * C8NUM * output_plane_,
                 reinterpret_cast<float16_t *>(bias_data_) + task_id * thread_stride_ * C8NUM,
                 batch_output_ + task_id * thread_stride_ * C8NUM, oc_res, conv_param_);
  return RET_OK;
}

int DeConvolutionFp16CPUKernel::Init() {
  matmul_param_ = new (std::nothrow) MatMulParameter();
  if (matmul_param_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  int ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "deconv InitWeightBias error!";
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int DeConvolutionFp16CPUKernel::Run() {
  ConvolutionBaseFP16CPUKernel::GetExecuteTensor();

  int error_code = InitRunBuf();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv fp16 InitRunBuf error! error_code[" << error_code << "]";
    FreeRunBuf();
    return RET_ERROR;
  }

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    batch_input_ = execute_input_ + batch_index * conv_param_->input_channel_ * input_plane_;
    batch_output_ = execute_output_ + batch_index * conv_param_->output_channel_ * output_plane_;

    RowMajor2Col16MajorFp16Opt(batch_input_, pack_input_, input_plane_, conv_param_->input_channel_);

    error_code = ParallelLaunch(this->context_->thread_pool_, DeConvFp16Run, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "deconv fp16 run error! error_code[" << error_code << "]";
    }
  }

  FreeRunBuf();
  return error_code;
}

kernel::LiteKernel *CpuDeConvFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(op_parameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2dTransposeFusion);

  auto weight_data_type = inputs.at(1)->data_type();
  TypeId bias_data_type = kTypeUnknown;
  if (inputs.size() == 3) {
    bias_data_type = inputs.at(2)->data_type();
  }
  kernel::LiteKernel *kernel = nullptr;
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  if (conv_param->group_ == 1) {
    if ((conv_param->stride_h_ != 1 || conv_param->stride_w_ != 1) &&
        (conv_param->dilation_h_ == 1 && conv_param->dilation_w_ == 1)) {
      kernel = new (std::nothrow)
        kernel::DeConvWinogradFp16CPUKernel(op_parameter, inputs, outputs, ctx, weight_data_type, bias_data_type);
    } else {
      kernel = new (std::nothrow)
        kernel::DeConvolutionFp16CPUKernel(op_parameter, inputs, outputs, ctx, weight_data_type, bias_data_type);
    }
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    kernel = new (std::nothrow)
      DeconvolutionDepthwiseFp16CPUKernel(op_parameter, inputs, outputs, ctx, weight_data_type, bias_data_type);
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

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Conv2dTransposeFusion, CpuDeConvFp16KernelCreator)
}  // namespace mindspore::kernel
