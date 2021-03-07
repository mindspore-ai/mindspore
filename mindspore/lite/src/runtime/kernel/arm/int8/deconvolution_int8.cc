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

#include "src/runtime/kernel/arm/int8/deconvolution_int8.h"
#include "src/runtime/kernel/arm/int8/deconvolution_depthwise_int8.h"
#include "src/runtime/runtime_api.h"
#include "src/common/utils.h"
#include "src/runtime/kernel/arm/int8/opt_op_handler.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2dTransposeFusion;

namespace mindspore::kernel {
DeConvInt8CPUKernel::~DeConvInt8CPUKernel() {
  FreeTmpBuffer();
  ConvolutionBaseCPUKernel::FreeQuantParam();

  if (matmul_param_ != nullptr) {
    delete matmul_param_;
    matmul_param_ = nullptr;
  }
}

void DeConvInt8CPUKernel::FreeTmpBuffer() {
  if (weight_ptr_ != nullptr) {
    free(weight_ptr_);
    weight_ptr_ = nullptr;
  }
  if (input_ptr_ != nullptr) {
    free(input_ptr_);
    input_ptr_ = nullptr;
  }
  if (weight_sum_ != nullptr) {
    free(weight_sum_);
    weight_sum_ = nullptr;
  }
  return;
}

int DeConvInt8CPUKernel::ReSize() {
  FreeTmpBuffer();

  ConvolutionBaseCPUKernel::Init();
  int error_code = InitParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv int8 InitParam error!";
    return error_code;
  }

  error_code = InitBiasWeight();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv int8 InitBiasWeight error!";
    return error_code;
  }

  error_code = InitData();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv int8 InitData error!";
    return error_code;
  }
  return RET_OK;
}

int DeConvInt8CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }

  CheckSupportOptimize();

  int error_code = ConvolutionBaseCPUKernel::SetQuantParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv int8 SetQuantParam error!";
    return error_code;
  }
  return ReSize();
}

void DeConvInt8CPUKernel::CheckSupportOptimize() {
  support_optimize_ = true;
  matmul_func_ = MatMulInt8_16x4;
#ifdef ENABLE_ARM64
  if (mindspore::lite::IsSupportSDot()) {
    support_optimize_ = true;
    matmul_func_ = MatMulR4Int8_optimize_handler;
  } else {
    support_optimize_ = false;
    matmul_func_ = MatMulR4Int8Neon64;
  }
#endif
  return;
}

int DeConvInt8CPUKernel::InitParam() {
  matmul_param_ = new (std::nothrow) MatMulParameter();
  if (matmul_param_ == nullptr) {
    MS_LOG(ERROR) << "new MatMulParameter fail!";
    return RET_ERROR;
  }
  matmul_param_->row_ = conv_param_->input_h_ * conv_param_->input_w_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->col_ = conv_param_->output_channel_ * conv_param_->kernel_h_ * conv_param_->kernel_w_;

  int oc4 = UP_DIV(conv_param_->output_channel_, C4NUM);
  thread_count_ = MSMIN(op_parameter_->thread_num_, oc4);
  thread_stride_ = UP_DIV(oc4, thread_count_);
  return RET_OK;
}

int DeConvInt8CPUKernel::InitBiasWeight() {
  size_t size = UP_ROUND(conv_param_->output_channel_, C4NUM) * sizeof(int32_t);
  bias_data_ = malloc(size);
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "deconv int8 malloc bias_data_ error!";
    return RET_ERROR;
  }
  memset(bias_data_, 0, size);
  if (in_tensors_.size() == 3) {
    memcpy(bias_data_, in_tensors_.at(0)->MutableData(), conv_param_->output_channel_ * sizeof(int32_t));
  }

  size = UP_ROUND(conv_param_->output_channel_, C4NUM) * UP_ROUND(conv_param_->input_channel_, C16NUM) *
         conv_param_->kernel_w_ * conv_param_->kernel_h_ * sizeof(int8_t);
  weight_ptr_ = reinterpret_cast<int8_t *>(malloc(size));
  if (weight_ptr_ == nullptr) {
    MS_LOG(ERROR) << "deconv int8 malloc weight_ptr_ error!";
    return RET_ERROR;
  }
  memset(weight_ptr_, 0, size);
  DeConvWeightTransInt8(reinterpret_cast<int8_t *>(in_tensors_.at(1)->MutableData()), weight_ptr_,
                        conv_param_->input_channel_, conv_param_->output_channel_,
                        conv_param_->kernel_h_ * conv_param_->kernel_w_, support_optimize_);

  size = UP_ROUND(conv_param_->output_channel_, C4NUM) * conv_param_->kernel_h_ * conv_param_->kernel_w_;
  weight_sum_ = reinterpret_cast<int32_t *>(malloc(size * sizeof(int32_t)));
  if (weight_sum_ == nullptr) {
    MS_LOG(ERROR) << "deconv int8 malloc weight_sum_ error!";
    return RET_ERROR;
  }
  memset(weight_sum_, 0, size * sizeof(int32_t));
  DeConvPackWeightSum(weight_ptr_, weight_sum_, conv_param_->conv_quant_arg_.input_quant_args_[0].zp_,
                      conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_, matmul_param_->deep_, size,
                      support_optimize_);

  return RET_OK;
}

int DeConvInt8CPUKernel::InitData() {
  int size =
    UP_ROUND(conv_param_->input_h_ * conv_param_->input_w_, C4NUM) * UP_ROUND(conv_param_->input_channel_, C16NUM);
  input_ptr_ = reinterpret_cast<int8_t *>(malloc(size * sizeof(int8_t)));
  if (input_ptr_ == nullptr) {
    return RET_MEMORY_FAILED;
  }
  memset(input_ptr_, static_cast<int8_t>(conv_param_->conv_quant_arg_.input_quant_args_[0].zp_), size * sizeof(int8_t));

  return RET_OK;
}
int DeConvInt8CPUKernel::InitRunBuf() {
  int size = UP_ROUND(conv_param_->input_h_ * conv_param_->input_w_, C4NUM) *
             UP_ROUND(conv_param_->output_channel_, C4NUM) * conv_param_->kernel_w_ * conv_param_->kernel_h_;
  tmp_buffer_ = reinterpret_cast<int32_t *>(ctx_->allocator->Malloc(size * sizeof(int32_t)));
  if (tmp_buffer_ == nullptr) {
    return RET_MEMORY_FAILED;
  }

  size = UP_ROUND(conv_param_->output_channel_, C4NUM) * conv_param_->output_h_ * conv_param_->output_w_;
  tmp_output_ = reinterpret_cast<int32_t *>(ctx_->allocator->Malloc(size * sizeof(int32_t)));
  if (tmp_output_ == nullptr) {
    return RET_MEMORY_FAILED;
  }

  size = UP_ROUND(matmul_param_->row_, C4NUM);
  input_sum_ = reinterpret_cast<int32_t *>(ctx_->allocator->Malloc(size * sizeof(int32_t)));
  if (input_sum_ == nullptr) {
    return RET_MEMORY_FAILED;
  }

  return RET_OK;
}

void DeConvInt8CPUKernel::FreeRunBuf() {
  if (tmp_buffer_ != nullptr) {
    ctx_->allocator->Free(tmp_buffer_);
    tmp_buffer_ = nullptr;
  }
  if (tmp_output_ != nullptr) {
    ctx_->allocator->Free(tmp_output_);
    tmp_output_ = nullptr;
  }
  if (input_sum_ != nullptr) {
    ctx_->allocator->Free(input_sum_);
    input_sum_ = nullptr;
  }
  return;
}

int DeConvInt8Run(void *cdata, int task_id) {
  auto deconv = reinterpret_cast<DeConvInt8CPUKernel *>(cdata);
  auto error_code = deconv->DoDeconv(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DeConvInt8Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeConvInt8CPUKernel::DoDeconv(int task_id) {
  int cur_stride = thread_stride_;
  int res_stride = UP_DIV(conv_param_->output_channel_, C8NUM) - task_id * thread_stride_;
  int cur_oc = MSMIN(cur_stride, res_stride);

  cur_stride = thread_stride_ * C4NUM;
  res_stride = conv_param_->output_channel_ - task_id * thread_stride_ * C4NUM;
  int cur_oc_res = MSMIN(cur_stride, res_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  int input_plane = conv_param_->input_h_ * conv_param_->input_w_;
  int kernel_plane = conv_param_->kernel_w_ * conv_param_->kernel_h_;
  int output_plane = conv_param_->output_h_ * conv_param_->output_w_;

  DeConvInt8(input_ptr_, weight_ptr_ + task_id * thread_stride_ * C4NUM * kernel_plane * conv_param_->input_channel_,
             tmp_buffer_ + task_id * thread_stride_ * C4NUM * input_plane * kernel_plane, weight_sum_, input_sum_,
             UP_ROUND(matmul_param_->row_, C4NUM), cur_oc * C4NUM * kernel_plane,
             UP_ROUND(matmul_param_->deep_, C16NUM), conv_param_, matmul_func_);

  DeConvPostInt8(tmp_buffer_ + task_id * thread_stride_ * C4NUM * input_plane * kernel_plane,
                 reinterpret_cast<int32_t *>(bias_data_) + task_id * thread_stride_ * C4NUM,
                 tmp_output_ + task_id * thread_stride_ * C4NUM * output_plane,
                 output_ptr_ + task_id * thread_stride_ * C4NUM, cur_oc_res, conv_param_, support_optimize_);
  return RET_OK;
}

int DeConvInt8CPUKernel::Run() {
  int8_t *src_in = reinterpret_cast<int8_t *>(in_tensors_[0]->MutableData());
  int8_t *src_out = reinterpret_cast<int8_t *>(out_tensors_[0]->MutableData());

  int error_code = InitRunBuf();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv int8 InitRunBuf error! error_code[" << error_code << "]";
    FreeRunBuf();
    return RET_ERROR;
  }

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    RowMajor2Row16x4MajorInt8(src_in + batch_index * matmul_param_->row_ * conv_param_->input_channel_, input_ptr_,
                              matmul_param_->row_, matmul_param_->deep_);
    output_ptr_ = src_out + batch_index * matmul_param_->col_;

    DeConvPackInputSum(input_ptr_, input_sum_, conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_,
                       UP_ROUND(matmul_param_->row_, C4NUM), UP_ROUND(matmul_param_->deep_, C16NUM), support_optimize_);

    error_code = ParallelLaunch(this->context_->thread_pool_, DeConvInt8Run, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "deconv int8 run error! error_code[" << error_code << "]";
    }
  }
  FreeRunBuf();
  return error_code;
}

kernel::LiteKernel *CpuDeConvInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(op_parameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2dTransposeFusion);

  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  kernel::LiteKernel *kernel = nullptr;

  if (conv_param->group_ == 1) {
    kernel = new (std::nothrow) kernel::DeConvInt8CPUKernel(op_parameter, inputs, outputs, ctx);
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    kernel = new (std::nothrow) kernel::DeconvolutionDepthwiseInt8CPUKernel(op_parameter, inputs, outputs, ctx);
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

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Conv2dTransposeFusion, CpuDeConvInt8KernelCreator)
}  // namespace mindspore::kernel
