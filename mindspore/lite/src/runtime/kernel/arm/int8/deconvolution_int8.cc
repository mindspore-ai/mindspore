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
#include "src/runtime/kernel/arm/nnacl/quantization/fixed_point.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DeConv2D;

namespace mindspore::kernel {
DeConvInt8CPUKernel::~DeConvInt8CPUKernel() {
  if (weight_ptr_ != nullptr) {
    free(weight_ptr_);
    weight_ptr_ = nullptr;
  }
  if (tmp_buffer_ != nullptr) {
    free(tmp_buffer_);
    tmp_buffer_ = nullptr;
  }
  if (input_ptr_ != nullptr) {
    free(input_ptr_);
    input_ptr_ = nullptr;
  }
  if (tmp_output_ != nullptr) {
    free(tmp_output_);
    tmp_output_ = nullptr;
  }
  ConvolutionBaseCPUKernel::FreeQuantParam();
}

int DeConvInt8CPUKernel::ReSize() { return RET_OK; }

int DeConvInt8CPUKernel::InitParam() {
  fc_param_ = new MatMulParameter();
  fc_param_->row_ = conv_param_->input_h_ * conv_param_->input_w_;
  fc_param_->deep_ = conv_param_->input_channel_;
  fc_param_->col_ = conv_param_->output_channel_ * conv_param_->kernel_h_ * conv_param_->kernel_w_;
  fc_param_->row_8_ = UP_ROUND(fc_param_->row_, C8NUM);
  fc_param_->col_8_ = UP_ROUND(conv_param_->output_channel_, C8NUM) * conv_param_->kernel_h_ * conv_param_->kernel_w_;

  size_t oc8 = UP_DIV(conv_param_->output_channel_, C8NUM);
  thread_count_ = MSMIN(op_parameter_->thread_num_, oc8);
  thread_stride_ = UP_DIV(oc8, thread_count_) * C8NUM;
  return RET_OK;
}

int DeConvInt8CPUKernel::InitBiasWeight() {
  if (in_tensors_.size() == 3) {
    size_t size = UP_ROUND(conv_param_->output_channel_, C8NUM) * sizeof(int32_t);
    bias_data_ = malloc(size);
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "deconv int8 malloc bias_data_ error!";
      return RET_ERROR;
    }
    memset(bias_data_, 0, size);
    memcpy(bias_data_, in_tensors_[0]->Data(), conv_param_->output_channel_ * sizeof(int32_t));
  } else {
    bias_data_ = nullptr;
  }

  /* weight:  ichwoc(nhwc)  ->  oc8 * h * w * inc * 8 */
  size_t size = conv_param_->kernel_w_ * conv_param_->kernel_h_ * UP_ROUND(conv_param_->output_channel_, C8NUM) *
                conv_param_->input_channel_ * sizeof(int8_t);
  weight_ptr_ = reinterpret_cast<int8_t *>(malloc(size));
  if (weight_ptr_ == nullptr) {
    MS_LOG(ERROR) << "deconv int8 malloc weight_ptr_ error!";
    return RET_ERROR;
  }
  memset(weight_ptr_, 0, size);
  PackNHWCToC8HWN8Int8(in_tensors_[1]->Data(), weight_ptr_, conv_param_->input_channel_,
                       conv_param_->kernel_h_ * conv_param_->kernel_w_, conv_param_->output_channel_);
  return RET_OK;
}

int DeConvInt8CPUKernel::InitData() {
  int size = UP_ROUND(conv_param_->input_h_ * conv_param_->input_w_, C8NUM) * conv_param_->input_channel_;
  input_ptr_ = reinterpret_cast<int8_t *>(malloc(size * sizeof(int8_t)));
  if (input_ptr_ == nullptr) {
    return RET_MEMORY_FAILED;
  }
  memset(input_ptr_, 0, size * sizeof(int8_t));

  size = UP_ROUND(conv_param_->input_h_ * conv_param_->input_w_, C8NUM) *
         UP_ROUND(conv_param_->output_channel_, C8NUM) * conv_param_->kernel_w_ * conv_param_->kernel_h_;
  tmp_buffer_ = reinterpret_cast<int32_t *>(malloc(size * sizeof(int32_t)));
  if (tmp_buffer_ == nullptr) {
    return RET_MEMORY_FAILED;
  }

  size = UP_ROUND(conv_param_->output_channel_, C8NUM) * conv_param_->output_h_ * conv_param_->output_w_;
  tmp_output_ = reinterpret_cast<int32_t *>(malloc(size * sizeof(int32_t)));
  if (tmp_output_ == nullptr) {
    return RET_MEMORY_FAILED;
  }
  return RET_OK;
}

int DeConvInt8CPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  ConvolutionBaseCPUKernel::Init();
  int error_code = ConvolutionBaseCPUKernel::SetQuantParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv int8 SetQuantParam error!";
    return error_code;
  }

  error_code = InitParam();
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

int DeConvInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto deconv = reinterpret_cast<DeConvInt8CPUKernel *>(cdata);
  auto error_code = deconv->DoDeconv(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DeConvInt8Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeConvInt8PostFuncRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto deconv = reinterpret_cast<DeConvInt8CPUKernel *>(cdata);
  auto error_code = deconv->DoPostFunc(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DeConvInt8PostFuncRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeConvInt8CPUKernel::DoDeconv(int task_id) {
  int cur_oc = MSMIN(thread_stride_, UP_ROUND(conv_param_->output_channel_, C8NUM) - task_id * thread_stride_);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  int input_plane = conv_param_->input_h_ * conv_param_->input_w_;
  int kernel_plane = conv_param_->kernel_w_ * conv_param_->kernel_h_;

  DeConvInt8(input_ptr_, weight_ptr_ + task_id * thread_stride_ * kernel_plane * conv_param_->input_channel_,
             tmp_buffer_ + task_id * thread_stride_ * input_plane * kernel_plane, fc_param_->row_8_,
             cur_oc * kernel_plane, fc_param_->deep_, conv_param_);

  return RET_OK;
}

int DeConvInt8CPUKernel::DoPostFunc(int task_id) {
  int input_plane = conv_param_->input_h_ * conv_param_->input_w_;
  int kernel_plane = conv_param_->kernel_w_ * conv_param_->kernel_h_;
  int output_plane = conv_param_->output_h_ * conv_param_->output_w_;

  int cur_oc = MSMIN(thread_stride_, conv_param_->output_channel_ - task_id * thread_stride_);
  if (cur_oc <= 0) {
    return RET_OK;
  }

  DeConvPostInt8(tmp_buffer_ + task_id * thread_stride_ * input_plane * kernel_plane,
                 reinterpret_cast<int32_t *>(bias_data_) + task_id * thread_stride_,
                 tmp_output_ + task_id * thread_stride_ * output_plane, output_ptr_ + task_id * thread_stride_, cur_oc,
                 conv_param_);
  return RET_OK;
}

int DeConvInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  int8_t *src_in = reinterpret_cast<int8_t *>(in_tensors_[0]->Data());
  int8_t *src_out = reinterpret_cast<int8_t *>(out_tensors_[0]->Data());

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    RowMajor2Col8MajorInt8(src_in + batch_index * fc_param_->row_ * conv_param_->input_channel_, input_ptr_,
                           fc_param_->row_, fc_param_->deep_);
    output_ptr_ = src_out + batch_index * fc_param_->col_;

    int error_code = LiteBackendParallelLaunch(DeConvInt8Run, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "deconv int8 run error! error_code[" << error_code << "]";
      return RET_ERROR;
    }
    error_code = LiteBackendParallelLaunch(DeConvInt8PostFuncRun, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "deconv int8 post run error! error_code[" << error_code << "]";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

kernel::LiteKernel *CpuDeConvInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx,
                                               const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_DeConv2D);
  auto kernel = new (std::nothrow) kernel::DeConvInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_DeConv2D, CpuDeConvInt8KernelCreator)
}  // namespace mindspore::kernel
