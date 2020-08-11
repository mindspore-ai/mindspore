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

#include "src/runtime/kernel/arm/fp32/deconvolution.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DeConv2D;

namespace mindspore::kernel {
DeConvolutionCPUKernel::~DeConvolutionCPUKernel() {
  if (weight_ptr_ != nullptr) {
    free(weight_ptr_);
    weight_ptr_ = nullptr;
  }
  if (tmp_buffer_ != nullptr) {
    free(tmp_buffer_);
    tmp_buffer_ = nullptr;
  }
  if (pack_input_ != nullptr) {
    free(pack_input_);
    pack_input_ = nullptr;
  }
  if (pack_output_ != nullptr) {
    free(pack_output_);
    pack_output_ = nullptr;
  }
  return;
}

int DeConvolutionCPUKernel::ReSize() {
  if (tmp_buffer_ != nullptr) {
    free(tmp_buffer_);
    tmp_buffer_ = nullptr;
  }
  if (pack_input_ != nullptr) {
    free(pack_input_);
    pack_input_ = nullptr;
  }
  if (pack_output_ != nullptr) {
    free(pack_output_);
    pack_output_ = nullptr;
  }
  InitParam();

  return RET_OK;
}

int DeConvolutionCPUKernel::InitWeightBias() {
  bias_data_ = malloc(UP_ROUND(conv_param_->output_channel_, C4NUM) * sizeof(float));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "deconv malloc bias_data_ error!";
    return RET_ERROR;
  }
  memset(bias_data_, 0, UP_ROUND(conv_param_->output_channel_, C4NUM) * sizeof(float));
  if (in_tensors_.size() == 3) {
    memcpy(bias_data_, in_tensors_[2]->Data(), conv_param_->output_channel_ * sizeof(float));
  }

  size_t weight_pack_size = conv_param_->input_channel_ * conv_param_->kernel_w_ * conv_param_->kernel_h_ *
                            UP_ROUND(conv_param_->output_channel_, C8NUM) * sizeof(float);
  weight_ptr_ = reinterpret_cast<float *>(malloc(weight_pack_size));
  if (weight_ptr_ == nullptr) {
    MS_LOG(ERROR) << "deconv malloc weight_ptr_ error!";
    return RET_ERROR;
  }
  memset(weight_ptr_, 0, weight_pack_size);
  PackNHWCToC8HWN8Fp32(reinterpret_cast<float *>(in_tensors_[1]->Data()), weight_ptr_, conv_param_->input_channel_,
                       kernel_plane_, conv_param_->output_channel_);
  return RET_OK;
}

int DeConvolutionCPUKernel::InitParam() {
  input_plane_ = conv_param_->input_h_ * conv_param_->input_w_;
  kernel_plane_ = conv_param_->kernel_w_ * conv_param_->kernel_h_;
  output_plane_ = conv_param_->output_h_ * conv_param_->output_w_;

  matmul_param_->row_ = input_plane_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->col_ = conv_param_->output_channel_ * kernel_plane_;
  matmul_param_->row_8_ = UP_ROUND(matmul_param_->row_, C8NUM);
  matmul_param_->col_8_ = UP_ROUND(conv_param_->output_channel_, C8NUM) * kernel_plane_;

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(conv_param_->output_channel_, C8NUM));
  thread_stride_ = UP_DIV(UP_DIV(conv_param_->output_channel_, C8NUM), thread_count_);

  pack_input_ = reinterpret_cast<float *>(malloc(matmul_param_->row_8_ * matmul_param_->deep_ * sizeof(float)));
  if (pack_input_ == nullptr) {
    MS_LOG(ERROR) << "deconv Malloc pack_input_ error!";
    return RET_ERROR;
  }

  pack_output_ =
    reinterpret_cast<float *>(malloc(UP_ROUND(conv_param_->output_channel_, C8NUM) * output_plane_ * sizeof(float)));
  if (pack_output_ == nullptr) {
    MS_LOG(ERROR) << "deconv Malloc pack_output_ error!";
    return RET_NULL_PTR;
  }

  tmp_buffer_ = reinterpret_cast<float *>(malloc(matmul_param_->row_8_ * matmul_param_->col_8_ * sizeof(float)));
  if (tmp_buffer_ == nullptr) {
    MS_LOG(ERROR) << "Conv1x1 Malloc tmp_buffer_ error!";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeConvFp32Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto deconv = reinterpret_cast<DeConvolutionCPUKernel *>(cdata);
  auto error_code = deconv->DoDeconv(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DeConvFp32Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeConvolutionCPUKernel::DoDeconv(int task_id) {
  int oc = MSMIN(thread_stride_, UP_DIV(conv_param_->output_channel_, C8NUM) - task_id * thread_stride_);
  int oc_res = MSMIN(thread_stride_ * C8NUM, conv_param_->output_channel_ - task_id * thread_stride_ * C8NUM);
  if (oc <= 0 || oc_res <= 0) {
    return RET_OK;
  }

  auto tmp_buffer = tmp_buffer_ + task_id * thread_stride_ * C8NUM * kernel_plane_ * matmul_param_->row_8_;
  MatMul(pack_input_, weight_ptr_ + task_id * thread_stride_ * C8NUM * kernel_plane_ * matmul_param_->deep_, tmp_buffer,
         nullptr, ActType_No, matmul_param_->deep_, matmul_param_->row_8_, oc * C8NUM * kernel_plane_,
         matmul_param_->col_, false);

  DeConvPostFp32C8x8(tmp_buffer, pack_output_ + task_id * thread_stride_ * C8NUM * output_plane_,
                     reinterpret_cast<float *>(bias_data_) + thread_stride_ * task_id * C8NUM,
                     output_ptr_ + task_id * thread_stride_ * C8NUM, oc_res, conv_param_);
  return RET_OK;
}

int DeConvolutionCPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  ConvolutionBaseCPUKernel::Init();

  int error_code = InitParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv InitParam error!";
    return error_code;
  }

  error_code = InitWeightBias();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv InitWeightBias error!";
    return error_code;
  }
  return RET_OK;
}

int DeConvolutionCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  float *src_in = reinterpret_cast<float *>(in_tensors_[0]->Data());
  float *src_out = reinterpret_cast<float *>(out_tensors_[0]->Data());

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    input_ptr_ = src_in + batch_index * input_plane_ * conv_param_->input_channel_;
    output_ptr_ = src_out + batch_index * output_plane_ * conv_param_->output_channel_;

    RowMajor2Col8Major(input_ptr_, pack_input_, input_plane_, conv_param_->input_channel_);

    int error_code = LiteBackendParallelLaunch(DeConvFp32Run, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "deconv fp32 run error! error_code[" << error_code << "]";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

kernel::LiteKernel *CpuDeConvFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx,
                                               const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_DeConv2D);
  auto kernel = new (std::nothrow) kernel::DeConvolutionCPUKernel(opParameter, inputs, outputs, ctx, primitive);
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_DeConv2D, CpuDeConvFp32KernelCreator)
}  // namespace mindspore::kernel
