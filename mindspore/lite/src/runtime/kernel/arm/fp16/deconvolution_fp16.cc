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
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DeConv2D;

namespace mindspore::kernel {
DeConvolutionFp16CPUKernel::~DeConvolutionFp16CPUKernel() {
  FreeParam();
  if (matmul_param_ != nullptr) {
    delete matmul_param_;
    matmul_param_ = nullptr;
  }
  return;
}

int DeConvolutionFp16CPUKernel::ReSize() {
  FreeParam();
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

void DeConvolutionFp16CPUKernel::FreeParam() {
  if (pack_input_ != nullptr) {
    free(pack_input_);
    pack_input_ = nullptr;
  }
  return;
}

int DeConvolutionFp16CPUKernel::InitWeightBias() {
  bias_data_ = malloc(UP_ROUND(conv_param_->output_channel_, C4NUM) * sizeof(float16_t));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "deconv malloc bias_data_ error!";
    return RET_ERROR;
  }
  memset(bias_data_, 0, UP_ROUND(conv_param_->output_channel_, C4NUM) * sizeof(float16_t));
  if (in_tensors_.size() == 3) {
    Float32ToFloat16(reinterpret_cast<float *>(in_tensors_[2]->Data()), reinterpret_cast<float16_t *>(bias_data_),
                     conv_param_->output_channel_);
  }

  size_t weight_pack_size = conv_param_->input_channel_ * conv_param_->kernel_w_ * conv_param_->kernel_h_ *
                            UP_ROUND(conv_param_->output_channel_, C8NUM) * sizeof(float16_t);
  execute_weight_ = reinterpret_cast<float16_t *>(malloc(weight_pack_size));
  if (execute_weight_ == nullptr) {
    MS_LOG(ERROR) << "deconv malloc execute_weight_ error!";
    return RET_ERROR;
  }
  memset(execute_weight_, 0, weight_pack_size);
  PackNHWCFp32ToC8HWN8Fp16(reinterpret_cast<float *>(in_tensors_[1]->Data()), execute_weight_,
                           conv_param_->input_channel_, kernel_plane_, conv_param_->output_channel_);
  return RET_OK;
}

int DeConvolutionFp16CPUKernel::InitParam() {
  input_plane_ = conv_param_->input_h_ * conv_param_->input_w_;
  kernel_plane_ = conv_param_->kernel_w_ * conv_param_->kernel_h_;
  output_plane_ = conv_param_->output_h_ * conv_param_->output_w_;

  matmul_param_->row_ = input_plane_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->col_ = conv_param_->output_channel_ * kernel_plane_;
  row16_ = UP_ROUND(matmul_param_->row_, 16);
  col8_ = UP_ROUND(matmul_param_->col_, 8);

  thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(conv_param_->output_channel_, C8NUM));
  thread_stride_ = UP_DIV(UP_DIV(conv_param_->output_channel_, C8NUM), thread_count_);

  size_t size = row16_ * matmul_param_->deep_ * sizeof(float16_t);
  pack_input_ = reinterpret_cast<float16_t *>(malloc(size));
  if (pack_input_ == nullptr) {
    MS_LOG(ERROR) << "deconv Malloc pack_input_ error!";
    return RET_ERROR;
  }
  memset(pack_input_, 0, size);
  return RET_OK;
}

int DeConvolutionFp16CPUKernel::InitRunBuf() {
  pack_output_ = reinterpret_cast<float16_t *>(
    ctx_->allocator->Malloc(UP_ROUND(conv_param_->output_channel_, C8NUM) * output_plane_ * sizeof(float16_t)));
  if (pack_output_ == nullptr) {
    MS_LOG(ERROR) << "deconv Malloc pack_output_ error!";
    return RET_NULL_PTR;
  }

  tmp_buffer_ = reinterpret_cast<float16_t *>(ctx_->allocator->Malloc(row16_ * col8_ * sizeof(float16_t)));
  if (tmp_buffer_ == nullptr) {
    MS_LOG(ERROR) << "deconv Malloc tmp_buffer_ error!";
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
  return;
}

int DeConvFp16Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto deconv = reinterpret_cast<DeConvolutionFp16CPUKernel *>(cdata);
  auto error_code = deconv->DoDeconv(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DeConvFp16Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeConvolutionFp16CPUKernel::DoDeconv(int task_id) {
  int oc = MSMIN(thread_stride_, UP_DIV(conv_param_->output_channel_, C8NUM) - task_id * thread_stride_);
  int oc_res = MSMIN(thread_stride_ * C8NUM, conv_param_->output_channel_ - task_id * thread_stride_ * C8NUM);
  if (oc <= 0) {
    return RET_OK;
  }

  auto tmp_buf = tmp_buffer_ + task_id * thread_stride_ * C8NUM * kernel_plane_ * row16_;
  MatMulFp16(pack_input_, execute_weight_ + task_id * thread_stride_ * C8NUM * kernel_plane_ * matmul_param_->deep_,
             tmp_buf, nullptr, ActType_No, matmul_param_->deep_, matmul_param_->row_, oc * C8NUM * kernel_plane_, 0,
             false);
  DeConvPostFp16(tmp_buf, pack_output_ + task_id * thread_stride_ * C8NUM * output_plane_,
                 reinterpret_cast<float16_t *>(bias_data_) + task_id * thread_stride_ * C8NUM,
                 execute_output_ + task_id * thread_stride_ * C8NUM, oc_res, conv_param_);
  return RET_OK;
}

int DeConvolutionFp16CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int DeConvolutionFp16CPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  ConvolutionBaseFP16CPUKernel::GetExecuteTensor();

  int error_code = InitRunBuf();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv fp32 InitRunBuf error! error_code[" << error_code << "]";
    return RET_ERROR;
  }

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    RowMajor2Col16MajorFp16(execute_input_, pack_input_, input_plane_, conv_param_->input_channel_);

    error_code = LiteBackendParallelLaunch(DeConvFp16Run, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "deconv fp32 run error! error_code[" << error_code << "]";
      return RET_ERROR;
    }
  }

  ConvolutionBaseFP16CPUKernel::IfCastOutput();
  ConvolutionBaseFP16CPUKernel::FreeTmpBuffer();
  FreeRunBuf();

  return RET_OK;
}

kernel::LiteKernel *CpuDeConvFp16KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx,
                                               const kernel::KernelKey &desc,
                                               const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_DeConv2D);
  auto kernel = new (std::nothrow) DeConvolutionFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
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

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_DeConv2D, CpuDeConvFp16KernelCreator)
}  // namespace mindspore::kernel
