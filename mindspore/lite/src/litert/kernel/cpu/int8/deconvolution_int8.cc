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

#include "src/litert/kernel/cpu/int8/deconvolution_int8.h"
#include "src/litert/kernel/cpu/int8/deconvolution_depthwise_int8.h"
#include "src/common/utils.h"
#include "src/litert/kernel/cpu/int8/opt_op_handler.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2dTransposeFusion;

namespace mindspore::kernel {
DeConvInt8CPUKernel::~DeConvInt8CPUKernel() {
  FreeTmpBuffer();
  ConvolutionBaseCPUKernel::FreeQuantParam();

  if (weight_ptr_ != nullptr) {
    free(weight_ptr_);
    weight_ptr_ = nullptr;
  }
  if (weight_sum_ != nullptr) {
    free(weight_sum_);
    weight_sum_ = nullptr;
  }
  if (matmul_param_ != nullptr) {
    delete matmul_param_;
    matmul_param_ = nullptr;
  }
}

void DeConvInt8CPUKernel::FreeTmpBuffer() {
  if (input_ptr_ != nullptr) {
    free(input_ptr_);
    input_ptr_ = nullptr;
  }
  return;
}

int DeConvInt8CPUKernel::ReSize() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_NULL_RETURN(in_tensors_.at(kInputIndex));
  CHECK_NULL_RETURN(in_tensors_.at(kWeightIndex));
  CHECK_NULL_RETURN(conv_param_);
  CHECK_NULL_RETURN(matmul_param_);

  FreeTmpBuffer();

  int error_code = ConvolutionBaseCPUKernel::Prepare();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv int8 convolution base init failed.";
    return error_code;
  }
  error_code = InitParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv int8 InitParam error!";
    return error_code;
  }

  error_code = InitData();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv int8 InitData error!";
    return error_code;
  }
  return RET_OK;
}

int DeConvInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_NULL_RETURN(in_tensors_.at(kInputIndex));
  CHECK_NULL_RETURN(in_tensors_.at(kWeightIndex));
  CHECK_NULL_RETURN(conv_param_);
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      in_tensors_[1]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", input1 data_type is "
                  << in_tensors_[1]->data_type() << ", output data_type is " << out_tensors_[0]->data_type();
    return RET_ERROR;
  }

  matmul_param_ = new (std::nothrow) MatMulParameter();
  if (matmul_param_ == nullptr) {
    MS_LOG(ERROR) << "new MatMulParameter fail!";
    return RET_ERROR;
  }

  CheckSupportOptimize();

  int error_code = ConvolutionBaseCPUKernel::SetQuantParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv int8 SetQuantParam error!";
    return error_code;
  }

  error_code = InitBiasWeight();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv int8 InitBiasWeight error!";
    return error_code;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

void DeConvInt8CPUKernel::CheckSupportOptimize() {
  support_optimize_ = true;
  matmul_func_ = MatMulInt8_16x4;
#ifdef ENABLE_ARM64
#if !defined(SUPPORT_NNIE) && !defined(SUPPORT_34XX) && !defined(MACHINE_LINUX_ARM64)
  if (mindspore::lite::IsSupportSDot()) {
    support_optimize_ = true;
    matmul_func_ = MatMulR4Int8_optimize_handler;
  } else {
#endif
    support_optimize_ = false;
    matmul_func_ = MatMulR4Int8Neon64;
#if !defined(SUPPORT_NNIE) && !defined(SUPPORT_34XX) && !defined(MACHINE_LINUX_ARM64)
  }
#endif
#endif
  return;
}

int DeConvInt8CPUKernel::InitParam() {
  matmul_param_->row_ = conv_param_->input_h_ * conv_param_->input_w_;
  matmul_param_->deep_ = conv_param_->input_channel_;
  matmul_param_->col_ = conv_param_->output_channel_ * conv_param_->kernel_h_ * conv_param_->kernel_w_;

  int oc4 = UP_DIV(conv_param_->output_channel_, C4NUM);
  thread_count_ = MSMIN(op_parameter_->thread_num_, oc4);
  NNACL_CHECK_ZERO_RETURN_ERR(thread_count_);
  thread_stride_ = UP_DIV(oc4, thread_count_);
  return RET_OK;
}

int DeConvInt8CPUKernel::InitBiasWeight() {
  auto weight_tensor = in_tensors_.at(1);
  size_t size = UP_ROUND(weight_tensor->Channel(), C4NUM) * sizeof(int32_t);
  bias_data_ = malloc(size);
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "deconv int8 malloc bias_data_ error!";
    return RET_ERROR;
  }
  (void)memset(bias_data_, 0, size);
  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_.at(kBiasIndex);
    CHECK_NULL_RETURN(bias_tensor);
    auto ori_bias = bias_tensor->data();
    CHECK_NULL_RETURN(ori_bias);
    (void)memcpy(bias_data_, ori_bias, conv_param_->output_channel_ * sizeof(int32_t));
  }

  size_t weight_col_size = UP_ROUND(weight_tensor->Channel(), C4NUM) * weight_tensor->Height() * weight_tensor->Width();
  size_t weight_row_size = UP_ROUND(weight_tensor->Batch(), C16NUM);
  size_t pack_weight_size = weight_col_size * weight_row_size;

  weight_ptr_ = reinterpret_cast<int8_t *>(malloc(pack_weight_size * sizeof(int8_t)));
  if (weight_ptr_ == nullptr) {
    MS_LOG(ERROR) << "deconv int8 malloc weight_ptr_ error!";
    return RET_ERROR;
  }
  (void)memset(weight_ptr_, 0, pack_weight_size * sizeof(int8_t));
  DeConvWeightTransInt8(reinterpret_cast<int8_t *>(weight_tensor->data()), weight_ptr_, weight_tensor->Batch(),
                        weight_tensor->Channel(), weight_tensor->Height() * weight_tensor->Width(), support_optimize_);

  weight_sum_ = reinterpret_cast<int32_t *>(malloc(weight_col_size * sizeof(int32_t)));
  if (weight_sum_ == nullptr) {
    MS_LOG(ERROR) << "deconv int8 malloc weight_sum_ error!";
    return RET_ERROR;
  }
  (void)memset(weight_sum_, 0, weight_col_size * sizeof(int32_t));
  DeConvPackWeightSum(weight_ptr_, weight_sum_, conv_param_->conv_quant_arg_.input_quant_args_[0].zp_,
                      conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_, weight_tensor->Batch(), weight_col_size,
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
  (void)memset(input_ptr_, static_cast<int8_t>(conv_param_->conv_quant_arg_.input_quant_args_[0].zp_),
               size * sizeof(int8_t));

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

int DeConvInt8Run(void *cdata, int task_id, float, float) {
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

  auto error_code =
    DeConvInt8(input_ptr_, weight_ptr_ + task_id * thread_stride_ * C4NUM * kernel_plane * conv_param_->input_channel_,
               tmp_buffer_ + task_id * thread_stride_ * C4NUM * input_plane * kernel_plane, weight_sum_, input_sum_,
               UP_ROUND(matmul_param_->row_, C4NUM), cur_oc * C4NUM * kernel_plane,
               UP_ROUND(matmul_param_->deep_, C16NUM), conv_param_, matmul_func_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DeConvInt8 failed, error code: " << error_code;
    return error_code;
  }

  error_code =
    DeConvPostInt8(tmp_buffer_ + task_id * thread_stride_ * C4NUM * input_plane * kernel_plane,
                   reinterpret_cast<int32_t *>(bias_data_) + task_id * thread_stride_ * C4NUM,
                   tmp_output_ + task_id * thread_stride_ * C4NUM * output_plane,
                   output_ptr_ + task_id * thread_stride_ * C4NUM, cur_oc_res, conv_param_, support_optimize_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DeConvPostInt8 failed, error code: " << error_code;
    return error_code;
  }
  return RET_OK;
}

int DeConvInt8CPUKernel::Run() {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto output_tensor = out_tensors_.at(kOutputIndex);
  auto src_in = reinterpret_cast<int8_t *>(input_tensor->data());
  auto src_out = reinterpret_cast<int8_t *>(output_tensor->data());
  CHECK_NULL_RETURN(src_in);
  CHECK_NULL_RETURN(src_out);

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

    error_code = ParallelLaunch(this->ms_context_, DeConvInt8Run, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "deconv int8 run error! error_code[" << error_code << "]";
    }
  }
  FreeRunBuf();
  return RET_OK;
}

kernel::LiteKernel *CpuDeConvInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                               const lite::Context *ctx, const kernel::KernelKey &desc) {
  MS_CHECK_TRUE_RET(op_parameter != nullptr, nullptr);
  MS_CHECK_TRUE_RET(ctx != nullptr, nullptr);

  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2dTransposeFusion);

  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  kernel::LiteKernel *kernel = nullptr;

  if (conv_param->group_ == 1) {
    kernel = new (std::nothrow)
      kernel::DeConvInt8CPUKernel(op_parameter, inputs, outputs, static_cast<const lite::InnerContext *>(ctx));
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    kernel = new (std::nothrow) kernel::DeconvolutionDepthwiseInt8CPUKernel(
      op_parameter, inputs, outputs, static_cast<const lite::InnerContext *>(ctx));
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

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Conv2dTransposeFusion, CpuDeConvInt8KernelCreator)
}  // namespace mindspore::kernel
