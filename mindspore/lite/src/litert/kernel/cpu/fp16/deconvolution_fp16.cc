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

#include "src/litert/kernel/cpu/fp16/deconvolution_fp16.h"
#include "src/litert/kernel/cpu/fp16/deconvolution_winograd_fp16.h"
#include "src/litert/kernel/cpu/fp16/deconvolution_depthwise_fp16.h"

using mindspore::kernel::KERNEL_ARCH;
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
  return;
}

int DeConvolutionFp16CPUKernel::ReSize() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(conv_param_);
  CHECK_NULL_RETURN(matmul_param_);

  auto ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBaseCPUKernel Prepare error!";
    return ret;
  }
  int error_code = InitParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv InitParam error!";
    return error_code;
  }
  return RET_OK;
}

void DeConvolutionFp16CPUKernel::PackWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = weight_tensor->Batch();
  auto output_channel = weight_tensor->Channel();
  auto kernel_h = weight_tensor->Height();
  auto kernel_w = weight_tensor->Width();
  void *origin_weight = (op_parameter_->is_train_session_) ? weight_tensor->data() : origin_weight_;
  CHECK_NULL_RETURN_VOID(origin_weight);
  PackNHWCFp16ToC8HWN8Fp16(reinterpret_cast<float16_t *>(origin_weight), reinterpret_cast<float16_t *>(packed_weight_),
                           input_channel, kernel_w * kernel_h, output_channel);
}

int DeConvolutionFp16CPUKernel::MallocWeightBiasData() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = weight_tensor->Batch();
  auto output_channel = weight_tensor->Channel();
  auto kernel_h = weight_tensor->Height();
  auto kernel_w = weight_tensor->Width();
  MS_CHECK_TRUE_RET(input_channel > 0 && output_channel > 0 && kernel_h > 0 && kernel_w > 0, RET_ERROR);
  size_t weight_pack_size = input_channel * kernel_w * kernel_h * UP_ROUND(output_channel, C8NUM) * sizeof(float16_t);
  if (!op_parameter_->is_train_session_) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, weight_pack_size);
    packed_weight_ = malloc(weight_pack_size);
    if (packed_weight_ == nullptr) {
      MS_LOG(ERROR) << "deconv malloc packed_weight_ error!";
      return RET_ERROR;
    }
    memset(packed_weight_, 0, weight_pack_size);
  }
  auto bias_size = UP_ROUND(output_channel, C8NUM) * sizeof(float16_t);
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, bias_size);
  bias_data_ = malloc(bias_size);
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "deconv malloc bias_data_ error!";
    return RET_ERROR;
  }
  memset(bias_data_, 0, UP_ROUND(output_channel, C8NUM) * sizeof(float16_t));
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

  thread_count_ = op_parameter_->thread_num_;
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

static int DeConvPreFp16Run(void *cdata, int task_id, float, float) {
  auto deconv = reinterpret_cast<DeConvolutionFp16CPUKernel *>(cdata);
  auto error_code = deconv->DoDeconvPre(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DoDeconvPre error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeConvolutionFp16CPUKernel::DoDeconvPre(int task_id) {
  int plan_stride = UP_DIV(matmul_param_->row_16_ / C16NUM, thread_count_) * C16NUM;
  int cur_plan_rest = input_plane_ - task_id * plan_stride;
  int plan = MSMIN(plan_stride, cur_plan_rest);
  if (plan <= 0) {
    return RET_OK;
  }

  float16_t *src_in = batch_input_ + task_id * plan_stride * conv_param_->input_channel_;
  float16_t *pack_in = pack_input_ + task_id * plan_stride * conv_param_->input_channel_;

  RowMajor2Col16MajorFp16Opt(src_in, pack_in, plan, conv_param_->input_channel_);
  return RET_OK;
}

static int DeConvFp16Run(void *cdata, int task_id, float, float) {
  auto deconv = reinterpret_cast<DeConvolutionFp16CPUKernel *>(cdata);
  auto error_code = deconv->DoDeconv(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DeConvFp16Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeConvolutionFp16CPUKernel::DoDeconv(int task_id) {
  int col_8 = UP_DIV(matmul_param_->col_8_, C8NUM);
  int stride = UP_DIV(col_8, thread_count_) * C8NUM;
  int cur_stride = matmul_param_->col_8_ - task_id * stride;
  int current_oc = MSMIN(stride, cur_stride);
  if (current_oc <= 0) {
    return RET_OK;
  }

  auto tmp_output = tmp_buffer_ + task_id * stride * matmul_param_->row_16_;
  auto tmp_weight = reinterpret_cast<float16_t *>(packed_weight_) + task_id * stride * matmul_param_->deep_;
  MatMulFp16(pack_input_, tmp_weight, tmp_output, nullptr, ActType_No, matmul_param_->deep_, matmul_param_->row_,
             current_oc, 0, OutType_C8);
  return RET_OK;
}

static int DeConvPostFp16Run(void *cdata, int task_id, float, float) {
  auto deconv = reinterpret_cast<DeConvolutionFp16CPUKernel *>(cdata);
  auto error_code = deconv->DoDeconvPost(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "DoDeconvPost error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeConvolutionFp16CPUKernel::DoDeconvPost(int task_id) {
  int oc8 = UP_DIV(conv_param_->output_channel_, C8NUM);
  int stride = UP_DIV(oc8, thread_count_) * C8NUM;
  int cur_stride = conv_param_->output_channel_ - task_id * stride;
  int cur_res = MSMIN(stride, cur_stride);
  if (cur_res <= 0) {
    return RET_OK;
  }

  auto tmp_buf = tmp_buffer_ + task_id * stride * kernel_plane_ * matmul_param_->row_16_;
  DeConvPostFp16(tmp_buf, pack_output_ + task_id * stride * output_plane_,
                 reinterpret_cast<float16_t *>(bias_data_) + task_id * stride, batch_output_ + task_id * stride,
                 cur_res, conv_param_);
  return RET_OK;
}

int DeConvolutionFp16CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(conv_param_);
  CHECK_NULL_RETURN(in_tensors_.at(kInputIndex));
  CHECK_NULL_RETURN(in_tensors_.at(kWeightIndex));
  UpdateOriginWeightAndBias();

  if (op_parameter_->is_train_session_) {
    auto weight_tensor = in_tensors_.at(kWeightIndex);
    auto input_channel = weight_tensor->Batch();
    auto output_channel = weight_tensor->Channel();
    auto kernel_h = weight_tensor->Height();
    auto kernel_w = weight_tensor->Width();
    size_t weight_pack_size = input_channel * kernel_w * kernel_h * UP_ROUND(output_channel, C8NUM) * sizeof(float16_t);
    set_workspace_size(weight_pack_size);
  }
  if (matmul_param_ == nullptr) {
    matmul_param_ = new (std::nothrow) MatMulParameter();
    if (matmul_param_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      return RET_ERROR;
    }
  }
  int ret = InitConvWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "deconv InitConvWeightBias error!";
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int DeConvolutionFp16CPUKernel::Run() {
  if (RepackWeight() != RET_OK) {
    MS_LOG(ERROR) << "Repack weight failed.";
    return RET_ERROR;
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  auto output_tensor = out_tensors_.at(kOutputIndex);
  auto *input_ptr = reinterpret_cast<float16_t *>(input_tensor->data());
  auto *output_ptr = reinterpret_cast<float16_t *>(output_tensor->data());
  CHECK_NULL_RETURN(input_ptr);
  CHECK_NULL_RETURN(output_ptr);

  int error_code = InitRunBuf();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "deconv fp16 InitRunBuf error! error_code[" << error_code << "]";
    FreeRunBuf();
    return RET_ERROR;
  }

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    batch_input_ = input_ptr + batch_index * conv_param_->input_channel_ * input_plane_;
    batch_output_ = output_ptr + batch_index * conv_param_->output_channel_ * output_plane_;

    error_code = ParallelLaunch(this->ms_context_, DeConvPreFp16Run, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "deconv fp16 pre run error! error_code[" << error_code << "]";
      FreeRunBuf();
      return error_code;
    }

    error_code = ParallelLaunch(this->ms_context_, DeConvFp16Run, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "deconv fp16 run error! error_code[" << error_code << "]";
      FreeRunBuf();
      return error_code;
    }

    error_code = ParallelLaunch(this->ms_context_, DeConvPostFp16Run, this, thread_count_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "deconv fp16 post run error! error_code[" << error_code << "]";
      FreeRunBuf();
      return error_code;
    }

    PackNC8HW8ToNHWCFp16(pack_output_, batch_output_, 1, conv_param_->output_w_ * conv_param_->output_h_,
                         conv_param_->output_channel_);
  }

  FreeRunBuf();
  return error_code;
}

kernel::LiteKernel *CpuDeConvFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_CHECK_TRUE_RET(op_parameter != nullptr, nullptr);
  MS_CHECK_TRUE_RET(ctx != nullptr, nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2dTransposeFusion);

  kernel::LiteKernel *kernel = nullptr;
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  if (conv_param->group_ == 1 && conv_param->input_channel_ == 1 && conv_param->output_channel_ == 1) {
    kernel = new (std::nothrow) DeconvolutionDepthwiseFp16CPUKernel(op_parameter, inputs, outputs, ctx);
  } else if (conv_param->group_ == 1) {
    if ((conv_param->stride_h_ != 1 && conv_param->stride_w_ != 1) &&
#ifndef ENABLE_ARM32
        (conv_param->kernel_h_ / conv_param->stride_h_ > C2NUM ||
         conv_param->kernel_w_ / conv_param->stride_w_ > C2NUM) &&
#endif
        (conv_param->dilation_h_ == 1 && conv_param->dilation_w_ == 1)) {
      kernel = new (std::nothrow) kernel::DeConvWinogradFp16CPUKernel(op_parameter, inputs, outputs, ctx);
    } else {
      kernel = new (std::nothrow) kernel::DeConvolutionFp16CPUKernel(op_parameter, inputs, outputs, ctx);
    }
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    kernel = new (std::nothrow) DeconvolutionDepthwiseFp16CPUKernel(op_parameter, inputs, outputs, ctx);
  }

  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    free(op_parameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Conv2dTransposeFusion, CpuDeConvFp16KernelCreator)
}  // namespace mindspore::kernel
