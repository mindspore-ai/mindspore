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

#include "src/runtime/kernel/arm/fp32/convolution.h"
#include "src/runtime/kernel/arm/fp32/convolution_1x1.h"
#include "src/runtime/kernel/arm/fp32/convolution_winograd.h"
#include "nnacl/fp32/conv.h"
#include "nnacl/common_func.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "src/runtime/kernel/arm/base/dequant.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;

namespace mindspore::kernel {
int ConvolutionCPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  int kernel_h = filter_tensor->Height();
  int kernel_w = filter_tensor->Width();
  int in_channel = filter_tensor->Channel();
  int out_channel = filter_tensor->Batch();
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;
  int kernel_plane = kernel_h * kernel_w;
  const int oc_block = C8NUM;
  int oc_block_num = UP_DIV(out_channel, C8NUM);
  int pack_weight_size = oc_block_num * oc_block * in_channel * kernel_plane;

  auto origin_weight = reinterpret_cast<float *>(filter_tensor->MutableData());
  packed_weight_ = reinterpret_cast<float *>(malloc(pack_weight_size * sizeof(float)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed weight failed.";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, pack_weight_size * sizeof(float));
  RowMajor2Col8Major(origin_weight, packed_weight_, out_channel, in_channel * kernel_plane);

  bias_data_ = reinterpret_cast<float *>(malloc(oc_block_num * oc_block * sizeof(float)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, oc_block_num * oc_block * sizeof(float));

  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->MutableData());
    memcpy(bias_data_, ori_bias, out_channel * sizeof(float));
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  return RET_OK;
}

int ConvolutionCPUKernel::InitTmpBuffer() {
  int in_channel = conv_param_->input_channel_;
  MS_ASSERT(ctx_->allocator != nullptr);

#ifdef ENABLE_ARM32
  int unit_size = conv_param_->kernel_h_ * conv_param_->kernel_w_ * in_channel * C4NUM * thread_count_;
#else
  int unit_size = conv_param_->kernel_h_ * conv_param_->kernel_w_ * in_channel * C12NUM * thread_count_;
#endif
  packed_input_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(unit_size * sizeof(float)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed input failed.";
    return RET_ERROR;
  }

  col_major_input_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(unit_size * sizeof(float)));
  if (col_major_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc col_major_input_ failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionCPUKernel::Init() {
  auto ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionCPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }

  ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionCPUKernel::RunImpl(int task_id) {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto ori_input_data = reinterpret_cast<float *>(input_tensor->MutableData());
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->MutableData());
  ConvFp32(ori_input_data, packed_input_, packed_weight_, reinterpret_cast<float *>(bias_data_), col_major_input_,
           output_addr, task_id, conv_param_);
  return RET_OK;
}

int ConvolutionImpl(void *cdata, int task_id) {
  auto conv = reinterpret_cast<ConvolutionCPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionCPUKernel::Run() {
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }

  int error_code = ParallelLaunch(this->context_->thread_pool_, ConvolutionImpl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv error error_code[" << error_code << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  FreeTmpBuffer();
  return RET_OK;
}

kernel::LiteKernel *CpuConvFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                             const InnerContext *ctx, const kernel::KernelKey &desc,
                                             const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(op_parameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2D);
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  bool use_winograd = false;
  int out_unit;
  if (primitive != nullptr && primitive->GetInferFlag()) {
    conv_param->input_h_ = inputs.front()->Height();
    conv_param->input_w_ = inputs.front()->Width();
    conv_param->input_channel_ = inputs.front()->Channel();
    conv_param->output_h_ = outputs.front()->Height();
    conv_param->output_w_ = outputs.front()->Width();
    conv_param->output_channel_ = outputs.front()->Channel();
    conv_param->op_parameter_.thread_num_ = ctx->thread_num_;
    CheckIfUseWinograd(&use_winograd, &out_unit, conv_param);
  }

  auto *weight_tensor = inputs.at(kWeightIndex);
  auto *restore_data = weight_tensor->MutableData();
  if (weight_tensor->data_type() == kNumberTypeInt8 || weight_tensor->data_type() == kNumberTypeInt16) {
    auto *dequant_weight = kernel::DequantUtil::DequantWeight(weight_tensor);
    if (dequant_weight == nullptr) {
      MS_LOG(ERROR) << "dequant data is nullptr.";
      free(op_parameter);
      return nullptr;
    }
    weight_tensor->SetData(dequant_weight);
  }

  kernel::LiteKernel *kernel;
  if (kernel_h == 1 && kernel_w == 1) {
    kernel = new (std::nothrow) kernel::Convolution1x1CPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  } else if (use_winograd) {
    kernel =
      new (std::nothrow) kernel::ConvolutionWinogradCPUKernel(op_parameter, inputs, outputs, ctx, primitive, out_unit);
  } else {
    kernel = new (std::nothrow) kernel::ConvolutionCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    if (weight_tensor->data_type() == kNumberTypeInt8 || weight_tensor->data_type() == kNumberTypeInt16) {
      weight_tensor->FreeData();
      weight_tensor->SetData(restore_data);
    }
    free(op_parameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK && ret != RET_INFER_INVALID) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    if (weight_tensor->data_type() == kNumberTypeInt8 || weight_tensor->data_type() == kNumberTypeInt16) {
      weight_tensor->FreeData();
      weight_tensor->SetData(restore_data);
    }
    return nullptr;
  }

  if (weight_tensor->data_type() == kNumberTypeInt8 || weight_tensor->data_type() == kNumberTypeInt16) {
    weight_tensor->FreeData();
    weight_tensor->SetData(restore_data);
  }

  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Conv2D, CpuConvFp32KernelCreator)
}  // namespace mindspore::kernel
