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

#include "src/runtime/kernel/arm/fp16/convolution_delegate_fp16.h"
#include <vector>
#include "src/runtime/kernel/arm/fp32/convolution_delegate_fp32.h"
#include "src/runtime/kernel/arm/fp16/convolution_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_winograd_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_1x1_fp16.h"
#include "src/runtime/kernel/arm/fp16/group_convolution_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_depthwise_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_depthwise_slidewindow_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2DFusion;
using mindspore::schema::Format::Format_NHWC;

namespace mindspore::kernel {
void ConvolutionDelegateFP16CPUKernel::FreeCopiedData() {
  if ((origin_weight_ != nullptr) && (need_free_ & WEIGHT_NEED_FREE)) {
    free(origin_weight_);
    origin_weight_ = nullptr;
  }
  if ((origin_bias_ != nullptr) && (need_free_ & BIAS_NEED_FREE)) {
    free(origin_bias_);
    origin_bias_ = nullptr;
  }
}

void *ConvolutionDelegateFP16CPUKernel::CopyData(lite::Tensor *tensor) {
  auto data_type = tensor->data_type();
  if (data_type != kNumberTypeFloat32 && data_type != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Not supported data type: " << data_type;
    return nullptr;
  }
  auto copied_data = malloc(tensor->Size());
  if (copied_data == nullptr) {
    MS_LOG(ERROR) << "Malloc copied_data failed.";
    return nullptr;
  }
  memcpy(copied_data, tensor->data_c(), tensor->Size());
  return copied_data;
}

int ConvolutionDelegateFP16CPUKernel::Init() {
  if (!InferShapeDone()) {
    origin_weight_ = CopyData(in_tensors_.at(kWeightIndex));
    need_free_ = need_free_ | WEIGHT_NEED_FREE;
    if (in_tensors_.size() == 3) {
      origin_bias_ = CopyData(in_tensors_.at(kBiasIndex));
      need_free_ = need_free_ | BIAS_NEED_FREE;
    }
    return RET_OK;
  }
  origin_weight_ = in_tensors_.at(kWeightIndex)->data_c();
  if (in_tensors_.size() == 3) {
    origin_bias_ = in_tensors_.at(kBiasIndex)->data_c();
  }
  return ReSize();
}

static void SetInputOutputShapeInfo(ConvParameter *conv_param, lite::Tensor *input, lite::Tensor *output,
                                    const InnerContext *ctx) {
  conv_param->input_batch_ = input->Batch();
  conv_param->input_h_ = input->Height();
  conv_param->input_w_ = input->Width();
  conv_param->input_channel_ = input->Channel();
  conv_param->output_batch_ = output->Batch();
  conv_param->output_h_ = output->Height();
  conv_param->output_w_ = output->Width();
  conv_param->output_channel_ = output->Channel();
  conv_param->op_parameter_.thread_num_ = ctx->thread_num_;
}

int ConvolutionDelegateFP16CPUKernel::ReSize() {
  // Update shape info of input and output
  kernel::SetInputOutputShapeInfo(reinterpret_cast<ConvParameter *>(op_parameter_), in_tensors_.front(),
                                  out_tensors_.front(), context_);
  if (fp16_conv_kernel_ == nullptr) {
    fp16_conv_kernel_ = CpuConvFp16KernelSelect(in_tensors_, out_tensors_, op_parameter_, context_, origin_weight_,
                                                origin_bias_, origin_weight_data_type_, origin_bias_data_type_);
    if (fp16_conv_kernel_ == nullptr) {
      MS_LOG(ERROR) << "Selecting execute kernel failed for conv_kernel, got a nullptr.";
      return RET_ERROR;
    }
  }
  // copied weight and bias are not be used anymore,free them.
  FreeCopiedData();
  return fp16_conv_kernel_->ReSize();
}

ConvParameter *CreateNewConvParameterFp16(ConvParameter *parameter) {
  auto conv_parameter = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (conv_parameter == nullptr) {
    MS_LOG(ERROR) << "Malloc new conv parameter failed.";
    return nullptr;
  }
  memcpy(conv_parameter, parameter, sizeof(ConvParameter));
  return conv_parameter;
}

kernel::LiteKernel *CpuConvDwFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const InnerContext *ctx, void *origin_weight, void *origin_bias,
                                               TypeId origin_weight_data_type, TypeId origin_bias_data_type) {
  MS_ASSERT(opParameter != nullptr);
  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  kernel::LiteKernel *kernel = nullptr;
  if (conv_param->input_channel_ < 32) {
    kernel = new (std::nothrow) kernel::ConvolutionDepthwiseSWFp16CPUKernel(
      opParameter, inputs, outputs, ctx, origin_weight, origin_bias, origin_weight_data_type, origin_bias_data_type);
  } else {
    kernel = new (std::nothrow) kernel::ConvolutionDepthwiseFp16CPUKernel(
      opParameter, inputs, outputs, ctx, origin_weight, origin_bias, origin_weight_data_type, origin_bias_data_type);
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

kernel::LiteKernel *CpuConvFp16KernelSelect(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                            const lite::InnerContext *ctx, void *origin_weight, void *origin_bias,
                                            TypeId origin_weight_data_type, TypeId origin_bias_data_type) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  bool use_winograd = false;
  int out_unit;
  CheckIfUseWinogradFp16(&use_winograd, &out_unit, conv_param);
  kernel::LiteKernel *kernel = nullptr;

  if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    kernel = CpuConvDwFp16KernelCreator(inputs, outputs, op_parameter, ctx, origin_weight, origin_bias,
                                        origin_weight_data_type, origin_bias_data_type);
  } else if (conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1) {
    kernel = new (std::nothrow) kernel::Convolution1x1FP16CPUKernel(
      op_parameter, inputs, outputs, ctx, origin_weight, origin_bias, origin_weight_data_type, origin_bias_data_type);
  } else if (use_winograd) {
    kernel = new (std::nothrow)
      kernel::ConvolutionWinogradFP16CPUKernel(op_parameter, inputs, outputs, ctx, out_unit, origin_weight, origin_bias,
                                               origin_weight_data_type, origin_bias_data_type);
  } else {
    kernel = new (std::nothrow) kernel::ConvolutionFP16CPUKernel(
      op_parameter, inputs, outputs, ctx, origin_weight, origin_bias, origin_weight_data_type, origin_bias_data_type);
  }
  // Once kernel is selected, init func will invoke InitWeightAndBias
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "kernel init failed.";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

void FreeMemoryFp16(const std::vector<kernel::LiteKernel *> &group_convs, const std::vector<lite::Tensor *> &new_inputs,
                    const std::vector<lite::Tensor *> &new_outputs) {
  for (auto sub_conv : group_convs) {
    delete sub_conv;
  }
  for (auto in_tensor : new_inputs) {
    delete in_tensor;
  }
  for (auto out_tensor : new_outputs) {
    delete out_tensor;
  }
}

static lite::Tensor *CreateInputTensorFp16(TypeId data_type, const std::vector<int> &in_shape, bool infered_flag) {
  auto in_tensor = new (std::nothrow) lite::Tensor(data_type, in_shape, Format_NHWC, lite::Tensor::Category::VAR);
  if (in_tensor == nullptr) {
    MS_LOG(ERROR) << "new in_tensor failed.";
    return nullptr;
  }
  if (infered_flag) {
    auto ret = in_tensor->MallocData();
    if (ret != RET_OK) {
      delete in_tensor;
      MS_LOG(ERROR) << "in tensor malloc failed.";
      return nullptr;
    }
  }
  return in_tensor;
}

static lite::Tensor *CreateConstTensorFp16(lite::Tensor *tensor, const std::vector<int> &shape, const int index) {
  auto new_tensor =
    new (std::nothrow) lite::Tensor(tensor->data_type(), shape, Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  if (new_tensor == nullptr) {
    MS_LOG(ERROR) << "Create new_tensor failed.";
    return nullptr;
  }
  auto ret = new_tensor->MallocData();
  if (ret != RET_OK) {
    delete new_tensor;
    MS_LOG(ERROR) << "Malloc new_tensor failed.";
    return nullptr;
  }
  memcpy(new_tensor->data_c(), reinterpret_cast<char *>(tensor->data_c()) + index * new_tensor->Size(),
         new_tensor->Size());
  return new_tensor;
}

static lite::Tensor *CreateOutputTensorFp16(const std::vector<int> &out_shape,
                                            const std::vector<lite::Tensor *> &outputs, bool infered_flag, int index) {
  auto out_tensor = new (std::nothrow) lite::Tensor();
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "new tmp_out_tensor failed.";
    return nullptr;
  }
  out_tensor->set_data_type(mindspore::kNumberTypeFloat16);
  out_tensor->set_format(outputs.at(index)->format());
  if (infered_flag) {
    out_tensor->set_shape(out_shape);
    auto ret = out_tensor->MallocData();
    if (ret != RET_OK) {
      delete out_tensor;
      MS_LOG(ERROR) << "out_tensor malloc data failed.";
      return nullptr;
    }
  }
  return out_tensor;
}

kernel::LiteKernel *CreateDelegateConvFp16(const std::vector<lite::Tensor *> &inputs,
                                           const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                           const InnerContext *ctx) {
  auto weight_data_type = inputs.at(1)->data_type();
  TypeId bias_data_type = kTypeUnknown;
  if (inputs.size() == 3) {
    bias_data_type = inputs.at(2)->data_type();
  }
  return new (std::nothrow)
    kernel::ConvolutionDelegateFP16CPUKernel(op_parameter, inputs, outputs, ctx, weight_data_type, bias_data_type);
}

kernel::LiteKernel *CpuGroupConvFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                                  const InnerContext *ctx) {
  bool infer_flag = op_parameter->infer_flag_;
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  // update new shape info for each sub kernel
  int new_in_channel = inputs.at(kWeightIndex)->Channel();
  int new_out_channel = 0;
  if (conv_param->group_ == 0) {
    MS_LOG(ERROR) << "Divisor 'group' cannot be 0.";
    return nullptr;
  } else {
    new_out_channel = inputs.at(kWeightIndex)->Batch() / conv_param->group_;
  }

  std::vector<int> in_shape;
  std::vector<int> out_shape;
  if (infer_flag) {
    conv_param->input_channel_ = new_in_channel;
    conv_param->output_channel_ = new_out_channel;
    in_shape = {inputs.front()->Batch(), inputs.front()->Height(), inputs.front()->Width(), new_in_channel};
    out_shape = {inputs.front()->Batch(), outputs.front()->Height(), outputs.front()->Width(), new_out_channel};
  }
  std::vector<int> filter_shape = {new_out_channel, conv_param->kernel_h_, conv_param->kernel_w_, new_in_channel};
  std::vector<int> bias_shape = {new_out_channel};

  // new group conv op
  std::vector<kernel::LiteKernel *> group_convs;
  // create tensors for every sub conv kernel
  for (int i = 0; i < conv_param->group_; ++i) {
    std::vector<lite::Tensor *> new_inputs;
    std::vector<lite::Tensor *> new_outputs;
    auto new_conv_parameter = CreateNewConvParameterFp16(conv_param);
    if (new_conv_parameter == nullptr) {
      FreeMemoryFp16(group_convs, new_inputs, new_outputs);
      MS_LOG(ERROR) << "Get new conv parameter failed.";
      return nullptr;
    }
    // create new input for each group
    auto in_tensor = CreateInputTensorFp16(mindspore::kNumberTypeFloat16, in_shape, infer_flag);
    if (in_tensor == nullptr) {
      delete new_conv_parameter;
      FreeMemoryFp16(group_convs, new_inputs, new_outputs);
      MS_LOG(ERROR) << "create input tensor failed.";
      return nullptr;
    }
    new_inputs.emplace_back(in_tensor);

    // create new weight
    auto filter_tensor = CreateConstTensorFp16(inputs.at(kWeightIndex), filter_shape, i);
    if (filter_tensor == nullptr) {
      delete new_conv_parameter;
      FreeMemoryFp16(group_convs, new_inputs, new_outputs);
      MS_LOG(ERROR) << "create filter tensor failed.";
      return nullptr;
    }
    new_inputs.emplace_back(filter_tensor);

    // if has bias, create new bias
    if (inputs.size() == 3) {
      auto bias_tensor = CreateConstTensorFp16(inputs.at(kBiasIndex), bias_shape, i);
      if (bias_tensor == nullptr) {
        delete new_conv_parameter;
        FreeMemoryFp16(group_convs, new_inputs, new_outputs);
        MS_LOG(ERROR) << "create bias_tensor failed.";
        return nullptr;
      }
      new_inputs.emplace_back(bias_tensor);
    }

    // create new output tensors
    for (size_t j = 0; j < outputs.size(); ++j) {
      auto out_tensor = CreateOutputTensorFp16(out_shape, outputs, infer_flag, j);
      if (out_tensor == nullptr) {
        delete new_conv_parameter;
        FreeMemoryFp16(group_convs, new_inputs, new_outputs);
        MS_LOG(ERROR) << "new out_tensor failed.";
        return nullptr;
      }
      new_outputs.emplace_back(out_tensor);
    }
    group_convs.emplace_back(
      CreateDelegateConvFp16(new_inputs, new_outputs, reinterpret_cast<OpParameter *>(new_conv_parameter), ctx));
  }
  return new (std::nothrow)
    GroupConvolutionFP16CPUKernel(op_parameter, inputs, outputs, ctx, group_convs, conv_param->group_);
}

kernel::LiteKernel *CpuConvFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                             const InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2DFusion);

  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  kernel::LiteKernel *kernel = nullptr;
  bool is_depthwise =
    (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_);

  if (conv_param->group_ > 1 && !is_depthwise) {
    kernel = CpuGroupConvFp16KernelCreator(inputs, outputs, opParameter, ctx);
  } else {
    kernel = CreateDelegateConvFp16(inputs, outputs, opParameter, ctx);
  }

  if (kernel == nullptr) {
    MS_LOG(DEBUG) << "Create conv fp16 kernel failed.";
    free(opParameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(INFO) << "Init fp16 kernel failed, name: " << opParameter->name_
                 << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Conv2DFusion, CpuConvFp16KernelCreator)
}  // namespace mindspore::kernel
