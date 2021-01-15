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
#include "src/runtime/kernel/arm/fp32/convolution_delegate_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_1x1_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_winograd_fp32.h"
#include "src/runtime/kernel/arm/fp32/group_convolution_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;
using mindspore::schema::Format::Format_NHWC;

namespace mindspore::kernel {
float *ConvolutionDelegateCPUKernel::CopyData(lite::Tensor *tensor) {
  auto data = reinterpret_cast<float *>(malloc(tensor->Size()));
  if (data == nullptr) {
    MS_LOG(ERROR) << "Malloc data failed.";
    return nullptr;
  }
  memcpy(data, tensor->data_c(), tensor->Size());
  return data;
}

void ConvolutionDelegateCPUKernel::FreeCopiedData() {
  if (origin_weight_ != nullptr && need_free_weight_) {
    free(origin_weight_);
    origin_weight_ = nullptr;
  }
  if (origin_bias_ != nullptr && need_free_bias_) {
    free(origin_bias_);
    origin_bias_ = nullptr;
  }
}

int ConvolutionDelegateCPUKernel::GetWeightAndBias() {
  auto ret = GetWeightData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get weight data failed.";
    return ret;
  }
  ret = GetBiasData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get bias data failed.";
    return ret;
  }
  return RET_OK;
}

int ConvolutionDelegateCPUKernel::GetWeightData() {
  if (InferShapeDone()) {
    origin_weight_ = reinterpret_cast<float *>(in_tensors_.at(kWeightIndex)->data_c());
    return RET_OK;
  } else {
    origin_weight_ = CopyData(in_tensors_.at(kWeightIndex));
    if (origin_weight_ == nullptr) {
      MS_LOG(ERROR) << "Copy weight data failed.";
      return RET_ERROR;
    }
    need_free_weight_ = true;
    return RET_OK;
  }
  return RET_OK;
}

int ConvolutionDelegateCPUKernel::GetBiasData() {
  if (in_tensors_.size() == 3) {
    if (InferShapeDone()) {
      origin_bias_ = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->data_c());
      return RET_OK;
    } else {
      origin_bias_ = CopyData(in_tensors_.at(kBiasIndex));
      if (origin_bias_ == nullptr) {
        MS_LOG(ERROR) << "Copy bias data failed.";
        return RET_ERROR;
      }
      need_free_bias_ = true;
      return RET_OK;
    }
  }
  return RET_OK;
}

int ConvolutionDelegateCPUKernel::Init() {
  auto ret = GetWeightAndBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get weight and bias failed.";
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionDelegateCPUKernel::ReSize() {
  // Update shape info of input and output
  SetInputOutputShapeInfo(reinterpret_cast<ConvParameter *>(op_parameter_), in_tensors_.front(), out_tensors_.front(),
                          context_);
  if (conv_kernel_ == nullptr) {
    // need to select actual execute kernel here
    conv_kernel_ = CpuConvFp32KernelSelect(in_tensors_, out_tensors_, op_parameter_, context_, primitive_,
                                           origin_weight_, origin_bias_);
    if (conv_kernel_ == nullptr) {
      MS_LOG(ERROR) << "Selecting execute kernel failed for conv_kernel, got a nullptr.";
      return RET_ERROR;
    }
  }
  FreeCopiedData();
  return conv_kernel_->ReSize();
}

void SetInputOutputShapeInfo(ConvParameter *conv_param, const lite::Tensor *input, const lite::Tensor *output,
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

ConvParameter *CreateNewConvParameter(ConvParameter *parameter) {
  auto conv_parameter = new (std::nothrow) ConvParameter;
  if (conv_parameter == nullptr) {
    MS_LOG(ERROR) << "Malloc new conv parameter failed.";
    return nullptr;
  }
  memcpy(conv_parameter, parameter, sizeof(ConvParameter));
  return conv_parameter;
}

void FreeMemory(const std::vector<kernel::LiteKernel *> &group_convs, const std::vector<lite::Tensor *> &new_inputs,
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

lite::Tensor *CreateInputTensor(TypeId data_type, const std::vector<int> &in_shape, bool infered_flag) {
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

// weight and bias are const
static lite::Tensor *CreateConstTensorFp32(lite::Tensor *tensor, const std::vector<int> &shape, const int index) {
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
  MS_ASSERT(tensor->data_type() == kNumberTypeFloat32);
  memcpy(new_tensor->data_c(), reinterpret_cast<char *>(tensor->data_c()) + index * new_tensor->Size(),
         new_tensor->Size());
  return new_tensor;
}

lite::Tensor *CreateOutputTensor(const std::vector<int> &out_shape, const std::vector<lite::Tensor *> &outputs,
                                 bool infered_flag, int index) {
  auto out_tensor = new (std::nothrow) lite::Tensor();
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "new tmp_out_tensor failed.";
    return nullptr;
  }
  out_tensor->set_data_type(outputs.at(index)->data_type());
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

kernel::LiteKernel *CpuConvFp32KernelSelect(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                            const InnerContext *ctx, const mindspore::lite::PrimitiveC *primitive,
                                            float *origin_weight, float *origin_bias) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  bool use_winograd = false;
  int out_unit;
  CheckIfUseWinograd(&use_winograd, &out_unit, conv_param);
  kernel::LiteKernel *kernel = nullptr;
  if (conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1) {
    kernel = new (std::nothrow)
      kernel::Convolution1x1CPUKernel(op_parameter, inputs, outputs, ctx, primitive, origin_weight, origin_bias);
  } else if (use_winograd) {
    kernel = new (std::nothrow) kernel::ConvolutionWinogradCPUKernel(op_parameter, inputs, outputs, ctx, primitive,
                                                                     out_unit, origin_weight, origin_bias);
  } else {
    kernel = new (std::nothrow)
      kernel::ConvolutionCPUKernel(op_parameter, inputs, outputs, ctx, primitive, origin_weight, origin_bias);
  }
  if (kernel != nullptr) {
    auto ret = kernel->Init();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "conv kernel init failed.";
      delete kernel;
      return nullptr;
    }
  }
  return kernel;
}

static kernel::LiteKernel *CreateDelegateConv(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                              const InnerContext *ctx, const mindspore::lite::PrimitiveC *primitive) {
  return new (std::nothrow) kernel::ConvolutionDelegateCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
}

kernel::LiteKernel *CpuGroupConvFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                                  const InnerContext *ctx,
                                                  const mindspore::lite::PrimitiveC *primitive) {
  bool infer_flag = primitive != nullptr && primitive->infer_flag();
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  int new_in_channel = inputs.at(kWeightIndex)->Channel();
  int new_out_channel;
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

  // create sub kernels
  std::vector<kernel::LiteKernel *> group_convs;
  for (int i = 0; i < conv_param->group_; ++i) {
    std::vector<lite::Tensor *> new_inputs;
    std::vector<lite::Tensor *> new_outputs;
    auto new_conv_parameter = CreateNewConvParameter(conv_param);
    if (new_conv_parameter == nullptr) {
      FreeMemory(group_convs, new_inputs, new_outputs);
      MS_LOG(ERROR) << "Get new conv parameter failed.";
      return nullptr;
    }

    // create new input for each group
    auto in_tensor = CreateInputTensor(inputs.front()->data_type(), in_shape, infer_flag);
    if (in_tensor == nullptr) {
      delete new_conv_parameter;
      FreeMemory(group_convs, new_inputs, new_outputs);
      MS_LOG(ERROR) << "create input tensor failed.";
      return nullptr;
    }
    new_inputs.emplace_back(in_tensor);

    // create new weight
    auto filter_tensor = CreateConstTensorFp32(inputs.at(kWeightIndex), filter_shape, i);
    if (filter_tensor == nullptr) {
      delete new_conv_parameter;
      FreeMemory(group_convs, new_inputs, new_outputs);
      MS_LOG(ERROR) << "create filter tensor failed.";
      return nullptr;
    }
    new_inputs.emplace_back(filter_tensor);

    // if has bias, create new bias
    if (inputs.size() == 3) {
      auto bias_tensor = CreateConstTensorFp32(inputs.at(kBiasIndex), bias_shape, i);
      if (bias_tensor == nullptr) {
        delete new_conv_parameter;
        FreeMemory(group_convs, new_inputs, new_outputs);
        MS_LOG(ERROR) << "create bias_tensor failed.";
        return nullptr;
      }
      new_inputs.emplace_back(bias_tensor);
    }

    // create new output tensor
    for (size_t j = 0; j < outputs.size(); ++j) {
      auto out_tensor = CreateOutputTensor(out_shape, outputs, infer_flag, j);
      if (out_tensor == nullptr) {
        delete new_conv_parameter;
        FreeMemory(group_convs, new_inputs, new_outputs);
        MS_LOG(ERROR) << "new out_tensor failed.";
        return nullptr;
      }
      new_outputs.emplace_back(out_tensor);
    }
    group_convs.emplace_back(
      CreateDelegateConv(new_inputs, new_outputs, reinterpret_cast<OpParameter *>(new_conv_parameter), ctx, primitive));
  }
  return new (std::nothrow)
    GroupConvolutionCPUKernel(op_parameter, inputs, outputs, ctx, primitive, group_convs, conv_param->group_);
}

kernel::LiteKernel *CpuConvFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                             const InnerContext *ctx, const kernel::KernelKey &desc,
                                             const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(op_parameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2D);
  MS_ASSERT(desc.data_type == kNumberTypeFloat32);

  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  kernel::LiteKernel *kernel = nullptr;
  if (conv_param->group_ == 1) {
    kernel = CreateDelegateConv(inputs, outputs, op_parameter, ctx, primitive);
  } else {
    kernel = CpuGroupConvFp32KernelCreator(inputs, outputs, op_parameter, ctx, primitive);
  }

  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    free(op_parameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK && ret != RET_INFER_INVALID) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Conv2D, CpuConvFp32KernelCreator)
}  // namespace mindspore::kernel
