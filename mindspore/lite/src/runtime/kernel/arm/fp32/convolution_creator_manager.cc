/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <vector>
#include "src/runtime/kernel/arm/fp32/convolution_creator_manager.h"
#include "src/runtime/kernel/arm/fp32/convolution_delegate_fp32.h"
#include "src/runtime/kernel/arm/fp32/group_convolution_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_depthwise_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_depthwise_3x3_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_depthwise_slidewindow_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_depthwise_indirect_fp32.h"
#include "src/runtime/kernel/arm/int8/convolution_int8.h"
#include "src/runtime/kernel/arm/int8/convolution_1x1_int8.h"
#include "src/runtime/kernel/arm/int8/convolution_3x3_int8.h"
#include "nnacl/conv_parameter.h"

namespace mindspore::lite {
using mindspore::lite::Format::Format_NHWC;

static inline lite::Tensor *TensorMalloc(lite::Tensor *tensor) {
  if (tensor->MallocData() != RET_OK) {
    delete tensor;
    MS_LOG(ERROR) << "malloc tensor data failed.";
    return nullptr;
  }
  return tensor;
}

lite::Tensor *CreateConstTensor(lite::Tensor *tensor, const std::vector<int> &shape, const int index) {
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

lite::Tensor *CreateVarTensor(const TensorInfo &tensor_info, bool inferred) {
  auto tensor = new (std::nothrow) lite::Tensor();
  if (!tensor) {
    MS_LOG(ERROR) << "new tensor failed.";
    return nullptr;
  }
  tensor->set_data_type(tensor_info.data_type_);
  tensor->set_format(tensor_info.format_);
  tensor->set_category(tensor_info.tensor_type_);
  if (tensor_info.is_in_) {
    tensor->set_shape(tensor_info.shape_);
  }

  if (inferred) {
    // set shape of out tensor
    if (!tensor_info.is_in_) {
      tensor->set_shape(tensor_info.shape_);
    }
    return TensorMalloc(tensor);
  }
  return tensor;
}

/* Kernel creator func part */
kernel::LiteKernel *CpuConvInt8KernelSelect(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                            const InnerContext *ctx) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  kernel::LiteKernel *kernel = nullptr;
  if (conv_param->kernel_h_ == 3 && conv_param->kernel_w_ == 3 && conv_param->stride_h_ == 1 &&
      conv_param->stride_w_ == 1 && conv_param->dilation_h_ == 1 && conv_param->dilation_w_ == 1) {
#ifdef ENABLE_ARM64
    if (mindspore::lite::IsSupportSDot()) {
      kernel = new (std::nothrow) kernel::ConvolutionInt8CPUKernel(op_parameter, inputs, outputs, ctx);
    } else {
      kernel = new (std::nothrow) kernel::Convolution3x3Int8CPUKernel(op_parameter, inputs, outputs, ctx);
    }
#else
    kernel = new (std::nothrow) kernel::Convolution3x3Int8CPUKernel(op_parameter, inputs, outputs, ctx);
#endif
  } else if (conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1) {
    kernel = new (std::nothrow) kernel::Convolution1x1Int8CPUKernel(op_parameter, inputs, outputs, ctx);
  } else {
    kernel = new (std::nothrow) kernel::ConvolutionInt8CPUKernel(op_parameter, inputs, outputs, ctx);
  }
  return kernel;
}

kernel::LiteKernel *DispatchConvDw(const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                   const InnerContext *ctx) {
  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  kernel::LiteKernel *kernel = nullptr;
  if (opParameter != nullptr && opParameter->infer_flag_) {
#if defined(ENABLE_ARM) || (defined(ENABLE_SSE) && !defined(ENABLE_AVX))
    if (CheckConvDw1DWinograd(conv_param, ctx->thread_num_)) {
      kernel = new (std::nothrow) kernel::ConvolutionDepthwise3x3CPUKernel(opParameter, inputs, outputs, ctx);
    }
#endif
#if defined(ENABLE_ARM64) || defined(ENABLE_AVX)
    if (kernel == nullptr && CheckConvDwUseIndirectBuffer(conv_param)) {
      kernel = new (std::nothrow) kernel::ConvolutionDepthwiseIndirectCPUKernel(opParameter, inputs, outputs, ctx);
    }
#endif
    if (kernel == nullptr && conv_param->input_channel_ < 32) {
      kernel = new (std::nothrow) kernel::ConvolutionDepthwiseSWCPUKernel(opParameter, inputs, outputs, ctx);
    }
  }
  if (kernel == nullptr) {
    kernel = new (std::nothrow) kernel::ConvolutionDepthwiseCPUKernel(opParameter, inputs, outputs, ctx);
  }
  return kernel;
}

kernel::LiteKernel *DispatchGroupConv(const std::vector<lite::Tensor *> &inputs,
                                      const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                      const InnerContext *ctx) {
  GroupConvCreator group_conv_creator(inputs, outputs, op_parameter, ctx, false);
  group_conv_creator.SetShapeOfTensors();
  if (group_conv_creator.CreatGroupConv() != RET_OK) {
    MS_LOG(ERROR) << "Create group conv failed.";
    return nullptr;
  }
  return new (std::nothrow)
    kernel::GroupConvolutionCPUKernel(op_parameter, inputs, outputs, ctx, group_conv_creator.get_group_conv(),
                                      reinterpret_cast<ConvParameter *>(op_parameter)->group_);
}

/* Class GroupConv Creator Implement Part*/
void GroupConvCreator::SetShapeOfTensors() {
  int new_in_channel = origin_inputs_.at(kWeightIndex)->Channel();
  int new_out_channel;
  if (conv_param_->group_ == 0) {
    MS_LOG(ERROR) << "Divisor 'group' cannot be 0.";
    return;
  } else {
    new_out_channel = origin_inputs_.at(kWeightIndex)->Batch() / conv_param_->group_;
  }

  /* set shape */
  set_filter_shape({new_out_channel, conv_param_->kernel_h_, conv_param_->kernel_w_, new_in_channel});
  set_bias_shape({new_out_channel});
  if (infered_) {
    conv_param_->input_channel_ = new_in_channel;
    conv_param_->output_channel_ = new_out_channel;
    set_input_shape({origin_inputs_.front()->Batch(), origin_inputs_.front()->Height(), origin_inputs_.front()->Width(),
                     new_in_channel});
    set_output_shape({origin_inputs_.front()->Batch(), origin_outputs_.front()->Height(),
                      origin_outputs_.front()->Width(), new_out_channel});
  }
}

int GroupConvCreator::CreatGroupConv() {
  for (int i = 0; i < conv_param_->group_; ++i) {
    auto new_conv_parameter = CreateNewConvParameter(conv_param_);
    if (!CheckIfValidPoint(new_conv_parameter)) {
      return RET_ERROR;
    }
    // create new input for each group
    std::vector<lite::Tensor *> new_inputs;
    if (NewInputTensor(&new_inputs) != RET_OK) {
      MS_LOG(ERROR) << "new input tensor failed.";
      FreeMemory(new_conv_parameter, new_inputs, {});
      return RET_ERROR;
    }
    // const tensor
    if (NewConstTensor(&new_inputs, i) != RET_OK) {
      MS_LOG(ERROR) << "new const tensor failed.";
      FreeMemory(new_conv_parameter, new_inputs, {});
      return RET_ERROR;
    }
    // create new output tensor
    std::vector<lite::Tensor *> new_outputs;
    for (auto &output : origin_outputs_) {
      if (NewOutputTensor(&new_outputs, output) != RET_OK) {
        MS_LOG(ERROR) << "new output tensor failed.";
        FreeMemory(new_conv_parameter, new_inputs, new_outputs);
        return RET_ERROR;
      }
    }

    if (is_quant_) {
      CopyQuantParam(&new_inputs);
      group_convs_.emplace_back(CpuConvInt8KernelSelect(new_inputs, new_outputs,
                                                        reinterpret_cast<OpParameter *>(new_conv_parameter), context_));
    } else {
      group_convs_.emplace_back(new (std::nothrow) kernel::ConvolutionDelegateCPUKernel(
        reinterpret_cast<OpParameter *>(new_conv_parameter), new_inputs, new_outputs, context_));
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
