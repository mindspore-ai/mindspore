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

#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "src/runtime/kernel/arm/fp32/convolution.h"
#include "src/runtime/kernel/arm/fp32/convolution_winograd.h"
#include "src/runtime/kernel/arm/fp32/deconvolution.h"
#include "src/runtime/kernel/arm/fp32/convolution_1x1.h"
#include "src/runtime/kernel/arm/fp32/convolution_3x3.h"
#include "src/runtime/kernel/arm/fp32/convolution_depthwise.h"
#include "src/runtime/kernel/arm/fp32/deconvolution_depthwise.h"
#ifdef ENABLE_FP16
#include "src/runtime/kernel/arm/fp16/convolution_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_3x3_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_depthwise_fp16.h"
#include "src/runtime/kernel/arm/fp16/deconvolution_depthwise_fp16.h"
#endif
#include "src/runtime/kernel/arm/int8/deconvolution_int8.h"
#include "src/runtime/kernel/arm/int8/convolution_int8.h"
#include "src/runtime/kernel/arm/int8/convolution_3x3_int8.h"
#include "src/runtime/kernel/arm/int8/convolution_depthwise_int8.h"
#include "src/runtime/kernel/arm/int8/deconvolution_depthwise_int8.h"
#include "schema/model_generated.h"
#include "src/kernel_factory.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType;
using mindspore::schema::PadMode;
using mindspore::schema::PrimitiveType_Conv2D;
using mindspore::schema::PrimitiveType_DeConv2D;
using mindspore::schema::PrimitiveType_DeDepthwiseConv2D;
using mindspore::schema::PrimitiveType_DepthwiseConv2D;

namespace mindspore::kernel {
ConvolutionBaseCPUKernel::~ConvolutionBaseCPUKernel() {
  if (bias_data_ != nullptr) {
    free(bias_data_);
    bias_data_ = nullptr;
  }
  if (nhwc4_input_ != nullptr) {
    free(nhwc4_input_);
    nhwc4_input_ = nullptr;
  }
}

void ConvolutionBaseCPUKernel::FreeQuantParam() {
  ConvQuantArg *conv_quant_arg_ = &conv_param_->conv_quant_arg_;
  if (conv_quant_arg_ == nullptr) {
    return;
  }
  if (conv_quant_arg_->real_multiplier_ != nullptr) {
    free(conv_quant_arg_->real_multiplier_);
    conv_quant_arg_->real_multiplier_ = nullptr;
  }
  if (conv_quant_arg_->left_shift_ != nullptr) {
    free(conv_quant_arg_->left_shift_);
    conv_quant_arg_->left_shift_ = nullptr;
  }
  if (conv_quant_arg_->right_shift_ != nullptr) {
    free(conv_quant_arg_->right_shift_);
    conv_quant_arg_->right_shift_ = nullptr;
  }
  if (conv_quant_arg_->quant_multiplier_ != nullptr) {
    free(conv_quant_arg_->quant_multiplier_);
    conv_quant_arg_->quant_multiplier_ = nullptr;
  }
  if (conv_quant_arg_->out_act_min_ != nullptr) {
    free(conv_quant_arg_->out_act_min_);
    conv_quant_arg_->out_act_min_ = nullptr;
  }
  if (conv_quant_arg_->out_act_max_ != nullptr) {
    free(conv_quant_arg_->out_act_max_);
    conv_quant_arg_->out_act_max_ = nullptr;
  }

  if (conv_quant_arg_->quant_args_ != nullptr) {
    for (int i = 0; i < 3; ++i) {
      if (*(conv_quant_arg_->quant_args_ + i) != nullptr) {
        free(*(conv_quant_arg_->quant_args_ + i));
      }
    }
  }
}

int ConvolutionBaseCPUKernel::Init() {
  auto input = this->inputs_.front();
  auto output = this->outputs_.front();
  conv_param_->input_batch_ = input->Batch();
  conv_param_->input_h_ = input->Height();
  conv_param_->input_w_ = input->Width();
  conv_param_->input_channel_ = input->Channel();
  conv_param_->output_batch_ = output->Batch();
  conv_param_->output_h_ = output->Height();
  conv_param_->output_w_ = output->Width();
  conv_param_->output_channel_ = output->Channel();
  conv_param_->thread_num_ = ctx_->threadNum;
  return RET_OK;
}

int ConvolutionBaseCPUKernel::CheckLayout(lite::tensor::Tensor *input_tensor) {
  auto data_type = input_tensor->data_type();
  auto input_format = input_tensor->GetFormat();
  schema::Format execute_format = schema::Format_NHWC4;
  convert_func_ = LayoutTransform(data_type, input_format, execute_format);
  if (convert_func_ == nullptr) {
    MS_LOG(ERROR) << "layout convert func is nullptr.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionBaseCPUKernel::SetQuantParam() {
  ConvQuantArg *conv_quant_arg_ = &conv_param_->conv_quant_arg_;
  conv_quant_arg_->quant_args_ = reinterpret_cast<QuantArg **>(malloc(3 * sizeof(QuantArg *)));
  if (conv_quant_arg_->quant_args_ == nullptr) {
    MS_LOG(ERROR) << "malloc quant_args_ failed.";
    return RET_ERROR;
  }
  // per-tensor init
  for (int j = 0; j < 3; ++j) {
    conv_quant_arg_->quant_args_[j] = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
    if (conv_quant_arg_->quant_args_[j] == nullptr) {
      MS_LOG(ERROR) << "malloc quant_args_ failed.";
      return RET_ERROR;
    }
  }
  auto input_tensor = inputs_.at(kInputIndex);
  auto weight_tensor = inputs_.at(kWeightIndex);
  auto output_tensor = outputs_.at(kOutputIndex);
  auto input_quant_arg = input_tensor->GetQuantParams().front();
  auto weight_quant_arg = weight_tensor->GetQuantParams().front();
  auto output_quant_arg = output_tensor->GetQuantParams().front();
  // input
  conv_quant_arg_->quant_args_[0][0].zp_ = input_quant_arg.zeroPoint;
  conv_quant_arg_->quant_args_[0][0].scale_ = input_quant_arg.scale;
  // weight
  conv_quant_arg_->quant_args_[1][0].zp_ = weight_quant_arg.zeroPoint;
  conv_quant_arg_->quant_args_[1][0].scale_ = weight_quant_arg.scale;
  // output
  conv_quant_arg_->quant_args_[2][0].zp_ = output_quant_arg.zeroPoint;
  conv_quant_arg_->quant_args_[2][0].scale_ = output_quant_arg.scale;

  conv_quant_arg_->real_multiplier_ = reinterpret_cast<double *>(malloc(sizeof(double)));
  conv_quant_arg_->left_shift_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  conv_quant_arg_->right_shift_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  conv_quant_arg_->quant_multiplier_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  conv_quant_arg_->out_act_min_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  conv_quant_arg_->out_act_max_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));

  double real_multiplier = weight_quant_arg.scale * input_quant_arg.scale / output_quant_arg.scale;
  conv_quant_arg_->real_multiplier_[0] = real_multiplier;
  QuantizeRoundParameter(real_multiplier, &conv_quant_arg_->quant_multiplier_[0], &conv_quant_arg_->left_shift_[0],
                         &conv_quant_arg_->right_shift_[0]);

  ComputeQuantOutRange(conv_param_);
  return RET_OK;
}

void ComputeQuantOutRange(ConvParameter *conv_param) {
  int32_t min = std::numeric_limits<int8_t>::min();
  int32_t max = std::numeric_limits<int8_t>::max();
  float scale = conv_param->conv_quant_arg_.quant_args_[2][0].scale_;
  int32_t zp = conv_param->conv_quant_arg_.quant_args_[2][0].zp_;
  bool is_relu = conv_param->is_relu_;
  bool is_relu6 = conv_param->is_relu6_;
  int32_t quantized_zero = QuantizeToInt8(0, scale, zp);
  int32_t quantized_six = QuantizeToInt8(6, scale, zp);
  if (is_relu) {
    min = min > quantized_zero ? min : quantized_zero;
  } else if (is_relu6) {
    min = min > quantized_zero ? min : quantized_zero;
    max = max < quantized_six ? max : quantized_six;
  } else {
    // do nothing
  }
  conv_param->conv_quant_arg_.out_act_min_[0] = min;
  conv_param->conv_quant_arg_.out_act_max_[0] = max;
}

void CheckIfUseWinograd(bool *use_winograd, int *output_unit, ConvParameter *conv_param,
                        InputTransformUnitFunc input_trans_func, OutputTransformUnitFunc output_trans_func) {
  if (conv_param->kernel_w_ == conv_param->kernel_h_ && conv_param->dilation_h_ == 1 && conv_param->dilation_w_ == 1 &&
      conv_param->stride_h_ == 1 && conv_param->stride_w_ == 1) {
    *output_unit = SelectOutputUnit(conv_param);
    if (*output_unit > 1) {
      *use_winograd = true;
      int input_unit = conv_param->kernel_h_ + *output_unit - 1;
      input_trans_func = GetInputTransFunc(input_unit);
      if (input_trans_func == nullptr) {
        MS_LOG(INFO) << "No matching input trans func. Turn back to common conv.";
        *use_winograd = false;
      }
      output_trans_func = GetOutputTransFunc(input_unit, *output_unit);
      if (output_trans_func == nullptr) {
        MS_LOG(INFO) << "No matching output trans func. Turn back to common conv.";
        *use_winograd = false;
      }
    } else {
      *use_winograd = false;
    }
  } else {
    *use_winograd = false;
  }
}

bool CheckSupportFP16() {
  bool support_fp16 = false;
#ifdef ENABLE_ARM64
  void *optimize_op_handler = OptimizeModule::GetInstance()->optimized_op_handler_;
  if (optimize_op_handler != nullptr) {
    support_fp16 = true;
    MS_LOG(INFO) << "Support FP16.";
  } else {
    support_fp16 = false;
    MS_LOG(INFO) << "Your machine doesn't support fp16, return back to float32 kernel.";
  }
#endif
  return support_fp16;
}

kernel::LiteKernel *CpuConvFloatKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const Context *ctx) {
  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int stride_h = conv_param->stride_h_;
  int stride_w = conv_param->stride_w_;
  int dilation_h = conv_param->dilation_h_;
  int dilation_w = conv_param->dilation_w_;
  conv_param->input_h_ = inputs.front()->Height();
  conv_param->input_w_ = inputs.front()->Width();
  conv_param->output_h_ = outputs.front()->Height();
  conv_param->output_w_ = outputs.front()->Width();
  bool use_winograd;
  int out_unit;
  InputTransformUnitFunc input_trans_func = nullptr;
  OutputTransformUnitFunc output_trans_func = nullptr;
  CheckIfUseWinograd(&use_winograd, &out_unit, conv_param, input_trans_func, output_trans_func);
  bool support_fp16 = CheckSupportFP16();

  if (kernel_h == 1 && kernel_w == 1) {
    auto kernel = new (std::nothrow) Convolution1x1CPUKernel(opParameter, inputs, outputs, ctx);
    return kernel;
  } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
    if (support_fp16) {
#ifdef ENABLE_FP16
      auto kernel = new (std::nothrow) Convolution3x3FP16CPUKernel(opParameter, inputs, outputs, ctx);
      return kernel;
#endif
    }
    auto kernel = new (std::nothrow) Convolution3x3CPUKernel(opParameter, inputs, outputs, ctx);
    return kernel;
  } else if (use_winograd) {
    auto kernel = new (std::nothrow) ConvolutionWinogradCPUKernel(opParameter, inputs, outputs, ctx, out_unit);
    return kernel;
  } else {
    if (support_fp16) {
#ifdef ENABLE_FP16
      auto kernel = new (std::nothrow) ConvolutionFP16CPUKernel(opParameter, inputs, outputs, ctx);
      return kernel;
#endif
    }
    auto kernel = new (std::nothrow) ConvolutionCPUKernel(opParameter, inputs, outputs, ctx);
    return kernel;
  }
}

kernel::LiteKernel *CpuConvInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs,
                                             OpParameter *opParameter, const Context *ctx) {
  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int stride_h = conv_param->stride_h_;
  int stride_w = conv_param->stride_w_;
  int dilation_h = conv_param->dilation_h_;
  int dilation_w = conv_param->dilation_w_;

  if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
    auto kernel = new (std::nothrow) Convolution3x3Int8CPUKernel(opParameter, inputs, outputs, ctx);
    return kernel;
  } else {
    auto kernel = new (std::nothrow) ConvolutionInt8CPUKernel(opParameter, inputs, outputs, ctx);
    return kernel;
  }
}

kernel::LiteKernel *CpuConvKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                         const std::vector<lite::tensor::Tensor *> &outputs, OpParameter *opParameter,
                                         const lite::Context *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2D);
  auto input_tensor = inputs.at(kInputIndex);
  auto data_type = input_tensor->data_type();
  kernel::LiteKernel *kernel = nullptr;
  switch (data_type) {
    case kNumberTypeInt8:
      kernel = CpuConvInt8KernelCreator(inputs, outputs, opParameter, ctx);
      break;
    case kNumberTypeFloat32:
      kernel = CpuConvFloatKernelCreator(inputs, outputs, opParameter, ctx);
      break;
    default:
      break;
  }

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

kernel::LiteKernel *CpuConvDwFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const Context *ctx) {
  auto kernel = new (std::nothrow) ConvolutionDepthwiseCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  return kernel;
}

#ifdef ENABLE_FP16
kernel::LiteKernel *CpuConvDwFp16KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const Context *ctx) {
  auto kernel = new (std::nothrow) ConvolutionDepthwiseFp16CPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  return kernel;
}
#endif

kernel::LiteKernel *CpuConvDwInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const Context *ctx) {
  auto kernel = new (std::nothrow) ConvolutionDepthwiseInt8CPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  return kernel;
}

kernel::LiteKernel *CpuConvDwKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                           const std::vector<lite::tensor::Tensor *> &outputs, OpParameter *opParameter,
                                           const lite::Context *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_DepthwiseConv2D);
  auto input_tensor = inputs.at(kInputIndex);
  auto data_type = input_tensor->data_type();
  kernel::LiteKernel *kernel = nullptr;
  switch (data_type) {
    case kNumberTypeInt8:
      kernel = CpuConvDwInt8KernelCreator(inputs, outputs, opParameter, ctx);
      break;
    case kNumberTypeFloat32:
#ifdef ENABLE_FP16
      kernel = CpuConvDwFp16KernelCreator(inputs, outputs, opParameter, ctx);
#else
      kernel = CpuConvDwFp32KernelCreator(inputs, outputs, opParameter, ctx);
#endif
      break;
    default:
      break;
  }
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

kernel::LiteKernel *CpuDeconvDwFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                 const std::vector<lite::tensor::Tensor *> &outputs,
                                                 OpParameter *opParameter, const lite::Context *ctx) {
  auto kernel = new (std::nothrow) DeconvolutionDepthwiseCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  return kernel;
}

#ifdef ENABLE_FP16
kernel::LiteKernel *CpuDeconvDwFp16KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                 const std::vector<lite::tensor::Tensor *> &outputs,
                                                 OpParameter *opParameter, const lite::Context *ctx) {
  auto kernel = new (std::nothrow) DeconvolutionDepthwiseFp16CPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  return kernel;
}
#endif

kernel::LiteKernel *CpuDeconvDwInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                 const std::vector<lite::tensor::Tensor *> &outputs,
                                                 OpParameter *opParameter, const lite::Context *ctx) {
  auto kernel = new (std::nothrow) DeconvolutionDepthwiseInt8CPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  return kernel;
}

kernel::LiteKernel *CpuDeconvDwKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs,
                                             OpParameter *opParameter, const lite::Context *ctx,
                                             const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_DepthwiseConv2D);
  auto input_tensor = inputs.at(kInputIndex);
  auto data_type = input_tensor->data_type();
  kernel::LiteKernel *kernel = nullptr;
  switch (data_type) {
    case kNumberTypeInt8:
      kernel = CpuDeconvDwInt8KernelCreator(inputs, outputs, opParameter, ctx);
      break;
    case kNumberTypeFloat32:
#ifdef ENABLE_FP16
      kernel = CpuDeconvDwFp16KernelCreator(inputs, outputs, opParameter, ctx);
#else
      kernel = CpuDeconvDwFp32KernelCreator(inputs, outputs, opParameter, ctx);
#endif
      break;
    default:
      break;
  }
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

kernel::LiteKernel *CpuDeConvFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx) {
  auto kernel = new (std::nothrow) DeConvolutionCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  return kernel;
}

kernel::LiteKernel *CpuDeConvInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx) {
  auto kernel = new (std::nothrow) DeConvInt8CPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  return kernel;
}

kernel::LiteKernel *CpuDeConvKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                           const std::vector<lite::tensor::Tensor *> &outputs, OpParameter *opParameter,
                                           const lite::Context *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_DeConv2D);
  auto input_tensor = inputs.at(kInputIndex);
  auto data_type = input_tensor->data_type();
  kernel::LiteKernel *kernel = nullptr;
  switch (data_type) {
    case kNumberTypeInt8:
      kernel = CpuDeConvInt8KernelCreator(inputs, outputs, opParameter, ctx);
      break;
#ifdef ENABLE_FP16
    case kNumberTypeFloat16:
      break;
#endif
    case kNumberTypeFloat32:
      kernel = CpuDeConvFp32KernelCreator(inputs, outputs, opParameter, ctx);
      break;
    default:
      break;
  }
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

REG_KERNEL(kCPU, PrimitiveType_Conv2D, CpuConvKernelCreator)
REG_KERNEL(kCPU, PrimitiveType_DeConv2D, CpuDeConvKernelCreator)
REG_KERNEL(kCPU, PrimitiveType_DepthwiseConv2D, CpuConvDwKernelCreator)
REG_KERNEL(kCPU, PrimitiveType_DeDepthwiseConv2D, CpuDeconvDwKernelCreator)
}  // namespace mindspore::kernel
