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

#include "src/runtime/kernel/arm/int8/convolution_int8.h"
#include "include/errorcode.h"
#include "nnacl/int8/conv_int8.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32/convolution_delegate_fp32.h"
#include "src/runtime/kernel/arm/int8/convolution_1x1_int8.h"
#include "src/runtime/kernel/arm/int8/convolution_3x3_int8.h"
#include "src/runtime/kernel/arm/int8/group_convolution_int8.h"
#include "src/runtime/runtime_api.h"
#ifdef ENABLE_ARM64
#include "src/runtime/kernel/arm/int8/opt_op_handler.h"
#endif

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;
using mindspore::schema::Format::Format_NHWC;

namespace mindspore::kernel {
void ConvolutionInt8CPUKernel::CheckSupportOptimize() {
  tile_num_ = 8;
#ifdef ENABLE_ARM32
  tile_num_ = 4;
  support_optimize_ = false;
#endif

#ifdef ENABLE_ARM64
  if (mindspore::lite::IsSupportSDot()) {
    matmul_func_ = MatMulRInt8_optimize_handler;
    support_optimize_ = true;
  } else {
    tile_num_ = 4;
    support_optimize_ = false;
  }
#endif
  conv_param_->tile_num_ = tile_num_;
}

int ConvolutionInt8CPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  auto output_channel = filter_tensor->Batch();
  int kernel_plane = filter_tensor->Height() * filter_tensor->Width();
  conv_param_->input_channel_ = input_channel;
  conv_param_->output_channel_ = output_channel;
  int up_round_deep;
  int up_round_oc;
#ifdef ENABLE_ARM32
  up_round_oc = UP_ROUND(output_channel, C2NUM);
  up_round_deep = UP_ROUND(kernel_plane * input_channel, C16NUM);
#else
  if (support_optimize_) {
    up_round_oc = UP_ROUND(output_channel, C8NUM);
    up_round_deep = UP_ROUND(kernel_plane * input_channel, C4NUM);
  } else {
    up_round_oc = UP_ROUND(output_channel, C4NUM);
    up_round_deep = UP_ROUND(kernel_plane * input_channel, C16NUM);
  }
#endif
  int pack_weight_size = up_round_oc * up_round_deep;
  size_t bias_size = up_round_oc * sizeof(int32_t);
  int32_t input_zp = conv_param_->conv_quant_arg_.input_quant_args_[0].zp_;

  // init weight
  auto origin_weight = reinterpret_cast<int8_t *>(in_tensors_.at(kWeightIndex)->data_c());
  packed_weight_ = reinterpret_cast<int8_t *>(malloc(pack_weight_size));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed_weight_ failed.";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, pack_weight_size);
#ifdef ENABLE_ARM32
  RowMajor2Row2x16MajorInt8(origin_weight, packed_weight_, output_channel, input_channel * kernel_plane);
#else
  if (support_optimize_) {
    RowMajor2Row8x4MajorInt8(origin_weight, packed_weight_, output_channel, input_channel * kernel_plane);
  } else {
    RowMajor2Row16x4MajorInt8(origin_weight, packed_weight_, output_channel, input_channel * kernel_plane);
  }
#endif

  // init bias
  bias_data_ = reinterpret_cast<int32_t *>(malloc(bias_size));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias_data_ failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, bias_size);
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<int32_t *>(in_tensors_.at(kBiasIndex)->data_c());
    memcpy(bias_data_, ori_bias, output_channel * sizeof(int32_t));
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  auto *bias_data = reinterpret_cast<int32_t *>(bias_data_);
  bool filter_peroc = conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL;
  if (filter_peroc) {
    filter_zp_ptr_ = reinterpret_cast<int32_t *>(malloc(output_channel * sizeof(int32_t)));
    if (filter_zp_ptr_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      return RET_ERROR;
    }
  }
  for (int oc = 0; oc < output_channel; oc++) {
    int32_t filter_zp = conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_;
    if (filter_peroc) {
      filter_zp = conv_param_->conv_quant_arg_.filter_quant_args_[oc].zp_;
      filter_zp_ptr_[oc] = filter_zp;
    }
    int32_t weight_sum_value = up_round_deep * filter_zp;
    for (int i = 0; i < kernel_plane * input_channel; i++) {
      weight_sum_value += origin_weight[oc * kernel_plane * input_channel + i] - filter_zp;
    }
    bias_data[oc] += filter_zp * input_zp * up_round_deep - weight_sum_value * input_zp;
  }

  size_t input_sum_size;
  if (conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL) {
    input_sum_size = up_round_oc * tile_num_ * thread_count_ * sizeof(int32_t);
  } else {
    input_sum_size = tile_num_ * thread_count_ * sizeof(int32_t);
  }
  input_sum_ = reinterpret_cast<int32_t *>(malloc(input_sum_size));
  if (input_sum_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_sum_ failed.";
    return RET_ERROR;
  }
  memset(input_sum_, 0, input_sum_size);
  return RET_OK;
}

int ConvolutionInt8CPUKernel::InitTmpBuffer() {
  MS_ASSERT(ctx_->allocator != nullptr);
  int kernel_plane = conv_param_->kernel_h_ * conv_param_->kernel_w_;
  int tmp_size;
  if (support_optimize_) {
    tmp_size = UP_ROUND(kernel_plane * conv_param_->input_channel_, C4NUM);
  } else {
    tmp_size = UP_ROUND(kernel_plane * conv_param_->input_channel_, C16NUM);
  }
  matmul_packed_input_ = reinterpret_cast<int8_t *>(
    ctx_->allocator->Malloc(thread_count_ * tile_num_ * kernel_plane * conv_param_->input_channel_));
  if (matmul_packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc matmul_packed_input_ failed.";
    return RET_ERROR;
  }
  packed_input_ = reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(tmp_size * thread_count_ * tile_num_));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed_input_ failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionInt8CPUKernel::Init() {
  CheckSupportOptimize();
  auto ret = SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set quant param failed.";
    return ret;
  }

  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Initialization for optimized int8 conv failed.";
    return RET_ERROR;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionInt8CPUKernel::ReSize() {
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

int ConvolutionInt8CPUKernel::RunImpl(int task_id) {
  auto ori_input_data = reinterpret_cast<int8_t *>(in_tensors_.at(kInputIndex)->data_c());
  auto output_addr = reinterpret_cast<int8_t *>(out_tensors_.at(kOutputIndex)->data_c());
  ConvInt8(ori_input_data, packed_input_, matmul_packed_input_, packed_weight_, reinterpret_cast<int32_t *>(bias_data_),
           output_addr, filter_zp_ptr_, input_sum_, task_id, conv_param_, matmul_func_, support_optimize_);
  return RET_OK;
}

int ConvolutionInt8Impl(void *cdata, int task_id) {
  auto conv = reinterpret_cast<ConvolutionInt8CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution Int8 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionInt8CPUKernel::Run() {
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }

  int error_code = ParallelLaunch(this->context_->thread_pool_, ConvolutionInt8Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv int8 error error_code[" << error_code << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  FreeTmpBuffer();
  return RET_OK;
}

lite::Tensor *CreateFilterTensorInt8(TypeId data_type, std::vector<int> filter_shape,
                                     const std::vector<lite::Tensor *> &inputs, int copy_length, int index) {
  MS_ASSERT(data_type == kNumberTypeInt8);
  auto filter_tensor =
    new (std::nothrow) lite::Tensor(data_type, filter_shape, Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  if (filter_tensor == nullptr) {
    MS_LOG(ERROR) << "new filter_tensor failed.";
    return nullptr;
  }
  auto ret = filter_tensor->MallocData();
  if (ret != RET_OK) {
    delete filter_tensor;
    MS_LOG(ERROR) << "filter_tensor malloc failed.";
    return nullptr;
  }
  auto *origin_weight = reinterpret_cast<int8_t *>(inputs.at(kWeightIndex)->data_c());
  memcpy(filter_tensor->data_c(), origin_weight + index * copy_length, copy_length * sizeof(int8_t));
  return filter_tensor;
}

lite::Tensor *CreateBiasTensorInt8(TypeId data_type, std::vector<int> bias_shape,
                                   const std::vector<lite::Tensor *> &inputs, int new_out_channel, int index) {
  MS_ASSERT(data_type == kNumberTypeInt32);
  auto *origin_bias = inputs.at(kBiasIndex)->data_c();
  auto bias_tensor =
    new (std::nothrow) lite::Tensor(data_type, bias_shape, Format_NHWC, lite::Tensor::Category::CONST_TENSOR);
  if (bias_tensor == nullptr) {
    MS_LOG(ERROR) << "new bias_tensor failed.";
    return nullptr;
  }
  auto ret = bias_tensor->MallocData();
  if (ret != RET_OK) {
    delete bias_tensor;
    MS_LOG(ERROR) << "bias_tensor malloc failed.";
    return nullptr;
  }
  auto bias_data = reinterpret_cast<int32_t *>(origin_bias);
  memcpy(bias_tensor->data_c(), bias_data + index * new_out_channel, new_out_channel * sizeof(int32_t));
  return bias_tensor;
}

kernel::LiteKernel *CpuConvInt8KernelSelect(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                            const InnerContext *ctx, const mindspore::lite::PrimitiveC *primitive) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  kernel::LiteKernel *kernel = nullptr;
  if (conv_param->kernel_h_ == 3 && conv_param->kernel_w_ == 3 && conv_param->stride_h_ == 1 &&
      conv_param->stride_w_ == 1 && conv_param->dilation_h_ == 1 && conv_param->dilation_w_ == 1) {
#ifdef ENABLE_ARM64
    if (mindspore::lite::IsSupportSDot()) {
      kernel = new (std::nothrow) kernel::ConvolutionInt8CPUKernel(op_parameter, inputs, outputs, ctx, primitive);
    } else {
      kernel = new (std::nothrow) kernel::Convolution3x3Int8CPUKernel(op_parameter, inputs, outputs, ctx, primitive);
    }
#else
    kernel = new (std::nothrow) kernel::Convolution3x3Int8CPUKernel(op_parameter, inputs, outputs, ctx, primitive);
#endif
  } else if (conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1) {
    kernel = new (std::nothrow) kernel::Convolution1x1Int8CPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  } else {
    kernel = new (std::nothrow) kernel::ConvolutionInt8CPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  }
  return kernel;
}

void CopyTensorQuantParam(lite::Tensor *dst, lite::Tensor *src) {
  for (size_t i = 0; i < src->quant_params().size(); i++) {
    dst->AddQuantParam(src->quant_params().at(i));
  }
}

kernel::LiteKernel *CpuGroupConvInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                                  const InnerContext *ctx, const mindspore::lite::PrimitiveC *primitive,
                                                  int group) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  std::vector<int> in_shape;
  std::vector<int> out_shape;
  int new_in_channel = inputs.at(kWeightIndex)->Channel();
  int new_out_channel = 0;
  if (group == 0) {
    MS_LOG(ERROR) << "Divisor 'group' cannot be 0.";
    return nullptr;
  } else {
    new_out_channel = inputs.at(kWeightIndex)->Batch() / group;
  }
  int batch = inputs.front()->Batch();
  conv_param->input_batch_ = batch;
  conv_param->output_batch_ = batch;
  bool infered_flag = primitive != nullptr && primitive->infer_flag();
  if (infered_flag) {
    int in_h = inputs.front()->Height();
    int in_w = inputs.front()->Width();
    conv_param->input_channel_ = new_in_channel;
    conv_param->output_channel_ = new_out_channel;
    in_shape = {batch, in_h, in_w, new_in_channel};
    out_shape = {batch, conv_param->output_h_, conv_param->output_w_, new_out_channel};
  }
  std::vector<int> filter_shape = {new_out_channel, conv_param->kernel_h_, conv_param->kernel_w_, new_in_channel};
  std::vector<int> bias_shape = {new_out_channel};

  // create sub kernels
  std::vector<kernel::LiteKernel *> group_convs;
  for (int i = 0; i < group; ++i) {
    std::vector<lite::Tensor *> new_inputs;
    std::vector<lite::Tensor *> new_outputs;
    auto new_conv_parameter = CreateNewConvParameter(conv_param);
    if (new_conv_parameter == nullptr) {
      FreeMemory(group_convs, new_inputs, new_outputs);
      MS_LOG(ERROR) << "Get new conv parameter failed.";
      return nullptr;
    }

    // create new input for each group
    auto input_data_type = inputs.front()->data_type();
    MS_ASSERT(input_data_type == kNumberTypeInt8);
    auto in_tensor = CreateInputTensor(input_data_type, in_shape, infered_flag);
    if (in_tensor == nullptr) {
      delete new_conv_parameter;
      FreeMemory(group_convs, new_inputs, new_outputs);
      MS_LOG(ERROR) << "create input tensor failed.";
      return nullptr;
    }
    CopyTensorQuantParam(in_tensor, inputs[kInputIndex]);
    new_inputs.emplace_back(in_tensor);

    // create new weight
    int copy_length = conv_param->kernel_h_ * conv_param->kernel_w_ * new_in_channel * new_out_channel;
    auto filter_tensor =
      CreateFilterTensorInt8(inputs.at(kWeightIndex)->data_type(), filter_shape, inputs, copy_length, i);
    if (filter_tensor == nullptr) {
      delete new_conv_parameter;
      FreeMemory(group_convs, new_inputs, new_outputs);
      MS_LOG(ERROR) << "create filter tensor failed.";
      return nullptr;
    }
    CopyTensorQuantParam(filter_tensor, inputs[kWeightIndex]);
    new_inputs.emplace_back(filter_tensor);

    // if has bias, create new bias
    if (inputs.size() == 3) {
      auto bias_tensor =
        CreateBiasTensorInt8(inputs.at(kBiasIndex)->data_type(), bias_shape, inputs, new_out_channel, i);
      if (bias_tensor == nullptr) {
        delete new_conv_parameter;
        FreeMemory(group_convs, new_inputs, new_outputs);
        MS_LOG(ERROR) << "create bias_tensor failed.";
        return nullptr;
      }
      CopyTensorQuantParam(bias_tensor, inputs[kBiasIndex]);
      new_inputs.emplace_back(bias_tensor);
    }

    // create new output tensor
    for (size_t j = 0; j < outputs.size(); ++j) {
      auto out_tensor = CreateOutputTensor(out_shape, outputs, infered_flag, j);
      if (out_tensor == nullptr) {
        delete new_conv_parameter;
        FreeMemory(group_convs, new_inputs, new_outputs);
        MS_LOG(ERROR) << "new out_tensor failed.";
        return nullptr;
      }
      CopyTensorQuantParam(out_tensor, outputs[j]);
      new_outputs.emplace_back(out_tensor);
    }
    group_convs.emplace_back(CpuConvInt8KernelSelect(
      new_inputs, new_outputs, reinterpret_cast<OpParameter *>(new_conv_parameter), ctx, primitive));
  }
  return new (std::nothrow)
    GroupConvolutionInt8CPUKernel(op_parameter, inputs, outputs, ctx, primitive, group_convs, group);
}

kernel::LiteKernel *CpuConvInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                             const InnerContext *ctx, const kernel::KernelKey &desc,
                                             const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2D);
  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  kernel::LiteKernel *kernel = nullptr;
  if (primitive != nullptr && primitive->infer_flag()) {
    conv_param->input_h_ = inputs.front()->Height();
    conv_param->input_w_ = inputs.front()->Width();
    conv_param->input_channel_ = inputs.front()->Channel();
    conv_param->output_h_ = outputs.front()->Height();
    conv_param->output_w_ = outputs.front()->Width();
    conv_param->output_channel_ = outputs.front()->Channel();
    conv_param->op_parameter_.thread_num_ = ctx->thread_num_;
  }
  if (conv_param->group_ == 1) {
    kernel = CpuConvInt8KernelSelect(inputs, outputs, opParameter, ctx, primitive);
  } else {
    MS_ASSERT(conv_param->group_ > 1);
    kernel = CpuGroupConvInt8KernelCreator(inputs, outputs, opParameter, ctx, primitive, conv_param->group_);
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Conv2D, CpuConvInt8KernelCreator)
}  // namespace mindspore::kernel
