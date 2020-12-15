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

#include "src/runtime/kernel/arm/fp32/convolution_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_1x1_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_winograd_fp32.h"
#include "src/runtime/kernel/arm/fp32/group_convolution_fp32.h"
#include "nnacl/fp32/conv_fp32.h"
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
using mindspore::schema::Format::Format_NHWC;

namespace mindspore::kernel {
int ConvolutionCPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  int in_channel = filter_tensor->Channel();
  int out_channel = filter_tensor->Batch();
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;
  int kernel_plane = filter_tensor->Height() * filter_tensor->Width();
#ifdef ENABLE_AVX
  const int oc_block = C16NUM;
#elif ENABLE_ARM32
  const int oc_block = C4NUM;
#else
  const int oc_block = C8NUM;
#endif
  int oc_block_num = UP_ROUND(out_channel, oc_block);
  int pack_weight_size = oc_block_num * in_channel * kernel_plane;

  auto origin_weight = reinterpret_cast<float *>(filter_tensor->data_c());
  packed_weight_ = reinterpret_cast<float *>(malloc(pack_weight_size * sizeof(float)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed weight failed.";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, pack_weight_size * sizeof(float));
#ifdef ENABLE_AVX
  RowMajor2Col16Major(origin_weight, packed_weight_, out_channel, in_channel * kernel_plane);
#elif ENABLE_ARM32
  RowMajor2Col4Major(origin_weight, packed_weight_, out_channel, in_channel * kernel_plane);
#else
  RowMajor2Col8Major(origin_weight, packed_weight_, out_channel, in_channel * kernel_plane);
#endif

  bias_data_ = reinterpret_cast<float *>(malloc(oc_block_num * sizeof(float)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, oc_block_num * sizeof(float));

  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->data_c());
    memcpy(bias_data_, ori_bias, out_channel * sizeof(float));
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  return RET_OK;
}

int ConvolutionCPUKernel::InitTmpBuffer() {
  MS_ASSERT(ctx_->allocator != nullptr);

#ifdef ENABLE_AVX
  int unit_size = conv_param_->kernel_h_ * conv_param_->kernel_w_ * conv_param_->input_channel_ * C6NUM * thread_count_;
#elif ENABLE_SSE
  int unit_size = conv_param_->kernel_h_ * conv_param_->kernel_w_ * conv_param_->input_channel_ * C4NUM * thread_count_;
#else
  int unit_size =
    conv_param_->kernel_h_ * conv_param_->kernel_w_ * conv_param_->input_channel_ * C12NUM * thread_count_;
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
  auto ori_input_data = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->data_c());
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->data_c());
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
    FreeTmpBuffer();
    return RET_ERROR;
  }

  ret = ParallelLaunch(this->context_->thread_pool_, ConvolutionImpl, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "conv error error_code[" << ret << "]";
  }
  FreeTmpBuffer();
  return ret;
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

lite::Tensor *CreateInputTensor(TypeId data_type, std::vector<int> in_shape, bool infered_flag) {
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

lite::Tensor *CreateFilterTensorFp32(TypeId data_type, std::vector<int> filter_shape,
                                     const std::vector<lite::Tensor *> &inputs, int copy_length, int index) {
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

  MS_ASSERT(data_type == kNumberTypeFloat32);
  auto *origin_weight = reinterpret_cast<float *>(inputs.at(kWeightIndex)->data_c());
  memcpy(filter_tensor->data_c(), origin_weight + index * copy_length, copy_length * sizeof(float));
  return filter_tensor;
}

lite::Tensor *CreateBiasTensorFp32(TypeId data_type, std::vector<int> bias_shape,
                                   const std::vector<lite::Tensor *> &inputs, int new_out_channel, int index) {
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
  MS_ASSERT(data_type == kNumberTypeFloat32);
  auto bias_data = reinterpret_cast<float *>(origin_bias);
  memcpy(bias_tensor->data_c(), bias_data + index * new_out_channel, new_out_channel * sizeof(float));

  return bias_tensor;
}

lite::Tensor *CreateOutputTensor(std::vector<int> out_shape, const std::vector<lite::Tensor *> &outputs,
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
                                            bool use_winograd, int out_unit) {
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  if (conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1) {
    return new (std::nothrow) kernel::Convolution1x1CPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  } else if (use_winograd) {
    return new (std::nothrow)
      kernel::ConvolutionWinogradCPUKernel(op_parameter, inputs, outputs, ctx, primitive, out_unit);
  } else {
    return new (std::nothrow) kernel::ConvolutionCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  }
  return nullptr;
}

kernel::LiteKernel *CpuGroupConvFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                                  const InnerContext *ctx, const mindspore::lite::PrimitiveC *primitive,
                                                  int group) {
  int out_unit;
  bool has_bias = inputs.size() == 3;
  bool use_winograd = false;
  bool infered_flag = primitive != nullptr && primitive->infer_flag();
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
  if (infered_flag) {
    int in_h = inputs.front()->Height();
    int in_w = inputs.front()->Width();
    conv_param->input_channel_ = new_in_channel;
    conv_param->output_channel_ = new_out_channel;
    CheckIfUseWinograd(&use_winograd, &out_unit, conv_param);
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
    auto in_tensor = CreateInputTensor(inputs.front()->data_type(), in_shape, infered_flag);
    if (in_tensor == nullptr) {
      delete new_conv_parameter;
      FreeMemory(group_convs, new_inputs, new_outputs);
      MS_LOG(ERROR) << "create input tensor failed.";
      return nullptr;
    }
    new_inputs.emplace_back(in_tensor);

    // create new weight
    int copy_length = conv_param->kernel_h_ * conv_param->kernel_w_ * new_in_channel * new_out_channel;
    auto filter_tensor =
      CreateFilterTensorFp32(inputs.at(kWeightIndex)->data_type(), filter_shape, inputs, copy_length, i);
    if (filter_tensor == nullptr) {
      delete new_conv_parameter;
      FreeMemory(group_convs, new_inputs, new_outputs);
      MS_LOG(ERROR) << "create filter tensor failed.";
      return nullptr;
    }
    new_inputs.emplace_back(filter_tensor);

    // if has bias, create new bias
    if (has_bias) {
      auto bias_tensor =
        CreateBiasTensorFp32(inputs.at(kBiasIndex)->data_type(), bias_shape, inputs, new_out_channel, i);
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
      auto out_tensor = CreateOutputTensor(out_shape, outputs, infered_flag, j);
      if (out_tensor == nullptr) {
        delete new_conv_parameter;
        FreeMemory(group_convs, new_inputs, new_outputs);
        MS_LOG(ERROR) << "new out_tensor failed.";
        return nullptr;
      }
      new_outputs.emplace_back(out_tensor);
    }
    group_convs.emplace_back(CpuConvFp32KernelSelect(new_inputs, new_outputs,
                                                     reinterpret_cast<OpParameter *>(new_conv_parameter), ctx,
                                                     primitive, use_winograd, out_unit));
  }

  return new (std::nothrow)
    GroupConvolutionCPUKernel(op_parameter, inputs, outputs, ctx, primitive, group_convs, group);
}

kernel::LiteKernel *CpuConvFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                             const InnerContext *ctx, const kernel::KernelKey &desc,
                                             const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(op_parameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2D);
  MS_ASSERT(desc.data_type == kNumberTypeFloat32);
  auto conv_param = reinterpret_cast<ConvParameter *>(op_parameter);
  int group = conv_param->group_;
  bool use_winograd = false;
  int out_unit;
  if (primitive != nullptr && primitive->infer_flag()) {
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
  auto *restore_data = weight_tensor->data_c();
  auto restore_type = weight_tensor->data_type();
  bool dequant_flag =
    !weight_tensor->quant_params().empty() && weight_tensor->quant_params().front().inited && restore_data != nullptr;
  if (dequant_flag) {
    auto *dequant_weight = kernel::DequantUtil::DequantWeight(weight_tensor);
    if (dequant_weight == nullptr) {
      MS_LOG(ERROR) << "dequant data is nullptr.";
      free(op_parameter);
      return nullptr;
    }
    weight_tensor->set_data(dequant_weight);
  }

  kernel::LiteKernel *kernel;
  if (group == 1) {
    kernel = CpuConvFp32KernelSelect(inputs, outputs, op_parameter, ctx, primitive, use_winograd, out_unit);
  } else {
    kernel = CpuGroupConvFp32KernelCreator(inputs, outputs, op_parameter, ctx, primitive, group);
  }

  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    if (dequant_flag) {
      weight_tensor->FreeData();
      weight_tensor->set_data(restore_data);
      weight_tensor->set_data_type(restore_type);
    }
    free(op_parameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK && ret != RET_INFER_INVALID) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    if (dequant_flag) {
      weight_tensor->FreeData();
      weight_tensor->set_data(restore_data);
      weight_tensor->set_data_type(restore_type);
    }
    delete kernel;
    return nullptr;
  }

  if (dequant_flag) {
    weight_tensor->FreeData();
    weight_tensor->set_data(restore_data);
    weight_tensor->set_data_type(restore_type);
  }

  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Conv2D, CpuConvFp32KernelCreator)
}  // namespace mindspore::kernel
