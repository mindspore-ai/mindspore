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
#include "src/runtime/kernel/arm/int8/convolution_3x3_int8.h"
#include "src/runtime/kernel/arm/nnacl/int8/conv_int8.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;

namespace mindspore::kernel {
void ConvolutionInt8CPUKernel::CheckSupportOptimize() {
  tile_num_ = 24;
#ifdef ENABLE_ARM32
  tile_num_ = 2;
  support_optimize_ = false;
#endif

#ifdef ENABLE_ARM64
  void *optimize_op_handler = OptimizeModule::GetInstance()->optimized_op_handler_;
  if (optimize_op_handler != nullptr) {
    dlerror();
    *(reinterpret_cast<void **>(&gemm_func_)) = dlsym(optimize_op_handler, "IndirectGemmInt8_optimize_handler");
    auto dlopen_error = dlerror();
    if (dlopen_error != nullptr) {
      MS_LOG(ERROR) << "load gemm func failed! " << dlopen_error << ".";
      tile_num_ = 4;
      support_optimize_ = false;
      gemm_func_ = nullptr;
    } else {
      // do nothing
    }
  } else {
    tile_num_ = 4;
    support_optimize_ = false;
  }
#endif
  conv_param_->tile_num_ = tile_num_;
}

int ConvolutionInt8CPUKernel::InitWeightBias() {
  int kernel_h = conv_param_->kernel_h_;
  int kernel_w = conv_param_->kernel_w_;
  int in_channel = conv_param_->input_channel_;
  int ic4 = UP_DIV(in_channel, C4NUM);
  int out_channel = conv_param_->output_channel_;
  int oc4 = UP_DIV(out_channel, C4NUM);
  int kernel_plane = kernel_h * kernel_w;
  int plane_c4 = UP_DIV(kernel_plane, C4NUM);
  int pack_weight_size = oc4 * ic4 * C4NUM * C4NUM * plane_c4 * C4NUM;
  auto filter_arg = conv_param_->conv_quant_arg_.filter_quant_args_;
  int32_t input_zp = conv_param_->conv_quant_arg_.input_quant_args_[0].zp_;

  // init weight
  auto origin_weight = reinterpret_cast<int8_t *>(in_tensors_.at(kWeightIndex)->Data());
  packed_weight_ = reinterpret_cast<int8_t *>(malloc(pack_weight_size));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed_weight_ failed.";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, pack_weight_size);
  auto *weight_sum = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t) * out_channel));
  for (int i = 0; i < out_channel; i++) weight_sum[i] = 0;
  PackWeightInt8(origin_weight, conv_param_, packed_weight_, weight_sum);

  // init bias
  bias_data_ = reinterpret_cast<int32_t *>(malloc(oc4 * C4NUM * sizeof(int32_t)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias_data_ failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, oc4 * C4NUM * sizeof(int32_t));
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<int32_t *>(in_tensors_.at(kBiasIndex)->Data());
    memcpy(bias_data_, ori_bias, out_channel * sizeof(int32_t));
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  auto *bias_data = reinterpret_cast<int32_t *>(bias_data_);
  int c4_kernel_plane_size = kernel_plane * ic4 * C4NUM;
  if (conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL) {
    for (int i = 0; i < out_channel; i++) {
      bias_data[i] += filter_arg[i].zp_ * input_zp * c4_kernel_plane_size - weight_sum[i] * input_zp;
    }
  } else {
    for (int i = 0; i < out_channel; i++) {
      bias_data[i] += filter_arg[0].zp_ * input_zp * c4_kernel_plane_size - weight_sum[i] * input_zp;
    }
  }
  free(weight_sum);
  return RET_OK;
}

int ConvolutionInt8CPUKernel::InitTmpBuffer() {
  int output_count = conv_param_->output_h_ * conv_param_->output_w_;
  int output_tile_count = UP_DIV(output_count, tile_num_);
  int in_channel = conv_param_->input_channel_;
  int ic4 = UP_DIV(in_channel, C4NUM);
  int kernel_plane = conv_param_->kernel_h_ * conv_param_->kernel_w_;
  int plane_c4 = UP_DIV(kernel_plane, C4NUM);
  int unit_size = plane_c4 * C4NUM * ic4 * C4NUM;
  int packed_input_size = output_tile_count * tile_num_ * unit_size;

  /*=============================packed_input_============================*/
  packed_input_ = reinterpret_cast<int8_t *>(malloc(conv_param_->input_batch_ * packed_input_size));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed_input_ failed.";
    return RET_ERROR;
  }
  memset(packed_input_, 0, conv_param_->input_batch_ * packed_input_size);

  /*=============================input_sum_============================*/
  size_t input_sum_size;
  if (conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL) {
    input_sum_size = conv_param_->output_channel_ * tile_num_ * thread_count_ * sizeof(int32_t);
  } else {
    input_sum_size = tile_num_ * thread_count_ * sizeof(int32_t);
  }
  input_sum_ = reinterpret_cast<int32_t *>(malloc(input_sum_size));
  if (input_sum_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_sum_ failed.";
    return RET_ERROR;
  }
  memset(input_sum_, 0, tile_num_ * thread_count_ * sizeof(int32_t));

  /*=============================tmp_dst_============================*/
  size_t tmp_dst_size = thread_count_ * tile_num_ * conv_param_->output_channel_ * sizeof(int32_t);
  tmp_dst_ = reinterpret_cast<int32_t *>(malloc(tmp_dst_size));
  if (tmp_dst_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_dst_ failed.";
    return RET_ERROR;
  }
  memset(tmp_dst_, 0, tmp_dst_size);

  /*=============================tmp_out_============================*/
  tmp_out_ = reinterpret_cast<int8_t *>(malloc(thread_count_ * tile_num_ * conv_param_->output_channel_));
  if (tmp_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_out_ failed.";
    return RET_ERROR;
  }

  /*=============================nhwc4_input_============================*/
  size_t nhwc4_input_size = ic4 * C4NUM * conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_;
  nhwc4_input_ = malloc(nhwc4_input_size);
  if (nhwc4_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc nhwc4 input failed.";
    return RET_ERROR;
  }
  memset(nhwc4_input_, 0, nhwc4_input_size);
  return RET_OK;
}

int ConvolutionInt8CPUKernel::InitWeightBiasOpt() {
  int kernel_h = conv_param_->kernel_h_;
  int kernel_w = conv_param_->kernel_w_;
  int in_channel = conv_param_->input_channel_;
  int ic4 = UP_DIV(in_channel, C4NUM);
  int out_channel = conv_param_->output_channel_;
  int oc4 = UP_DIV(out_channel, C4NUM);
  int kernel_plane = kernel_h * kernel_w;
  int pack_weight_size = oc4 * ic4 * C4NUM * C4NUM * kernel_plane;
  auto filter_arg = conv_param_->conv_quant_arg_.filter_quant_args_;
  int32_t input_zp = conv_param_->conv_quant_arg_.input_quant_args_[0].zp_;

  // init weight
  auto origin_weight = reinterpret_cast<int8_t *>(in_tensors_.at(kWeightIndex)->Data());
  packed_weight_ = reinterpret_cast<int8_t *>(malloc(pack_weight_size));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed_weight_ failed.";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, pack_weight_size);
  auto *weight_sum = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t) * out_channel));
  for (int i = 0; i < out_channel; i++) weight_sum[i] = 0;
  PackWeightInt8Opt(origin_weight, conv_param_, packed_weight_, weight_sum);

  // init bias
  bias_data_ = reinterpret_cast<int32_t *>(malloc(oc4 * C4NUM * sizeof(int32_t)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias_data_ failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, oc4 * C4NUM * sizeof(int32_t));
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<int32_t *>(in_tensors_.at(kBiasIndex)->Data());
    memcpy(bias_data_, ori_bias, out_channel * sizeof(int32_t));
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  auto *bias_data = reinterpret_cast<int32_t *>(bias_data_);
  int c4_kernel_plane_size = kernel_plane * ic4 * C4NUM;
  if (conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL) {
    for (int i = 0; i < out_channel; i++) {
      bias_data[i] += filter_arg[i].zp_ * input_zp * c4_kernel_plane_size - weight_sum[i] * input_zp;
    }
  } else {
    for (int i = 0; i < out_channel; i++) {
      bias_data[i] += filter_arg[0].zp_ * input_zp * c4_kernel_plane_size - weight_sum[i] * input_zp;
    }
  }
  free(weight_sum);
  return RET_OK;
}

int ConvolutionInt8CPUKernel::InitTmpBufferOpt() {
  int output_count = conv_param_->output_h_ * conv_param_->output_w_;
  int output_tile_count = UP_DIV(output_count, tile_num_);
  int in_channel = conv_param_->input_channel_;
  int ic4 = UP_DIV(in_channel, C4NUM);
  int kernel_plane = conv_param_->kernel_h_ * conv_param_->kernel_w_;
  int unit_size = kernel_plane * ic4 * C4NUM;
  int packed_input_size = output_tile_count * tile_num_ * unit_size;

  /*=============================packed_input_============================*/
  packed_input_ = reinterpret_cast<int8_t *>(malloc(conv_param_->input_batch_ * packed_input_size));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed_input_ failed.";
    return RET_ERROR;
  }
  memset(packed_input_, 0, conv_param_->input_batch_ * packed_input_size);

  /*=============================input_sum_============================*/
  size_t input_sum_size;
  if (conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL) {
    input_sum_size = conv_param_->output_channel_ * tile_num_ * thread_count_ * sizeof(int32_t);
  } else {
    input_sum_size = tile_num_ * thread_count_ * sizeof(int32_t);
  }
  input_sum_ = reinterpret_cast<int32_t *>(malloc(input_sum_size));
  if (input_sum_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_sum_ failed.";
    return RET_ERROR;
  }
  memset(input_sum_, 0, tile_num_ * thread_count_ * sizeof(int32_t));

  /*=============================tmp_dst_============================*/
  size_t tmp_dst_size = thread_count_ * tile_num_ * conv_param_->output_channel_ * sizeof(int32_t);
  tmp_dst_ = reinterpret_cast<int32_t *>(malloc(tmp_dst_size));
  if (tmp_dst_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_dst_ failed.";
    return RET_ERROR;
  }
  memset(tmp_dst_, 0, tmp_dst_size);

  /*=============================tmp_out_============================*/
  tmp_out_ = reinterpret_cast<int8_t *>(malloc(thread_count_ * tile_num_ * conv_param_->output_channel_));
  if (tmp_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_out_ failed.";
    return RET_ERROR;
  }

  /*=============================nhwc4_input_============================*/
  size_t nhwc4_input_size = ic4 * C4NUM * conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_;
  nhwc4_input_ = malloc(nhwc4_input_size);
  if (nhwc4_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc nhwc4 input failed.";
    return RET_ERROR;
  }
  memset(nhwc4_input_, 0, nhwc4_input_size);
  return RET_OK;
}

void ConvolutionInt8CPUKernel::ConfigInputOutput() {
  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_tensor->SetFormat(schema::Format_NHWC);
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto ret = CheckLayout(input_tensor);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Check layout failed.";
    return;
  }
}

int ConvolutionInt8CPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return RET_ERROR;
  }
  // config input output
  ConfigInputOutput();
  CheckSupportOptimize();
  ret = SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set quant param failed.";
    return ret;
  }
  // init for opt
  if (support_optimize_) {
    ret = InitOpt();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Initialization for optimized int8 conv failed.";
      return RET_ERROR;
    }
    return RET_OK;
  }

  // init for situation that not support sdot
  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  // init tmp input, output
  ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionInt8CPUKernel::InitOpt() {
  auto ret = InitWeightBiasOpt();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  // init tmp input, output
  ret = InitTmpBufferOpt();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionInt8CPUKernel::ReSize() {
  if (packed_input_ != nullptr) {
    free(packed_input_);
  }
  if (input_sum_ != nullptr) {
    free(input_sum_);
  }
  if (tmp_dst_ != nullptr) {
    free(tmp_dst_);
  }
  if (tmp_out_ != nullptr) {
    free(tmp_out_);
  }

  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return RET_ERROR;
  }
  if (support_optimize_) {
    ret = InitTmpBufferOpt();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Init tmp buffer for opt failed.";
      return RET_ERROR;
    }
    return RET_OK;
  }
  // init tmp input, output
  ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionInt8CPUKernel::RunImpl(int task_id) {
  auto output_addr = reinterpret_cast<int8_t *>(out_tensors_.at(kOutputIndex)->Data());
  if (support_optimize_) {
    ConvInt8Opt(reinterpret_cast<int8_t *>(nhwc4_input_), packed_input_, packed_weight_,
                reinterpret_cast<int32_t *>(bias_data_), tmp_dst_, tmp_out_, output_addr, input_sum_, task_id,
                conv_param_, gemm_func_);
  } else {
    ConvInt8(reinterpret_cast<int8_t *>(nhwc4_input_), packed_input_, packed_weight_,
             reinterpret_cast<int32_t *>(bias_data_), tmp_dst_, tmp_out_, output_addr, input_sum_, task_id,
             conv_param_);
  }
  return RET_OK;
}

int ConvolutionInt8Impl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv = reinterpret_cast<ConvolutionInt8CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution Int8 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto ori_input_data = input_tensor->Data();
  int in_batch = conv_param_->input_batch_;
  int in_h = conv_param_->input_h_;
  int in_w = conv_param_->input_w_;
  int in_channel = conv_param_->input_channel_;
  convert_func_(ori_input_data, nhwc4_input_, in_batch, in_h * in_w, in_channel);

  int error_code = LiteBackendParallelLaunch(ConvolutionInt8Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv int8 error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuConvInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs,
                                             OpParameter *opParameter, const Context *ctx,
                                             const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Conv2D);
  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int stride_h = conv_param->stride_h_;
  int stride_w = conv_param->stride_w_;
  int dilation_h = conv_param->dilation_h_;
  int dilation_w = conv_param->dilation_w_;
  kernel::LiteKernel *kernel;
  if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
    kernel = new (std::nothrow) kernel::Convolution3x3Int8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  } else {
    kernel = new (std::nothrow) kernel::ConvolutionInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
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

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Conv2D, CpuConvInt8KernelCreator)
}  // namespace mindspore::kernel
