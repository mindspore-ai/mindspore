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
#include "src/runtime/kernel/arm/fp32/convolution_slidewindow.h"
#include "src/runtime/kernel/arm/fp32/convolution_1x1.h"
#include "src/runtime/kernel/arm/fp32/convolution_3x3.h"
#include "src/runtime/kernel/arm/fp32/convolution_winograd.h"
#include "src/runtime/kernel/arm/nnacl/fp32/conv.h"
#include "src/runtime/kernel/arm/nnacl/common_func.h"
#include "schema/model_generated.h"
#include "src/kernel_factory.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;

namespace mindspore::kernel {
int ConvolutionCPUKernel::InitWeightBias() {
  int kernel_h = conv_param_->kernel_h_;
  int kernel_w = conv_param_->kernel_w_;
  int in_channel = conv_param_->input_channel_;
  int out_channel = conv_param_->output_channel_;
  int ic4 = UP_DIV(in_channel, C4NUM);
  int kernel_plane = kernel_h * kernel_w;
  int oc_block, oc_block_num;
#ifdef ENABLE_ARM32
  oc_block = C4NUM;
  oc_block_num = UP_DIV(out_channel, C4NUM);
#else
  oc_block = C8NUM;
  oc_block_num = UP_DIV(out_channel, C8NUM);
#endif
  int pack_weight_size = oc_block_num * oc_block * ic4 * C4NUM * kernel_plane;

  // init weight
  auto origin_weight = reinterpret_cast<float *>(in_tensors_.at(kWeightIndex)->Data());
  packed_weight_ = reinterpret_cast<float *>(malloc(pack_weight_size * sizeof(float)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed weight failed.";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, pack_weight_size * sizeof(float));
  PackWeightFp32(origin_weight, conv_param_, packed_weight_, oc_block, oc_block_num);

  // init bias
  bias_data_ = reinterpret_cast<float *>(malloc(oc_block_num * oc_block * sizeof(float)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, oc_block_num * oc_block * sizeof(float));
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->Data());
    memcpy(bias_data_, ori_bias, out_channel * sizeof(float));
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  return RET_OK;
}

int ConvolutionCPUKernel::InitTmpBuffer() {
  int kernel_h = conv_param_->kernel_h_;
  int kernel_w = conv_param_->kernel_w_;
  int in_batch = conv_param_->input_batch_;
  int in_channel = conv_param_->input_channel_;
  int ic4 = UP_DIV(in_channel, C4NUM);
  int out_channel = conv_param_->output_channel_;
  int kernel_plane = kernel_h * kernel_w;

  // malloc packed_inputs
  int output_count = conv_param_->output_h_ * conv_param_->output_w_;
  int output_tile_count = UP_DIV(output_count, TILE_NUM);
  int unit_size = kernel_plane * ic4 * C4NUM;
  int packed_input_size = output_tile_count * TILE_NUM * unit_size;
  /*=============================packed_input============================*/
  packed_input_ = reinterpret_cast<float *>(malloc(in_batch * packed_input_size * sizeof(float)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed input failed.";
    return RET_ERROR;
  }
  memset(packed_input_, 0, in_batch * packed_input_size * sizeof(float));

  /*=============================nhwc4_input_============================*/
  size_t nhwc4_input_size =
    ic4 * C4NUM * conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * sizeof(float);
  nhwc4_input_ = malloc(nhwc4_input_size);
  if (nhwc4_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc nhwc4 input failed.";
    return RET_ERROR;
  }
  memset(nhwc4_input_, 0, nhwc4_input_size);

  /*=============================tmp_output_block_============================*/
  tmp_output_block_ = reinterpret_cast<float *>(malloc(TILE_NUM * out_channel * sizeof(float)));
  if (tmp_output_block_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp output block failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

void ConvolutionCPUKernel::ConfigInputOutput() {
  // set output format
  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_tensor->SetFormat(schema::Format_NHWC);

  // select trans func for input
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto ret = CheckLayout(input_tensor);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Check layout failed.";
    return;
  }
#ifdef ENABLE_ARM32
  gemm_func_ = IndirectGemmFp32_8x4;
#else
  gemm_func_ = IndirectGemmFp32_8x8;
#endif
}

int ConvolutionCPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return RET_ERROR;
  }
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
  // config input output
  ConfigInputOutput();
  return RET_OK;
}

int ConvolutionCPUKernel::ReSize() {
  if (packed_input_ != nullptr) {
    free(packed_input_);
  }
  if (tmp_output_block_ != nullptr) {
    free(tmp_output_block_);
  }
  if (nhwc4_input_ != nullptr) {
    free(nhwc4_input_);
  }

  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
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

int ConvolutionCPUKernel::RunImpl(int task_id) {
  if (gemm_func_ == nullptr) {
    MS_LOG(ERROR) << "gemm_func is nullptr.";
    return RET_ERROR;
  }
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->Data());
  ConvFp32(reinterpret_cast<float *>(nhwc4_input_), packed_input_, packed_weight_,
           reinterpret_cast<float *>(bias_data_), tmp_output_block_, output_addr, task_id, conv_param_, gemm_func_);
  return RET_OK;
}

int ConvolutionImpl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv = reinterpret_cast<ConvolutionCPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto ori_input_data = input_tensor->Data();
  int in_batch = conv_param_->input_batch_;
  int in_h = conv_param_->input_h_;
  int in_w = conv_param_->input_w_;
  int in_channel = conv_param_->input_channel_;
  convert_func_(ori_input_data, nhwc4_input_, in_batch, in_h * in_w, in_channel);

  int error_code = LiteBackendParallelLaunch(ConvolutionImpl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

bool CheckIfUseSlideWindow(ConvParameter *conv_param) {
  int in_channel = conv_param->input_channel_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;
  int out_channel = conv_param->output_channel_;
  int ic4 = UP_DIV(in_channel, C4NUM);
  int oc4 = UP_DIV(out_channel, C4NUM);
  if (out_h * out_w <= 32 || ic4 < 4 || oc4 < 4) {
    return true;
  }
  return false;
}

kernel::LiteKernel *CpuConvFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
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
  conv_param->input_h_ = inputs.front()->Height();
  conv_param->input_w_ = inputs.front()->Width();
  conv_param->output_h_ = outputs.front()->Height();
  conv_param->output_w_ = outputs.front()->Width();
  bool use_winograd = false;
  int out_unit;
  InputTransformUnitFunc input_trans_func = nullptr;
  OutputTransformUnitFunc output_trans_func = nullptr;
  CheckIfUseWinograd(&use_winograd, &out_unit, conv_param, input_trans_func, output_trans_func);
  bool use_sw = CheckIfUseSlideWindow(conv_param);

  kernel::LiteKernel *kernel;
  if (kernel_h == 1 && kernel_w == 1) {
    kernel = new (std::nothrow) kernel::Convolution1x1CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  } else if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
    kernel = new (std::nothrow) kernel::Convolution3x3CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  } else if (use_winograd) {
    kernel =
      new (std::nothrow) kernel::ConvolutionWinogradCPUKernel(opParameter, inputs, outputs, ctx, primitive, out_unit);
  } else if (use_sw) {
    // kernel = new (std::nothrow) kernel::ConvolutionSWCPUKernel(opParameter, inputs, outputs, ctx, primitive);
    kernel = new (std::nothrow) kernel::ConvolutionCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  } else {
    kernel = new (std::nothrow) kernel::ConvolutionCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK && ret != RET_INFER_INVALID) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Conv2D, CpuConvFp32KernelCreator)
}  // namespace mindspore::kernel
