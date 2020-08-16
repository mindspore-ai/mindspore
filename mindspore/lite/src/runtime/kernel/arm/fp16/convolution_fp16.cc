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

#include "src/runtime/kernel/arm/fp16/convolution_fp16.h"
#include <vector>
#include "src/runtime/kernel/arm/fp16/convolution_sw_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_3x3_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_1x1_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/conv_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/cast_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/pack_fp16.h"
#include "src/runtime/kernel/arm/fp16/layout_transform_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "nnacl/winograd_utils.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;

namespace mindspore::kernel {
int ConvolutionFP16CPUKernel::InitWeightBias() {
  int kernel_h = conv_param_->kernel_h_;
  int kernel_w = conv_param_->kernel_w_;
  int in_channel = conv_param_->input_channel_;
  int out_channel = conv_param_->output_channel_;
  int oc8 = UP_DIV(out_channel, C8NUM);
  int ic4 = UP_DIV(in_channel, C4NUM);
  int kernel_plane = kernel_h * kernel_w;
  int pack_weight_size = oc8 * ic4 * C8NUM * C4NUM * kernel_plane;

  // init weight
  auto ret = ConvolutionBaseFP16CPUKernel::GetExecuteFilter();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get Execute filter failed.";
    return ret;
  }
  packed_weight_ = reinterpret_cast<float16_t *>(malloc(pack_weight_size * sizeof(float16_t)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed_weight_ failed.";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, pack_weight_size * sizeof(float16_t));
  PackWeightFp16(execute_weight_, conv_param_, packed_weight_);

  // init bias
  bias_data_ = malloc(oc8 * C8NUM * sizeof(float16_t));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias_data_ failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, oc8 * C8NUM * sizeof(float16_t));
  auto fp16_bias_data = reinterpret_cast<float16_t *>(bias_data_);
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->Data());
    for (int i = 0; i < out_channel; ++i) {
      fp16_bias_data[i] = (float16_t)ori_bias[i];
    }
  } else {
    MS_ASSERT(inputs_.size() == kInputSize1);
  }
  return RET_OK;
}

int ConvolutionFP16CPUKernel::InitTmpBuffer() {
  int kernel_h = conv_param_->kernel_h_;
  int kernel_w = conv_param_->kernel_w_;
  int in_batch = conv_param_->input_batch_;
  int in_channel = conv_param_->input_channel_;
  int out_channel = conv_param_->output_channel_;
  int channel_block = UP_DIV(in_channel, C4NUM);
  int kernel_plane = kernel_h * kernel_w;

  // malloc packed_inputs
  int cal_num = 16;
  int output_count = conv_param_->output_h_ * conv_param_->output_w_;
  int output_tile_count = UP_DIV(output_count, cal_num);
  int unit_size = kernel_plane * channel_block * C4NUM;
  int packed_input_size = output_tile_count * cal_num * unit_size;

  /*=============================packed_input_============================*/
  packed_input_ = reinterpret_cast<float16_t *>(malloc(in_batch * packed_input_size * sizeof(float16_t)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed_input_ failed.";
    return RET_ERROR;
  }
  memset(packed_input_, 0, in_batch * packed_input_size * sizeof(float16_t));

  /*=============================fp16_input_============================*/
  size_t fp16_input_size =
    in_channel * conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * sizeof(float16_t);
  fp16_input_ = reinterpret_cast<float16_t *>(malloc(fp16_input_size));
  if (fp16_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc fp16_input_ failed.";
    return RET_ERROR;
  }

  /*=============================nhwc4_input_============================*/
  size_t nhwc4_input_size = channel_block * C4NUM * conv_param_->input_batch_ * conv_param_->input_h_ *
                            conv_param_->input_w_ * sizeof(float16_t);
  nhwc4_input_ = malloc(nhwc4_input_size);
  if (nhwc4_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc nhwc4_input_ failed.";
    return RET_ERROR;
  }
  memset(nhwc4_input_, 0, nhwc4_input_size);

  /*=============================tmp_output_block_============================*/
  tmp_output_block_ = reinterpret_cast<float16_t *>(malloc(cal_num * out_channel * sizeof(float16_t)));
  if (tmp_output_block_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_output_block_ failed.";
    return RET_ERROR;
  }

  /*=============================fp16_out_============================*/
  size_t fp16_output_size =
    out_channel * conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * sizeof(float16_t);
  fp16_out_ = reinterpret_cast<float16_t *>(malloc(fp16_output_size));
  if (fp16_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc fp16_out_ failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

void ConvolutionFP16CPUKernel::ConfigInputOutput() {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto input_format = input_tensor->GetFormat();
  schema::Format execute_format = schema::Format_NHWC4;
  convert_func_ = LayoutTransformFp16(input_format, execute_format);
  if (convert_func_ == nullptr) {
    MS_LOG(ERROR) << "layout convert func is nullptr.";
    return;
  }
}

int ConvolutionFP16CPUKernel::Init() {
  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init fail!ret: " << ret;
    return ret;
  }
  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }
  ConfigInputOutput();
  return RET_OK;
}

int ConvolutionFP16CPUKernel::ReSize() {
  if (packed_input_ != nullptr) {
    free(packed_input_);
  }
  if (tmp_output_block_ != nullptr) {
    free(tmp_output_block_);
  }
  if (nhwc4_input_ != nullptr) {
    free(nhwc4_input_);
  }
  if (fp16_input_ != nullptr) {
    free(fp16_input_);
  }
  if (fp16_out_ != nullptr) {
    free(fp16_out_);
  }

  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return ret;
  }
  ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionFP16CPUKernel::RunImpl(int task_id) {
  ConvFp16(reinterpret_cast<float16_t *>(nhwc4_input_), packed_input_, packed_weight_,
           reinterpret_cast<float16_t *>(bias_data_), tmp_output_block_, execute_output_, task_id, conv_param_);
  return RET_OK;
}

int ConvolutionFp16Impl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv = reinterpret_cast<ConvolutionFP16CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionFp16 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionFP16CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }

  ret = ConvolutionBaseFP16CPUKernel::GetExecuteTensor();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get Execute tensor failed.";
    return ret;
  }

  int in_batch = conv_param_->input_batch_;
  int in_h = conv_param_->input_h_;
  int in_w = conv_param_->input_w_;
  int in_channel = conv_param_->input_channel_;
  convert_func_(reinterpret_cast<void *>(execute_input_), nhwc4_input_, in_batch, in_h * in_w, in_channel);

  int error_code = LiteBackendParallelLaunch(ConvolutionFp16Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv fp16 error error_code[" << error_code << "]";
    return RET_ERROR;
  }

  ConvolutionBaseFP16CPUKernel::IfCastOutput();
  return RET_OK;
}

kernel::LiteKernel *CpuConvFp16KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
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
  kernel::LiteKernel *kernel = nullptr;
  if (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1) {
    kernel = new (std::nothrow) kernel::Convolution3x3FP16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  } else if (kernel_h == 1 && kernel_w == 1) {
    kernel = new (std::nothrow) kernel::Convolution1x1FP16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  } else {
    bool use_winograd = false;
    int out_unit;
    InputTransformUnitFunc input_trans_func = nullptr;
    OutputTransformUnitFunc output_trans_func = nullptr;
    CheckIfUseWinograd(&use_winograd, &out_unit, conv_param, input_trans_func, output_trans_func);
    if (kernel_h != 1 && kernel_w != 1 && !use_winograd) {
      kernel = new (std::nothrow) kernel::ConvolutionFP16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
    }
  }
  if (kernel == nullptr) {
    MS_LOG(DEBUG) << "Create conv fp16 kernel failed.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(INFO) << "Init fp16 kernel failed, name: " << opParameter->name_
                 << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Conv2D, CpuConvFp16KernelCreator)
}  // namespace mindspore::kernel
