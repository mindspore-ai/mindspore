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
#include "src/runtime/kernel/arm/opclib/fp32/conv.h"
#include "schema/model_generated.h"
#include "src/kernel_factory.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;

namespace mindspore::kernel {
int ConvolutionCPUKernel::InitWeightBias() {
  int kernel_h = conv_param_->kernel_h_;
  int kernel_w = conv_param_->kernel_w_;
  int in_channel = conv_param_->input_channel_;
  int out_channel = conv_param_->output_channel_;
  int oc8 = UP_DIV(out_channel, C8NUM);
  int ic4 = UP_DIV(in_channel, C4NUM);
  int kernel_plane = kernel_h * kernel_w;
  int pack_weight_size = oc8 * ic4 * C8NUM * C4NUM * kernel_plane;

  // init weight
  auto origin_weight = reinterpret_cast<float *>(inputs_.at(kWeightIndex)->Data());
  packed_weight_ = reinterpret_cast<float *>(malloc(pack_weight_size * sizeof(float)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed weight failed.";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, pack_weight_size * sizeof(float));
  PackWeightFp32(origin_weight, conv_param_, packed_weight_);

  // init bias
  bias_data_ = reinterpret_cast<float *>(malloc(oc8 * C8NUM * sizeof(float)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, oc8 * C8NUM * sizeof(float));
  if (inputs_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<float *>(inputs_.at(kBiasIndex)->Data());
    memcpy(bias_data_, ori_bias, out_channel * sizeof(float));
  } else {
    MS_ASSERT(inputs_.size() == kInputSize1);
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
  packed_input_ = reinterpret_cast<float *>(malloc(in_batch * packed_input_size * sizeof(float)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed input failed.";
    return RET_ERROR;
  }
  memset(packed_input_, 0, in_batch * packed_input_size * sizeof(float));

  size_t nhwc4_input_size =
    ic4 * C4NUM * conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * sizeof(float);
  nhwc4_input_ = malloc(nhwc4_input_size);
  if (nhwc4_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc nhwc4 input failed.";
    return RET_ERROR;
  }
  memset(nhwc4_input_, 0, nhwc4_input_size);

  // tmp out
  tmp_output_block_ = reinterpret_cast<float *>(malloc(TILE_NUM * out_channel * sizeof(float)));
  if (tmp_output_block_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp output block failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

void ConvolutionCPUKernel::ConfigInputOutput() {
  // set output format
  auto output_tensor = outputs_.at(kOutputIndex);
  output_tensor->SetFormat(schema::Format_NHWC);

  // select trans func for input
  auto input_tensor = inputs_.at(kInputIndex);
  auto ret = CheckLayout(input_tensor);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Check layout failed.";
    return;
  }
}

int ConvolutionCPUKernel::Init() {
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
  auto output_addr = reinterpret_cast<float *>(outputs_.at(kOutputIndex)->Data());
  ConvFp32(reinterpret_cast<float *>(nhwc4_input_), packed_input_, packed_weight_,
           reinterpret_cast<float *>(bias_data_), tmp_output_block_, output_addr, task_id, conv_param_);
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
  auto input_tensor = inputs_.at(kInputIndex);
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
}  // namespace mindspore::kernel

