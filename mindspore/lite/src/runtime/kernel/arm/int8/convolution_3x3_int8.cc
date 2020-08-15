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

#include "src/runtime/kernel/arm/int8/convolution_3x3_int8.h"
#include "src/runtime/kernel/arm/nnacl/int8/conv_int8.h"
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
void ProcessFilterUint8(int8_t *origin_weight, int16_t *dst_weight, ConvParameter *conv_param) {
  auto input_channel = conv_param->input_channel_;
  auto output_channel = conv_param->output_channel_;
  auto kernel_plane = conv_param->kernel_w_ * conv_param->kernel_h_;
  int iC8 = UP_DIV(input_channel, C8NUM);

  size_t tmp_size = output_channel * iC8 * C8NUM * kernel_plane * sizeof(int16_t);
  auto tmp_addr = reinterpret_cast<int16_t *>(malloc(tmp_size));
  memset(tmp_addr, 0, tmp_size);
  PackWeightToC8Int8(origin_weight, tmp_addr, conv_param);
  Conv3x3Int8FilterTransform(tmp_addr, dst_weight, iC8, output_channel, kernel_plane);

  free(tmp_addr);
}

Convolution3x3Int8CPUKernel::~Convolution3x3Int8CPUKernel() {
  if (transformed_filter_addr_ != nullptr) {
    free(transformed_filter_addr_);
  }
  if (input_data_ != nullptr) {
    free(input_data_);
  }
  if (tile_buffer_ != nullptr) {
    free(tile_buffer_);
  }
  if (block_unit_buffer_ != nullptr) {
    free(block_unit_buffer_);
  }
  if (tmp_dst_buffer_ != nullptr) {
    free(tmp_dst_buffer_);
  }
  if (tmp_out_ != nullptr) {
    free(tmp_out_);
  }
  FreeQuantParam();
}

int Convolution3x3Int8CPUKernel::InitWeightBias() {
  auto input_channel = conv_param_->input_channel_;
  auto output_channel = conv_param_->output_channel_;
  int iC8 = UP_DIV(input_channel, C8NUM);
  int oC4 = UP_DIV(output_channel, C4NUM);
  // init weight
  size_t transformed_size = iC8 * C8NUM * oC4 * C4NUM * 16 * sizeof(int16_t);
  transformed_filter_addr_ = reinterpret_cast<int16_t *>(malloc(transformed_size));
  if (transformed_filter_addr_ == nullptr) {
    MS_LOG(ERROR) << "malloc transformed_filter_addr_ failed.";
    return RET_ERROR;
  }
  memset(transformed_filter_addr_, 0, transformed_size);
  auto weight_data = reinterpret_cast<int8_t *>(in_tensors_.at(kWeightIndex)->Data());
  ProcessFilterUint8(weight_data, transformed_filter_addr_, conv_param_);

  // init bias
  size_t new_bias_size = oC4 * C4NUM * sizeof(int32_t);
  bias_data_ = reinterpret_cast<int32_t *>(malloc(new_bias_size));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias_data_ failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, new_bias_size);
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias_addr = reinterpret_cast<int32_t *>(in_tensors_.at(kBiasIndex)->Data());
    memcpy(bias_data_, ori_bias_addr, output_channel * sizeof(int32_t));
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  return RET_OK;
}

int Convolution3x3Int8CPUKernel::InitTmpBuffer() {
  int ic8 = UP_DIV(conv_param_->input_channel_, C8NUM);
  int oc4 = UP_DIV(conv_param_->output_channel_, C4NUM);
  int in_batch = conv_param_->input_batch_;
  int input_w = conv_param_->input_w_;
  int input_h = conv_param_->input_h_;
  int output_batch = conv_param_->output_batch_;
  int output_w = conv_param_->output_w_;
  int output_h = conv_param_->output_h_;

  /*=============================tile_buffer_============================*/
  size_t tile_buffer_size = thread_count_ * TILE_NUM * 16 * ic8 * C8NUM * sizeof(int16_t);
  tile_buffer_ = reinterpret_cast<int16_t *>(malloc(tile_buffer_size));
  if (tile_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc tile_buffer_ failed.";
    return RET_ERROR;
  }
  memset(tile_buffer_, 0, tile_buffer_size);

  /*=============================block_unit_buffer_============================*/
  size_t block_unit_buffer_size = thread_count_ * 4 * 4 * C8NUM * sizeof(int16_t);
  block_unit_buffer_ = reinterpret_cast<int16_t *>(malloc(block_unit_buffer_size));
  if (block_unit_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc block_unit_buffer_ failed.";
    return RET_ERROR;
  }
  memset(block_unit_buffer_, 0, block_unit_buffer_size);

  /*=============================tmp_dst_buffer_============================*/
  size_t tmp_dst_buffer_size = thread_count_ * TILE_NUM * 16 * oc4 * C4NUM * sizeof(int32_t);
  tmp_dst_buffer_ = reinterpret_cast<int32_t *>(malloc(tmp_dst_buffer_size));
  if (tmp_dst_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_dst_buffer_ failed.";
    return RET_ERROR;
  }
  memset(tmp_dst_buffer_, 0, tmp_dst_buffer_size);

  /*=============================tmp_out_============================*/
  size_t tmp_out_size = oc4 * C4NUM * output_batch * output_w * output_h * sizeof(uint8_t);
  tmp_out_ = reinterpret_cast<int8_t *>(malloc(tmp_out_size));
  if (tmp_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_out_ failed.";
    return RET_ERROR;
  }
  memset(tmp_out_, 0, tmp_out_size);

  /*=============================input_data_============================*/
  size_t c8_input_size = in_batch * input_h * input_w * ic8 * C8NUM * sizeof(int16_t);
  input_data_ = reinterpret_cast<int16_t *>(malloc(c8_input_size));
  if (input_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_data_ failed.";
    return RET_ERROR;
  }
  memset(input_data_, 0, c8_input_size);
  return RET_OK;
}

void Convolution3x3Int8CPUKernel::ConfigInputOutput() {
  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_tensor->SetFormat(schema::Format_NHWC);
}

int Convolution3x3Int8CPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return RET_ERROR;
  }
  ret = SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set quant param failed.";
    return ret;
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

int Convolution3x3Int8CPUKernel::ReSize() {
  if (input_data_ != nullptr) {
    free(input_data_);
  }
  if (tile_buffer_ != nullptr) {
    free(tile_buffer_);
  }
  if (block_unit_buffer_ != nullptr) {
    free(block_unit_buffer_);
  }
  if (tmp_dst_buffer_ != nullptr) {
    free(tmp_dst_buffer_);
  }
  if (tmp_out_ != nullptr) {
    free(tmp_out_);
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

int Convolution3x3Int8CPUKernel::RunImpl(int task_id) {
  auto output_addr = reinterpret_cast<int8_t *>(out_tensors_.at(kOutputIndex)->Data());
  Conv3x3Int8(input_data_, transformed_filter_addr_, reinterpret_cast<int32_t *>(bias_data_), output_addr, tile_buffer_,
              block_unit_buffer_, tmp_dst_buffer_, tmp_out_, task_id, conv_param_);
  return RET_OK;
}

int Convolution3x3Int8Impl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv = reinterpret_cast<Convolution3x3Int8CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution3x3 Int8 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution3x3Int8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  auto input_addr = reinterpret_cast<int8_t *>(in_tensors_.at(kInputIndex)->Data());
  PackInputToC8Int8(input_addr, input_data_, conv_param_);

  int error_code = LiteBackendParallelLaunch(Convolution3x3Int8Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv3x3 int8 error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  // get real output
  auto out_tensor = out_tensors_.front();
  auto out_data = reinterpret_cast<int8_t *>(out_tensor->Data());
  PackNC4HW4ToNHWCInt8(tmp_out_, out_data, conv_param_->output_batch_, conv_param_->output_h_ * conv_param_->output_w_,
                       conv_param_->output_channel_);
  return RET_OK;
}
}  // namespace mindspore::kernel
