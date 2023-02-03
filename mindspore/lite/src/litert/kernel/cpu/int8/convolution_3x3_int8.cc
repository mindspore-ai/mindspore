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

#include "src/litert/kernel/cpu/int8/convolution_3x3_int8.h"
#include "nnacl/int8/conv3x3_int8.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
constexpr size_t kUnitBufferMultipler = 4 * 4;
}  // namespace
int ProcessFilterUint8(const int8_t *origin_weight, int16_t *dst_weight, const ConvParameter *conv_param) {
  CHECK_NULL_RETURN(conv_param);
  CHECK_NULL_RETURN(origin_weight);
  auto input_channel = conv_param->input_channel_;
  auto output_channel = conv_param->output_channel_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param->kernel_w_, conv_param->kernel_h_, RET_ERROR);
  auto kernel_plane = conv_param->kernel_w_ * conv_param->kernel_h_;
  int iC8 = UP_DIV(input_channel, C8NUM);

  size_t tmp_size =
    static_cast<size_t>(output_channel) * static_cast<size_t>(iC8) * C8NUM * kernel_plane * sizeof(int16_t);
  auto tmp_addr = reinterpret_cast<int16_t *>(malloc(tmp_size));
  if (tmp_addr == nullptr) {
    return RET_ERROR;
  }
  memset(tmp_addr, 0, tmp_size);
  PackWeightToC8Int8(origin_weight, tmp_addr, conv_param);
  Conv3x3Int8FilterTransform(tmp_addr, dst_weight, iC8, output_channel, kernel_plane);

  free(tmp_addr);
  return RET_OK;
}

void Convolution3x3Int8CPUKernel::FreeTmpBuffer() {
  if (input_data_ != nullptr) {
    ctx_->allocator->Free(input_data_);
    input_data_ = nullptr;
  }
  if (tile_buffer_ != nullptr) {
    ctx_->allocator->Free(tile_buffer_);
    tile_buffer_ = nullptr;
  }
  if (block_unit_buffer_ != nullptr) {
    ctx_->allocator->Free(block_unit_buffer_);
    block_unit_buffer_ = nullptr;
  }
  if (tmp_dst_buffer_ != nullptr) {
    ctx_->allocator->Free(tmp_dst_buffer_);
    tmp_dst_buffer_ = nullptr;
  }
  if (tmp_out_ != nullptr) {
    ctx_->allocator->Free(tmp_out_);
    tmp_out_ = nullptr;
  }
}

Convolution3x3Int8CPUKernel::~Convolution3x3Int8CPUKernel() {
  if (transformed_filter_addr_ != nullptr) {
    free(transformed_filter_addr_);
    transformed_filter_addr_ = nullptr;
  }
  FreeQuantParam();
}

int Convolution3x3Int8CPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  CHECK_NULL_RETURN(filter_tensor);
  auto input_channel = filter_tensor->Channel();
  if (input_channel < 0) {
    MS_LOG(ERROR) << "get channel from filter_tensor failed.";
    return RET_ERROR;
  }
  auto output_channel = filter_tensor->Batch();
  if (output_channel < 0) {
    MS_LOG(ERROR) << "get batch from filter_tensor failed.";
    return RET_ERROR;
  }
  conv_param_->input_channel_ = input_channel;
  conv_param_->output_channel_ = output_channel;
  int iC8 = UP_DIV(input_channel, C8NUM);
  int oC4 = UP_DIV(output_channel, C4NUM);
  // init weight
  size_t transformed_size =
    static_cast<size_t>(iC8) * C8NUM * static_cast<size_t>(oC4) * C4NUM * kUnitBufferMultipler * sizeof(int16_t);
  if (transformed_size > 0) {
    transformed_filter_addr_ = reinterpret_cast<int16_t *>(malloc(transformed_size));
    if (transformed_filter_addr_ == nullptr) {
      MS_LOG(ERROR) << "malloc transformed_filter_addr_ failed.";
      return RET_ERROR;
    }
    memset(transformed_filter_addr_, 0, transformed_size);
  }
  auto weight_data = reinterpret_cast<int8_t *>(in_tensors_.at(kWeightIndex)->MutableData());
  CHECK_NULL_RETURN(weight_data);
  auto ret = ProcessFilterUint8(weight_data, transformed_filter_addr_, conv_param_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ProcessFilterUint8 failed.";
    return ret;
  }

  // init bias
  size_t new_bias_size = static_cast<size_t>(oC4) * C4NUM * sizeof(int32_t);
  if (new_bias_size > 0) {
    bias_data_ = reinterpret_cast<int32_t *>(malloc(new_bias_size));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc bias_data_ failed.";
      return RET_ERROR;
    }
    memset(bias_data_, 0, new_bias_size);
  }
  if (in_tensors_.size() == kInputSize2) {
    CHECK_NULL_RETURN(in_tensors_.at(kBiasIndex));
    auto ori_bias_addr = reinterpret_cast<int32_t *>(in_tensors_.at(kBiasIndex)->MutableData());
    CHECK_NULL_RETURN(ori_bias_addr);
    memcpy(bias_data_, ori_bias_addr, static_cast<size_t>(output_channel) * sizeof(int32_t));
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  return RET_OK;
}

int Convolution3x3Int8CPUKernel::InitTmpBuffer() {
  int oc4 = UP_DIV(conv_param_->output_channel_, C4NUM);
  int output_batch = conv_param_->output_batch_;
  int output_w = conv_param_->output_w_;
  int output_h = conv_param_->output_h_;
  int ic8 = UP_DIV(conv_param_->input_channel_, C8NUM);
  MS_ASSERT(ctx_->allocator != nullptr);

  size_t c8_input_size = static_cast<size_t>(conv_param_->input_batch_) * static_cast<size_t>(conv_param_->input_h_) *
                         static_cast<size_t>(conv_param_->input_w_) * static_cast<size_t>(ic8) * C8NUM *
                         sizeof(int16_t);
  input_data_ = reinterpret_cast<int16_t *>(ctx_->allocator->Malloc(c8_input_size));
  if (input_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_data_ failed.";
    return RET_ERROR;
  }

  size_t tile_buffer_size = thread_count_ * TILE_NUM * C16NUM * ic8 * C8NUM * sizeof(int16_t);
  tile_buffer_ = reinterpret_cast<int16_t *>(ctx_->allocator->Malloc(tile_buffer_size));
  if (tile_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc tile_buffer_ failed.";
    return RET_ERROR;
  }

  size_t block_unit_buffer_size = thread_count_ * kUnitBufferMultipler * C8NUM * sizeof(int16_t);
  block_unit_buffer_ = reinterpret_cast<int16_t *>(ctx_->allocator->Malloc(block_unit_buffer_size));
  if (block_unit_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc block_unit_buffer_ failed.";
    return RET_ERROR;
  }

  size_t tmp_dst_buffer_size = thread_count_ * TILE_NUM * kUnitBufferMultipler * oc4 * C4NUM * sizeof(int32_t);
  tmp_dst_buffer_ = reinterpret_cast<int32_t *>(ctx_->allocator->Malloc(tmp_dst_buffer_size));
  if (tmp_dst_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_dst_buffer_ failed.";
    return RET_ERROR;
  }

  size_t tmp_out_size = oc4 * C4NUM * output_batch * output_w * output_h * sizeof(uint8_t);
  tmp_out_ = reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(tmp_out_size));
  if (tmp_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_out_ failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution3x3Int8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(in_tensors_[1]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      in_tensors_[1]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", input1 data_type is "
                  << in_tensors_[1]->data_type() << ", output data_type is " << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  auto ret = SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set quant param failed.";
    return ret;
  }
  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int Convolution3x3Int8CPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }

  ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution3x3Int8CPUKernel::RunImpl(int task_id) {
  Conv3x3Int8(input_data_, transformed_filter_addr_, reinterpret_cast<int32_t *>(bias_data_), nullptr, tile_buffer_,
              block_unit_buffer_, tmp_dst_buffer_, tmp_out_, task_id, conv_param_);
  return RET_OK;
}

int Convolution3x3Int8Impl(void *cdata, int task_id, float, float) {
  auto conv = reinterpret_cast<Convolution3x3Int8CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution3x3 Int8 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution3x3Int8CPUKernel::Run() {
  // malloc tmp buffer
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(in_tensors_.at(kInputIndex));
  auto input_addr = reinterpret_cast<int8_t *>(in_tensors_.at(kInputIndex)->MutableData());
  CHECK_NULL_RETURN(input_addr);
  PackInputToC8Int8(input_addr, input_data_, conv_param_);

  int error_code = ParallelLaunch(this->ms_context_, Convolution3x3Int8Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv3x3 int8 error error_code[" << error_code << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  // get real output
  auto out_tensor = out_tensors_.front();
  CHECK_NULL_RETURN(out_tensor);
  auto out_data = reinterpret_cast<int8_t *>(out_tensor->MutableData());
  CHECK_NULL_RETURN(out_data);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_h_, conv_param_->output_w_, RET_ERROR);
  PackNC4HW4ToNHWCInt8(tmp_out_, out_data, conv_param_->output_batch_, conv_param_->output_h_ * conv_param_->output_w_,
                       conv_param_->output_channel_);
  FreeTmpBuffer();
  return RET_OK;
}
}  // namespace mindspore::kernel
