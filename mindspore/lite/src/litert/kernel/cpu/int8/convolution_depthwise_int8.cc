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

#include "src/litert/kernel/cpu/int8/convolution_depthwise_int8.h"
#include "include/errorcode.h"
#include "nnacl/int8/conv_depthwise_int8.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
ConvolutionDepthwiseInt8CPUKernel::~ConvolutionDepthwiseInt8CPUKernel() {
  if (packed_weight_sub_ != nullptr) {
    free(packed_weight_sub_);
    packed_weight_sub_ = nullptr;
  }
  FreeQuantParam();
}

int ConvolutionDepthwiseInt8CPUKernel::InitWeightBias() {
  // init weight, int8 -> int16
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  CHECK_NULL_RETURN(weight_tensor);
  auto origin_weight = reinterpret_cast<int8_t *>(weight_tensor->MutableData());
  CHECK_NULL_RETURN(origin_weight);
  int32_t channel = 0;
  int32_t height = 0;
  int32_t width = 0;
  if (CheckAndGetWeightParam(&channel, &height, &width) != RET_OK) {
    MS_LOG(ERROR) << "check weight shape info of weight tensor failed!";
    return RET_ERROR;
  }
  auto element_num = weight_tensor->ElementsNum();
  if (element_num <= 0 || element_num == INT32_MAX) {
    MS_LOG(ERROR) << "get element num failed! element num: " << element_num;
    return RET_ERROR;
  }
  int32_t pack_weight_size = channel * height * width;
  if (pack_weight_size < 0) {
    MS_LOG(ERROR) << "get pack_weight_size from weight_tensor failed.";
    return RET_ERROR;
  }
  auto tmp_weight = reinterpret_cast<int8_t *>(malloc(pack_weight_size * sizeof(int8_t)));
  if (tmp_weight == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  PackNCHWToNHWCInt8(origin_weight, tmp_weight, 1, height * width, channel);

  packed_weight_sub_ = reinterpret_cast<int16_t *>(malloc(pack_weight_size * sizeof(int16_t)));
  if (packed_weight_sub_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    free(tmp_weight);
    return RET_ERROR;
  }

  bool filter_per_channel = static_cast<bool>(conv_param_->conv_quant_arg_.per_channel_ & FILTER_PER_CHANNEL);
  if (filter_per_channel) {
    for (int i = 0; i < height * width; i++) {
      for (int c = 0; c < channel; c++) {
        int per_channel_weight_zp = conv_param_->conv_quant_arg_.filter_quant_args_[c].zp_;
        packed_weight_sub_[i * channel + c] = (int16_t)(tmp_weight[i * channel + c] - per_channel_weight_zp);
      }
    }
  } else {
    int weight_zp = conv_param_->conv_quant_arg_.filter_quant_args_[0].zp_;
    if (element_num == 0 || element_num > pack_weight_size) {
      MS_LOG(ERROR) << "weight_tensor->ElementsNum() is 0 or larger than pack_weight_size.";
      free(tmp_weight);
      return RET_ERROR;
    }
    for (int i = 0; i < element_num; i++) {
      packed_weight_sub_[i] = (int16_t)(tmp_weight[i] - weight_zp);
    }
  }
  free(tmp_weight);

  bias_data_ = reinterpret_cast<int32_t *>(malloc(channel * sizeof(int32_t)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  (void)memset(bias_data_, 0, channel * sizeof(int32_t));
  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_.at(kBiasIndex);
    CHECK_NULL_RETURN(bias_tensor);
    auto ori_bias = reinterpret_cast<int32_t *>(bias_tensor->MutableData());
    CHECK_NULL_RETURN(ori_bias);
    auto bias_element_num = bias_tensor->ElementsNum();
    MS_CHECK_GT(bias_element_num, 0, RET_ERROR);
    (void)memcpy(bias_data_, ori_bias, static_cast<size_t>(bias_element_num) * sizeof(int32_t));
  }
  return RET_OK;
}

int ConvolutionDepthwiseInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(in_tensors_[1]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  CHECK_LESS_RETURN(conv_param_->dilation_h_, C1NUM);
  CHECK_LESS_RETURN(conv_param_->dilation_w_, C1NUM);
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      in_tensors_[1]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", input1 data_type is "
                  << in_tensors_[1]->data_type() << ", output data_type is " << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  auto ret = ConvolutionBaseCPUKernel::SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set quant param failed.";
    return ret;
  }
  conv_param_->thread_num_ = MSMIN(thread_count_, conv_param_->output_h_);
  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Depthwise int8 InitWeightBias error!";
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionDepthwiseInt8CPUKernel::ReSize() { return ConvolutionBaseCPUKernel::Prepare(); }

int ConvolutionDepthwiseInt8CPUKernel::DoExecute(int task_id) {
  auto buffer = row_buffer_ + conv_param_->output_w_ * conv_param_->output_channel_ * task_id;
  ConvDwInt8(output_ptr_, buffer, input_ptr_, packed_weight_sub_, reinterpret_cast<int32_t *>(bias_data_), conv_param_,
             task_id);
  return RET_OK;
}

int ConvDwInt8Run(void *cdata, int task_id, float, float) {
  auto conv_dw_int8 = reinterpret_cast<ConvolutionDepthwiseInt8CPUKernel *>(cdata);
  auto ret = conv_dw_int8->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwiseInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseInt8CPUKernel::InitBuffer() {
  int output_row_size = conv_param_->thread_num_ * conv_param_->output_w_ * conv_param_->output_channel_;
  row_buffer_ = reinterpret_cast<int32_t *>(ms_context_->allocator->Malloc(output_row_size * sizeof(int)));
  if (row_buffer_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseInt8CPUKernel::Run() {
  auto ret = InitBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Depthwise int8 ReSize error!";
    ms_context_->allocator->Free(row_buffer_);
    row_buffer_ = nullptr;
    return ret;
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  CHECK_NULL_RETURN(input_tensor);
  input_ptr_ = reinterpret_cast<int8_t *>(input_tensor->MutableData());
  CHECK_NULL_RETURN(input_ptr_);

  auto output_tensor = out_tensors_.at(kOutputIndex);
  CHECK_NULL_RETURN(output_tensor);
  output_ptr_ = reinterpret_cast<int8_t *>(output_tensor->MutableData());
  CHECK_NULL_RETURN(output_ptr_);

  ret = ParallelLaunch(this->ms_context_, ConvDwInt8Run, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDwInt8Run error: error_code[" << ret << "]";
  }
  ms_context_->allocator->Free(row_buffer_);
  row_buffer_ = nullptr;
  return ret;
}
}  // namespace mindspore::kernel
