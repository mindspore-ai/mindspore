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

#include "src/litert/kernel/cpu/base/convolution_base.h"
#include <cfloat>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType;

namespace mindspore::kernel {
void *ConvolutionBaseCPUKernel::MallocAlignedData(size_t alignment, size_t size) {
  MS_CHECK_TRUE_RET(size + alignment < MAX_MALLOC_SIZE, nullptr);
  auto ptr = malloc(size + alignment);
  if (ptr == nullptr) {
    MS_LOG(ERROR) << "MallocAlignedData failed!";
    return nullptr;
  }
  uintptr_t aligned_ptr = (reinterpret_cast<uintptr_t>(ptr) + alignment - 1) & (~(alignment - 1));
  addr_map[aligned_ptr] = ptr;
  return reinterpret_cast<void *>(aligned_ptr);
}

void ConvolutionBaseCPUKernel::FreeAlignedData(void **ptr) {
  if (*ptr != nullptr && addr_map[reinterpret_cast<uintptr_t>(*ptr)] != nullptr) {
    free(addr_map[reinterpret_cast<uintptr_t>(*ptr)]);
    addr_map[reinterpret_cast<uintptr_t>(*ptr)] = nullptr;
    *ptr = nullptr;
  }
}

ConvolutionBaseCPUKernel::~ConvolutionBaseCPUKernel() {
  if (addr_map.find(reinterpret_cast<uintptr_t>(packed_weight_)) != addr_map.end()) {
    FreeAlignedData(reinterpret_cast<void **>(&packed_weight_));
  } else if (!op_parameter_->is_train_session_) {
    if (!is_sharing_pack_) {
      free(packed_weight_);
    } else {
      lite::PackWeightManager::GetInstance()->Free(packed_weight_);
    }
    packed_weight_ = nullptr;
  }
  if (addr_map.find(reinterpret_cast<uintptr_t>(bias_data_)) != addr_map.end()) {
    FreeAlignedData(reinterpret_cast<void **>(&bias_data_));
  } else if (bias_data_ != nullptr) {
    free(bias_data_);
    bias_data_ = nullptr;
  }
}

void ConvolutionBaseCPUKernel::FreeQuantParam() {
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
  if (conv_quant_arg_->input_quant_args_ != nullptr) {
    free(conv_quant_arg_->input_quant_args_);
    conv_quant_arg_->input_quant_args_ = nullptr;
  }
  if (conv_quant_arg_->filter_quant_args_ != nullptr) {
    free(conv_quant_arg_->filter_quant_args_);
    conv_quant_arg_->filter_quant_args_ = nullptr;
  }
  if (conv_quant_arg_->output_quant_args_ != nullptr) {
    free(conv_quant_arg_->output_quant_args_);
    conv_quant_arg_->output_quant_args_ = nullptr;
  }
}

int ConvolutionBaseCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), kBiasIndex);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto input = this->in_tensors_.front();
  auto output = this->out_tensors_.front();
  CHECK_NULL_RETURN(input);
  CHECK_NULL_RETURN(in_tensors_[1]);
  CHECK_NULL_RETURN(output);
  CHECK_NULL_RETURN(conv_param_);
  MS_CHECK_TRUE_MSG(input->shape().size() == C4NUM, RET_ERROR, "Conv-like: input-shape should be 4D.");
  MS_CHECK_TRUE_MSG(in_tensors_[1]->shape().size() == C4NUM, RET_ERROR, "Conv-like: weight-shape only support 4D.");
  MS_CHECK_TRUE_MSG(output->shape().size() == C4NUM, RET_ERROR, "Conv-like: out-shape should be 4D.");
  conv_param_->input_batch_ = input->Batch();
  conv_param_->input_h_ = input->Height();
  conv_param_->input_w_ = input->Width();
  conv_param_->input_channel_ = input->Channel();
  conv_param_->output_batch_ = output->Batch();
  conv_param_->output_h_ = output->Height();
  conv_param_->output_w_ = output->Width();
  conv_param_->output_channel_ = output->Channel();
  conv_param_->thread_num_ = op_parameter_->thread_num_;
  return RET_OK;
}

bool ConvolutionBaseCPUKernel::CheckParamsValid() const {
  auto weight = this->in_tensors_.at(kWeightIndex);
  MS_CHECK_GT(conv_param_->group_, 0, false);
  MS_CHECK_GE(conv_param_->pad_u_, 0, false);
  MS_CHECK_GE(conv_param_->pad_d_, 0, false);
  MS_CHECK_GE(conv_param_->pad_l_, 0, false);
  MS_CHECK_GE(conv_param_->pad_r_, 0, false);
  MS_CHECK_GE(conv_param_->output_padding_h_, 0, false);
  MS_CHECK_GE(conv_param_->output_padding_w_, 0, false);
  MS_CHECK_GT(conv_param_->dilation_h_, 0, false);
  MS_CHECK_GT(conv_param_->dilation_w_, 0, false);
  MS_CHECK_GT(conv_param_->stride_h_, 0, false);
  MS_CHECK_GT(conv_param_->stride_w_, 0, false);
  MS_CHECK_TRUE_MSG(conv_param_->kernel_h_ == weight->Height(), false, "Invalid kernel height in conv params.");
  MS_CHECK_TRUE_MSG(conv_param_->kernel_w_ == weight->Width(), false, "Invalid kernel Width in conv params.");
  if (conv_param_->group_ > conv_param_->input_channel_) {
    conv_param_->group_ = conv_param_->input_channel_;
  }
  return true;
}

int ConvolutionBaseCPUKernel::InitConvWeightBias() {
  if (op_parameter_->is_train_session_) {
    UpdateOriginWeightAndBias();
  }
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  CHECK_NULL_RETURN(weight_tensor);
  auto shape = weight_tensor->shape();
  if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
    MS_LOG(WARNING) << "The shape of weight tensor is not ready, the weight and bias would be inited in runtime.";
    return RET_OK;
  }
  if (MallocWeightBiasData() != RET_OK) {
    MS_LOG(ERROR) << "Malloc data for bias and weight failed.";
    return RET_ERROR;
  }

  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_.at(kBiasIndex);
    CHECK_NULL_RETURN(bias_tensor);
    MS_CHECK_FALSE(bias_tensor->Size() == 0, RET_ERROR);
    if (origin_bias_ == nullptr) {
      MS_LOG(ERROR) << "Convolution op " << this->name() << " bias data is nullptr.";
      return RET_ERROR;
    }
    memcpy(bias_data_, origin_bias_, bias_tensor->Size());
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  if (!op_parameter_->is_train_session_) {
    if (weight_is_packed_) {
      MS_LOG(DEBUG) << "not do weight pack.";
      return RET_OK;
    }
    if (origin_weight_ != nullptr) {
      PackWeight();
    } else {
      is_repack_ = true;
      MS_LOG(WARNING) << "The weight is nullptr, will pack in runtime.";
    }
  }
  return RET_OK;
}

int ConvolutionBaseCPUKernel::RepackWeight() {
  if (origin_weight_ == nullptr && in_tensors_.at(kWeightIndex)->data() == nullptr) {
    MS_LOG(ERROR) << "Convolution op " << this->name() << " weight data is nullptr.";
    return RET_ERROR;
  }
  origin_weight_ = origin_weight_ != nullptr ? origin_weight_ : in_tensors_.at(kWeightIndex)->data();
  if (packed_weight_ == nullptr && InitConvWeightBias() != RET_OK) {
    MS_LOG(ERROR) << "Malloc data for bias and weight failed.";
    return RET_ERROR;
  }
  if (IsRepack() || (op_parameter_->is_train_session_)) {
    if (op_parameter_->is_train_session_) {
      packed_weight_ = reinterpret_cast<float *>(workspace());
      memset(packed_weight_, 0, workspace_size());
    } else {
      is_repack_ = false;
    }
    PackWeight();
  }
  return RET_OK;
}

int ConvolutionBaseCPUKernel::CheckResizeValid() {
  // ===============check in channel================= //
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  CHECK_NULL_RETURN(filter_tensor);
  auto filter_in_channel = filter_tensor->Channel();
  int resize_in_channel = in_tensors_.at(kInputIndex)->Channel();
  if (filter_in_channel != resize_in_channel) {
    MS_LOG(ERROR) << "Channel of resized input should be equal to in channel of filter.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionBaseCPUKernel::CheckDeconvResizeValid() {
  // ===============check in channel================= //
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  CHECK_NULL_RETURN(filter_tensor);
  auto filter_out_channel = filter_tensor->Batch();
  int resize_out_channel = in_tensors_.at(kInputIndex)->Channel();
  if (filter_out_channel != resize_out_channel) {
    MS_LOG(ERROR) << "Channel of resized input should be equal to in channel of filter.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionBaseCPUKernel::SetIfPerChannel() {
  if (in_tensors_.size() < kInputSize1) {
    MS_LOG(ERROR) << "filter tensor not exist.";
    return RET_ERROR;
  }
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  if (!filter_tensor->IsConst()) {
    MS_LOG(WARNING) << "filter tensor is not const.";
    return RET_OK;
  }
  auto input_channel = filter_tensor->Channel();
  auto output_channel = filter_tensor->Batch();
  if (this->op_parameter_->type_ == schema::PrimitiveType_Conv2dTransposeFusion) {
    auto parameter = reinterpret_cast<const ConvParameter *>(this->op_parameter_);
    if (parameter->input_channel_ != parameter->group_ || parameter->output_channel_ != parameter->group_) {
      input_channel = filter_tensor->Batch();
      output_channel = filter_tensor->Channel();
    }
  }

  uint8_t per_channel = 0b0;
  if (conv_quant_arg_->input_arg_num_ != kPerTensor) {
    if (static_cast<int>(conv_quant_arg_->input_arg_num_) != input_channel) {
      MS_LOG(ERROR) << "input per channel quant param length is not equal to input channel.";
      return RET_ERROR;
    }
    per_channel = per_channel | INPUT_PER_CHANNEL;
  }

  if (conv_quant_arg_->filter_arg_num_ != kPerTensor) {
    if (static_cast<int>(conv_quant_arg_->filter_arg_num_) != output_channel) {
      MS_LOG(ERROR) << "weight per channel quant param length is not equal to filter num.";
      return RET_ERROR;
    }
    per_channel = per_channel | FILTER_PER_CHANNEL;
  }

  if (conv_quant_arg_->output_arg_num_ != kPerTensor) {
    if (static_cast<int>(conv_quant_arg_->output_arg_num_) != output_channel) {
      MS_LOG(ERROR) << "output per channel quant param length is not equal to output channel.";
      return RET_ERROR;
    }
    per_channel = per_channel | OUTPUT_PER_CHANNEL;
  }
  conv_quant_arg_->per_channel_ = per_channel;
  return RET_OK;
}

int ConvolutionBaseCPUKernel::MallocQuantParam() {
  conv_quant_arg_ = &(conv_param_->conv_quant_arg_);
  auto input_tensor = in_tensors_.at(kInputIndex);
  CHECK_NULL_RETURN(input_tensor);
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  CHECK_NULL_RETURN(weight_tensor);
  auto output_tensor = out_tensors_.at(kOutputIndex);
  CHECK_NULL_RETURN(output_tensor);
  size_t input_arg_num = input_tensor->quant_params().size();
  size_t filter_arg_num = weight_tensor->quant_params().size();
  size_t output_arg_num = output_tensor->quant_params().size();
  conv_quant_arg_->input_arg_num_ = input_arg_num;
  conv_quant_arg_->filter_arg_num_ = filter_arg_num;
  conv_quant_arg_->output_arg_num_ = output_arg_num;

  MS_CHECK_TRUE_RET(input_arg_num > 0 && input_arg_num <= MAX_MALLOC_SIZE, RET_ERROR);
  conv_quant_arg_->input_quant_args_ = reinterpret_cast<QuantArg *>(malloc(input_arg_num * sizeof(QuantArg)));
  if (conv_quant_arg_->input_quant_args_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_quant_args_ failed.";
    return RET_MEMORY_FAILED;
  }
  MS_CHECK_TRUE_RET(filter_arg_num > 0 && filter_arg_num <= MAX_MALLOC_SIZE, RET_ERROR);
  conv_quant_arg_->filter_quant_args_ = reinterpret_cast<QuantArg *>(malloc(filter_arg_num * sizeof(QuantArg)));
  if (conv_quant_arg_->filter_quant_args_ == nullptr) {
    MS_LOG(ERROR) << "malloc filter_quant_args_ failed.";
    return RET_MEMORY_FAILED;
  }
  MS_CHECK_TRUE_RET(output_arg_num > 0 && output_arg_num <= MAX_MALLOC_SIZE, RET_ERROR);
  conv_quant_arg_->output_quant_args_ = reinterpret_cast<QuantArg *>(malloc(output_arg_num * sizeof(QuantArg)));
  if (conv_quant_arg_->output_quant_args_ == nullptr) {
    MS_LOG(ERROR) << "malloc output_quant_args_ failed.";
    return RET_MEMORY_FAILED;
  }
  return RET_OK;
}

int ConvolutionBaseCPUKernel::SetInputTensorQuantParam() {
  auto input_tensor = in_tensors_.at(kInputIndex);
  CHECK_NULL_RETURN(input_tensor);
  auto in_arg_num = conv_quant_arg_->input_arg_num_;
  if (in_arg_num == kPerTensor) {
    auto input_quant_arg = input_tensor->quant_params().front();
    conv_quant_arg_->input_quant_args_[0].zp_ = input_quant_arg.zeroPoint;
    conv_quant_arg_->input_quant_args_[0].scale_ = input_quant_arg.scale;
  } else {
    // per channel
    MS_LOG(ERROR) << "Not Support Per Channel for input now.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionBaseCPUKernel::SetFilterTensorQuantParam() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  CHECK_NULL_RETURN(weight_tensor);
  auto weight_arg_num = conv_quant_arg_->filter_arg_num_;
  if (weight_arg_num == kPerTensor) {
    auto weight_quant_arg = weight_tensor->quant_params().front();
    conv_quant_arg_->filter_quant_args_[0].zp_ = weight_quant_arg.zeroPoint;
    conv_quant_arg_->filter_quant_args_[0].scale_ = weight_quant_arg.scale;
  } else {
    auto weight_quant_arg = weight_tensor->quant_params();
    for (size_t i = 0; i < weight_arg_num; ++i) {
      conv_quant_arg_->filter_quant_args_[i].zp_ = weight_quant_arg[i].zeroPoint;
      conv_quant_arg_->filter_quant_args_[i].scale_ = weight_quant_arg[i].scale;
    }
  }
  return RET_OK;
}

int ConvolutionBaseCPUKernel::SetOutputTensorQuantParam() {
  auto output_tensor = out_tensors_.at(kOutputIndex);
  CHECK_NULL_RETURN(output_tensor);
  auto out_arg_num = conv_quant_arg_->output_arg_num_;
  if (out_arg_num == kPerTensor) {
    auto output_quant_arg = output_tensor->quant_params().front();
    conv_quant_arg_->output_quant_args_[0].zp_ = output_quant_arg.zeroPoint;
    conv_quant_arg_->output_quant_args_[0].scale_ = output_quant_arg.scale;
  } else {
    MS_LOG(ERROR) << "Not Support Per Channel for input now.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionBaseCPUKernel::SetQuantMultiplier() {
  // now only support weight tensor is per channel, others are per tensor.
  int weight_arg_num = kPerTensor;
  if (conv_quant_arg_->per_channel_ & FILTER_PER_CHANNEL) {
    weight_arg_num = static_cast<int>(conv_quant_arg_->filter_arg_num_);
  }
  conv_quant_arg_->real_multiplier_ = reinterpret_cast<double *>(malloc(weight_arg_num * sizeof(double)));
  if (conv_quant_arg_->real_multiplier_ == nullptr) {
    MS_LOG(ERROR) << "malloc conv_quant_arg_->real_multiplier_ failed.";
    return RET_MEMORY_FAILED;
  }
  conv_quant_arg_->left_shift_ = reinterpret_cast<int32_t *>(malloc(weight_arg_num * sizeof(int32_t)));
  if (conv_quant_arg_->left_shift_ == nullptr) {
    MS_LOG(ERROR) << "malloc conv_quant_arg_->left_shift_ failed.";
    return RET_MEMORY_FAILED;
  }
  conv_quant_arg_->right_shift_ = reinterpret_cast<int32_t *>(malloc(weight_arg_num * sizeof(int32_t)));
  if (conv_quant_arg_->right_shift_ == nullptr) {
    MS_LOG(ERROR) << "malloc conv_quant_arg_->right_shift_ failed.";
    return RET_MEMORY_FAILED;
  }
  conv_quant_arg_->quant_multiplier_ = reinterpret_cast<int32_t *>(malloc(weight_arg_num * sizeof(int32_t)));
  if (conv_quant_arg_->quant_multiplier_ == nullptr) {
    MS_LOG(ERROR) << "malloc conv_quant_arg_->quant_multiplier_ failed.";
    return RET_MEMORY_FAILED;
  }
  conv_quant_arg_->out_act_min_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  if (conv_quant_arg_->out_act_min_ == nullptr) {
    MS_LOG(ERROR) << "malloc conv_quant_arg_->out_act_min_ failed.";
    return RET_MEMORY_FAILED;
  }
  conv_quant_arg_->out_act_max_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t)));
  if (conv_quant_arg_->out_act_max_ == nullptr) {
    MS_LOG(ERROR) << "malloc conv_quant_arg_->out_act_max_ failed.";
    return RET_MEMORY_FAILED;
  }

  for (int i = 0; i < weight_arg_num; ++i) {
    const double in_scale =
      static_cast<double>(conv_quant_arg_->input_quant_args_[0].scale_ * conv_quant_arg_->filter_quant_args_[i].scale_);
    double real_multiplier = in_scale / static_cast<double>(conv_quant_arg_->output_quant_args_[0].scale_);
    conv_quant_arg_->real_multiplier_[i] = real_multiplier;
    if (conv_quant_arg_->quant_multiplier_mode_ == Method_SinglePrecision) {
      QuantizeRoundParameterWithSinglePrecision(real_multiplier, &conv_quant_arg_->quant_multiplier_[i],
                                                &conv_quant_arg_->left_shift_[i], &conv_quant_arg_->right_shift_[i]);
    } else if (conv_quant_arg_->quant_multiplier_mode_ == Method_DoublePrecision) {
      QuantizeRoundParameterWithDoublePrecision(real_multiplier, &conv_quant_arg_->quant_multiplier_[i],
                                                &conv_quant_arg_->left_shift_[i], &conv_quant_arg_->right_shift_[i]);
    }
  }
  return RET_OK;
}

void ConvolutionBaseCPUKernel::SetRoundingAndMultipilerMode() {
  if (!in_tensors_.at(kInputIndex)->quant_params().empty()) {
    auto input_quant_arg = in_tensors_.at(kInputIndex)->quant_params().front();
    int round_type = input_quant_arg.roundType;
    switch (round_type) {
      case 1:
        conv_quant_arg_->round_mode_ = Rounding_Away_from_zero;
        break;
      case 2:
        conv_quant_arg_->round_mode_ = Rounding_Up;
        break;
      default:
        conv_quant_arg_->round_mode_ = Rounding_No;
    }
    int cal_multiplier_type = input_quant_arg.multiplier;
    switch (cal_multiplier_type) {
      case 0:
        conv_quant_arg_->quant_multiplier_mode_ = Method_SinglePrecision;
        break;
      case 1:
        conv_quant_arg_->quant_multiplier_mode_ = Method_DoublePrecision;
        break;
      default:
        conv_quant_arg_->quant_multiplier_mode_ = Method_No;
    }
  }
}

int ConvolutionBaseCPUKernel::SetQuantParam() {
  auto ret = MallocQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc quant param failed.";
    return ret;
  }
  ret = SetInputTensorQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set Input Tensor Quant Param Failed.";
    return ret;
  }
  ret = SetFilterTensorQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set Filter Tensor Quant Param Failed.";
    return ret;
  }
  ret = SetOutputTensorQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set Output Tensor Quant Param Failed.";
    return ret;
  }
  ret = SetIfPerChannel();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set if per tensor channel failed.";
    return ret;
  }
  SetRoundingAndMultipilerMode();
  ret = SetQuantMultiplier();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set Quant Multiplier Failed.";
    return ret;
  }
  bool relu = conv_param_->act_type_ == ActType_Relu;
  bool relu6 = conv_param_->act_type_ == ActType_Relu6;
  CalculateActivationRangeQuantized(relu, relu6, conv_param_->conv_quant_arg_.output_quant_args_[0].zp_,
                                    conv_param_->conv_quant_arg_.output_quant_args_[0].scale_,
                                    &conv_param_->conv_quant_arg_.out_act_min_[0],
                                    &conv_param_->conv_quant_arg_.out_act_max_[0]);
  return RET_OK;
}

void ConvolutionBaseCPUKernel::UpdateOriginWeightAndBias() {
  if (in_tensors_.at(kWeightIndex)->data() != nullptr) {
    origin_weight_ = in_tensors_.at(kWeightIndex)->data();
  }
  if (in_tensors_.size() == kInputSize2 && in_tensors_.at(kBiasIndex) != nullptr &&
      in_tensors_.at(kBiasIndex)->data() != nullptr) {
    origin_bias_ = in_tensors_.at(kBiasIndex)->data();
  }
}

bool ConvolutionBaseCPUKernel::CheckInputsValid() const {
  // the data type of input and weight must be the same, while the bias data type of int8 convolution is int32.
  MS_CHECK_TRUE_RET(in_tensors_.size() >= kInputSize1, false);
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  MS_CHECK_TRUE_RET(input_tensor != nullptr && weight_tensor != nullptr, false);
  MS_CHECK_TRUE_RET(input_tensor->data() != nullptr, false);
  return input_tensor->data_type() == weight_tensor->data_type();
}

int ConvolutionBaseCPUKernel::CheckAndGetWeightParam(int32_t *batch, int32_t *height, int32_t *width) const {
  CHECK_NULL_RETURN(batch);
  CHECK_NULL_RETURN(height);
  CHECK_NULL_RETURN(width);
  if (kWeightIndex >= in_tensors_.size()) {
    MS_LOG(ERROR) << "Input tensor size " << in_tensors_.size() << " invalid, expected weight index: " << kWeightIndex;
    return RET_ERROR;
  }
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  CHECK_NULL_RETURN(weight_tensor);
  auto _batch = weight_tensor->Batch();
  if (_batch <= 0) {
    MS_LOG(ERROR) << "get batch from weight_tensor failed, batch: " << _batch;
    return RET_ERROR;
  }
  *batch = _batch;
  auto _height = weight_tensor->Height();
  if (_height <= 0) {
    MS_LOG(ERROR) << "get height from weight_tensor failed, height: " << _height;
    return RET_ERROR;
  }
  *height = _height;
  auto _width = weight_tensor->Width();
  if (_width <= 0) {
    MS_LOG(ERROR) << "get width from weight_tensor failed, width: " << _width;
    return RET_ERROR;
  }
  *width = _width;
  if (INT32_MAX / _batch < _height || INT32_MAX / (_batch * _height) < _width) {
    MS_LOG(ERROR) << "Element number of tensor should be smaller than INT32_MAX, batch: " << _batch
                  << ", height: " << _height << ", width: " << _width;
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
