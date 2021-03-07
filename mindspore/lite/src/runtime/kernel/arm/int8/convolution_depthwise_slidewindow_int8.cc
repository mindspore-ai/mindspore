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

#include "src/runtime/kernel/arm/int8/convolution_depthwise_slidewindow_int8.h"
#include "include/errorcode.h"
#include "nnacl/int8/conv_depthwise_int8.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
ConvolutionDepthwiseSWInt8CPUKernel::~ConvolutionDepthwiseSWInt8CPUKernel() {
  if (sliding_ != nullptr) {
    delete sliding_;
    sliding_ = nullptr;
  }
  if (packed_weight_ != nullptr) {
    free(packed_weight_);
    packed_weight_ = nullptr;
  }
  FreeTmpQuant();
  FreeQuantParam();
}

int ConvolutionDepthwiseSWInt8CPUKernel::InitWeightBias() {
  // init weight, int8 -> int16
  // o, h, w, i -> o/8, h, w, i, 8; o == group, i == 1
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto origin_weight = reinterpret_cast<int8_t *>(weight_tensor->MutableData());
  int OC8 = UP_DIV(weight_tensor->Batch(), C8NUM);
  int pack_weight_size = C8NUM * OC8 * weight_tensor->Height() * weight_tensor->Width();
  packed_weight_ = reinterpret_cast<int16_t *>(malloc(pack_weight_size * sizeof(int16_t)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  PackDepthwiseInt8Weight(origin_weight, packed_weight_, weight_tensor->Height() * weight_tensor->Width(),
                          weight_tensor->Batch(), &(conv_param_->conv_quant_arg_));

  bias_data_ = reinterpret_cast<int32_t *>(malloc(C8NUM * OC8 * sizeof(int32_t)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, C8NUM * OC8 * sizeof(int32_t));
  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_.at(kBiasIndex);
    auto ori_bias = reinterpret_cast<int32_t *>(bias_tensor->MutableData());
    memcpy(bias_data_, ori_bias, bias_tensor->ElementsNum() * sizeof(int32_t));
  }

  conv_param_->thread_num_ = MSMIN(thread_count_, OC8);
  return RET_OK;
}

int ConvolutionDepthwiseSWInt8CPUKernel::InitPackedInputOutput() {
  if (conv_param_->input_channel_ % C8NUM != 0) {
    need_align_ = true;

    int pack_input_size = conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * C8NUM *
                          UP_DIV(conv_param_->input_channel_, C8NUM);
    packed_input_ = reinterpret_cast<int8_t *>(context_->allocator->Malloc(pack_input_size * sizeof(int8_t)));
    if (packed_input_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }

    int pack_output_size = conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * C8NUM *
                           UP_DIV(conv_param_->output_channel_, C8NUM);
    packed_output_ = reinterpret_cast<int8_t *>(context_->allocator->Malloc(pack_output_size * sizeof(int8_t)));
    if (packed_output_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void ConvolutionDepthwiseSWInt8CPUKernel::FreeTmpQuant() {
  if (input_scale_ != nullptr) {
    free(input_scale_);
    input_scale_ = nullptr;
  }
  if (input_zp_ != nullptr) {
    free(input_zp_);
    input_zp_ = nullptr;
  }
  if (weight_scale_ != nullptr) {
    free(weight_scale_);
    weight_scale_ = nullptr;
  }
  if (output_scale_ != nullptr) {
    free(output_scale_);
    output_scale_ = nullptr;
  }
  if (output_zp_ != nullptr) {
    free(output_zp_);
    output_zp_ = nullptr;
  }
}

int ConvolutionDepthwiseSWInt8CPUKernel::ReinitFreeBefore() {
  FreeTmpQuant();
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
  return RET_OK;
}

int ConvolutionDepthwiseSWInt8CPUKernel::ReinitQuantParam() {
  ReinitFreeBefore();  // remalloc quant param buffer

  auto input_tensor = in_tensors_.at(kInputIndex);
  auto channel = conv_param_->input_channel_;
  input_scale_ = reinterpret_cast<float *>(malloc(channel * sizeof(float)));
  MSLITE_CHECK_PTR(input_scale_);

  input_zp_ = reinterpret_cast<int8_t *>(malloc(channel * sizeof(int8_t)));
  MSLITE_CHECK_PTR(input_zp_);

  if (input_tensor->quant_params().size() == kPerTensor) {
    for (int i = 0; i < channel; i++) {
      auto input_quant_arg = input_tensor->quant_params().front();
      input_zp_[i] = input_quant_arg.zeroPoint;
      input_scale_[i] = input_quant_arg.scale;
    }
  } else {
    for (int i = 0; i < channel; i++) {
      auto input_quant_arg = input_tensor->quant_params()[i];
      input_zp_[i] = input_quant_arg.zeroPoint;
      input_scale_[i] = input_quant_arg.scale;
    }
  }

  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_scale_ = reinterpret_cast<float *>(malloc(channel * sizeof(float)));
  MSLITE_CHECK_PTR(output_scale_);

  output_zp_ = reinterpret_cast<int32_t *>(malloc(channel * sizeof(int32_t)));
  MSLITE_CHECK_PTR(output_zp_);

  if (output_tensor->quant_params().size() == kPerTensor) {
    for (int i = 0; i < channel; i++) {
      auto output_quant_arg = output_tensor->quant_params().front();
      output_zp_[i] = output_quant_arg.zeroPoint;
      output_scale_[i] = output_quant_arg.scale;
    }
  } else {
    for (int i = 0; i < channel; i++) {
      auto output_quant_arg = output_tensor->quant_params()[i];
      output_zp_[i] = output_quant_arg.zeroPoint;
      output_scale_[i] = output_quant_arg.scale;
    }
  }

  conv_quant_arg_->real_multiplier_ = reinterpret_cast<double *>(malloc(channel * sizeof(double)));
  MSLITE_CHECK_PTR(conv_quant_arg_->real_multiplier_);

  conv_quant_arg_->left_shift_ = reinterpret_cast<int32_t *>(malloc(channel * sizeof(int32_t)));
  MSLITE_CHECK_PTR(conv_quant_arg_->left_shift_);

  conv_quant_arg_->right_shift_ = reinterpret_cast<int32_t *>(malloc(channel * sizeof(int32_t)));
  MSLITE_CHECK_PTR(conv_quant_arg_->right_shift_);

  conv_quant_arg_->quant_multiplier_ = reinterpret_cast<int32_t *>(malloc(channel * sizeof(int32_t)));
  MSLITE_CHECK_PTR(conv_quant_arg_->quant_multiplier_);

  conv_quant_arg_->out_act_min_ = reinterpret_cast<int32_t *>(malloc(channel * sizeof(int32_t)));
  MSLITE_CHECK_PTR(conv_quant_arg_->out_act_min_);

  conv_quant_arg_->out_act_max_ = reinterpret_cast<int32_t *>(malloc(channel * sizeof(int32_t)));
  MSLITE_CHECK_PTR(conv_quant_arg_->out_act_max_);

  weight_scale_ = reinterpret_cast<float *>(malloc(channel * sizeof(float)));
  MSLITE_CHECK_PTR(weight_scale_);

  auto weight_tensor = in_tensors_.at(kWeightIndex);
  if (weight_tensor->quant_params().size() == kPerTensor) {
    for (int i = 0; i < channel; i++) {
      auto weight_quant_arg = weight_tensor->quant_params().front();
      weight_scale_[i] = weight_quant_arg.scale;
    }
  } else {
    for (int i = 0; i < channel; i++) {
      auto weight_quant_arg = weight_tensor->quant_params()[i];
      weight_scale_[i] = weight_quant_arg.scale;
    }
  }

  for (int i = 0; i < channel; ++i) {
    const double in_scale = static_cast<double>(input_scale_[i] * weight_scale_[i]);
    double real_multiplier = in_scale / static_cast<double>(output_scale_[i]);
    conv_quant_arg_->real_multiplier_[i] = real_multiplier;
    QuantizeRoundParameterWithDoublePrecision(real_multiplier, &conv_quant_arg_->quant_multiplier_[i],
                                              &conv_quant_arg_->left_shift_[i], &conv_quant_arg_->right_shift_[i]);
  }

  // now only consider per tensor for output
  bool relu = conv_param_->act_type_ == ActType_Relu;
  bool relu6 = conv_param_->act_type_ == ActType_Relu6;
  for (int i = 0; i < channel; ++i) {
    CalculateActivationRangeQuantized(relu, relu6, output_zp_[i], output_scale_[i],
                                      &conv_param_->conv_quant_arg_.out_act_min_[i],
                                      &conv_param_->conv_quant_arg_.out_act_max_[i]);
  }
  return RET_OK;
}

int ConvolutionDepthwiseSWInt8CPUKernel::Init() {
  sliding_ = new (std::nothrow) SlidingWindowParam;
  if (sliding_ == nullptr) {
    MS_LOG(ERROR) << "new sliding window param.";
    return RET_ERROR;
  }
  auto ret = ConvolutionBaseCPUKernel::SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set quant param failed.";
    return ret;
  }
  ret = ReinitQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "reinit quant param failed.";
    return ret;
  }
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

int ConvolutionDepthwiseSWInt8CPUKernel::ReSize() {
  ConvolutionBaseCPUKernel::Init();
  InitSlidingParamConvDw(sliding_, conv_param_, C8NUM);
  return RET_OK;
}

int ConvolutionDepthwiseSWInt8CPUKernel::Execute(int task_id) {
  ConvDwInt8SW(packed_output_, packed_input_, packed_weight_, reinterpret_cast<int32_t *>(bias_data_), input_zp_,
               output_zp_, conv_param_, sliding_, task_id);
  return RET_OK;
}

int ConvDwSWInt8Run(void *cdata, int task_id) {
  auto conv_dw_int8 = reinterpret_cast<ConvolutionDepthwiseSWInt8CPUKernel *>(cdata);
  auto ret = conv_dw_int8->Execute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionDepthwiseSWInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionDepthwiseSWInt8CPUKernel::Run() {
  auto ret = InitPackedInputOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Depthwise int8 ReSize error!";
    FreePackedInputOutput();
    return ret;
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  auto input_addr = reinterpret_cast<int8_t *>(input_tensor->MutableData());
  if (need_align_) {
    PackNHWCToNHWC8Int8(input_addr, packed_input_, conv_param_->output_batch_,
                        conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_);
  } else {
    packed_input_ = input_addr;
  }

  auto output_addr = reinterpret_cast<int8_t *>(out_tensors_.at(kOutputIndex)->MutableData());
  if (!need_align_) {
    packed_output_ = output_addr;
  }

  ret = ParallelLaunch(this->context_->thread_pool_, ConvDwSWInt8Run, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvDwSWInt8Run error: error_code[" << ret << "]";
  }

  if (need_align_) {
    PackNHWC8ToNHWCInt8(packed_output_, output_addr, conv_param_->output_batch_,
                        conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_);
  }
  FreePackedInputOutput();
  return ret;
}

void ConvolutionDepthwiseSWInt8CPUKernel::FreePackedInputOutput() {
  if (need_align_) {
    context_->allocator->Free(packed_input_);
    context_->allocator->Free(packed_output_);
    packed_input_ = nullptr;
    packed_output_ = nullptr;
  }
}
}  // namespace mindspore::kernel
