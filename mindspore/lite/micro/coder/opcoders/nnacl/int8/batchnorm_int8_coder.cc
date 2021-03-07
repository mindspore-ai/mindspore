/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/nnacl/int8/batchnorm_int8_coder.h"
#include <string>
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_BatchNorm;

namespace mindspore::lite::micro::nnacl {

int BatchNormInt8Coder::Prepare(CoderContext *const context) {
  std::vector<int> input_shapes = input_tensor_->shape();
  size_t n_dim = input_shapes.size();
  batchnorm_param_->channel_ = input_shapes[n_dim - 1];
  batchnorm_param_->units_ = 1;
  for (size_t i = 0; i < n_dim - 1; i++) {
    batchnorm_param_->units_ *= input_shapes[i];
  }
  batchnorm_param_->op_parameter_.thread_num_ =
    MSMIN(batchnorm_param_->op_parameter_.thread_num_, batchnorm_param_->channel_);
  if (target_ == kARM32M) {
    batchnorm_param_->unit_ = batchnorm_param_->units_;
  } else {
    batchnorm_param_->unit_ = UP_DIV(batchnorm_param_->units_, kMaxThreadNumSupported);
  }
  if (batchnorm_param_->fused_) {
    MS_CHECK_RET_CODE(InitFusedConstTensor(), "InitFusedConstTensor failed");
  } else {
    MS_CHECK_RET_CODE(InitConstTensor(), "InitConstTensor failed");
  }

  return RET_OK;
}
int BatchNormInt8Coder::DoCode(CoderContext *context) {
  std::vector<std::string> headers = {"nnacl/slice_parameter.h"};
  std::vector<std::string> cFiles = {"batchnorm_int8.c"};
  NNaclInt8Serializer code;

  code.CodeStruct("param", *batchnorm_param_);
  code.CodeFunction("BatchNormInt8", output_tensor_, input_tensor_, alpha_addr_, beta_addr_, kDefaultTaskId, "&param");

  Collect(context, headers, cFiles);
  context->AppendCode(code.str());

  return RET_OK;
}

int BatchNormInt8Coder::InitConstTensor() {
  MS_CHECK_TRUE(input_tensors_.size() >= kInputSize2, "input tensors number not match");
  Tensor *input = input_tensor_;
  Tensor *mean = input_tensors_.at(1);
  Tensor *variance = input_tensors_.at(2);
  Tensor *output = output_tensor_;

  auto mean_ptr = reinterpret_cast<int8_t *>(mean->MutableData());
  auto var_ptr = reinterpret_cast<int8_t *>(variance->MutableData());

  MS_CHECK_PTR(mean_ptr);
  MS_CHECK_PTR(var_ptr);

  alpha_addr_ = reinterpret_cast<float *>(
    allocator_->Malloc(kNumberTypeFloat, mean->ElementsNum() * sizeof(float), kOfflinePackWeight));
  MS_CHECK_PTR(alpha_addr_);
  beta_addr_ = reinterpret_cast<float *>(
    allocator_->Malloc(kNumberTypeFloat, variance->ElementsNum() * sizeof(float), kOfflinePackWeight));
  MS_CHECK_PTR(beta_addr_);
  // compute alpha, beta;
  auto eps = batchnorm_param_->epsilon_;
  int32_t zp_in = input->quant_params().at(0).zeroPoint;
  int32_t zp_mean = mean->quant_params().at(0).zeroPoint;
  int32_t zp_var = variance->quant_params().at(0).zeroPoint;
  int32_t zp_out = output->quant_params().at(0).zeroPoint;
  auto s_in = static_cast<float>(input->quant_params().at(0).scale);
  auto s_mean = static_cast<float>(mean->quant_params().at(0).scale);
  auto s_var = static_cast<float>(variance->quant_params().at(0).scale);
  auto s_out = static_cast<float>(output->quant_params().at(0).scale);

  for (int i = 0; i < batchnorm_param_->channel_; ++i) {
    float tmp = s_out * sqrt(eps + s_var * (var_ptr[i] - zp_var));
    float tmp_a = s_in / tmp;
    float tmp_b = zp_out - tmp_a * zp_in - (s_mean * (mean_ptr[i] - zp_mean)) / tmp;
    alpha_addr_[i] = tmp_a;
    beta_addr_[i] = tmp_b;
  }

  return RET_OK;
}

int BatchNormInt8Coder::InitFusedConstTensor() {
  MS_CHECK_TRUE(input_tensors_.size() >= 5, "input tensors number not match");
  Tensor *input = input_tensors_.at(0);
  Tensor *scale = input_tensors_.at(1);
  Tensor *offset = input_tensors_.at(2);
  Tensor *mean = input_tensors_.at(3);
  Tensor *variance = input_tensors_.at(4);
  Tensor *output = output_tensor_;

  auto scale_ptr = reinterpret_cast<int8_t *>(scale->MutableData());
  auto offset_ptr = reinterpret_cast<int8_t *>(offset->MutableData());
  auto mean_ptr = reinterpret_cast<int8_t *>(mean->MutableData());
  auto var_ptr = reinterpret_cast<int8_t *>(variance->MutableData());

  MS_CHECK_PTR(scale_ptr);
  MS_CHECK_PTR(offset_ptr);
  MS_CHECK_PTR(mean_ptr);
  MS_CHECK_PTR(var_ptr);

  alpha_addr_ = reinterpret_cast<float *>(
    allocator_->Malloc(kNumberTypeFloat, mean->ElementsNum() * sizeof(float), kOfflinePackWeight));
  MS_CHECK_PTR(alpha_addr_);
  beta_addr_ = reinterpret_cast<float *>(
    allocator_->Malloc(kNumberTypeFloat, variance->ElementsNum() * sizeof(float), kOfflinePackWeight));
  MS_CHECK_PTR(beta_addr_);
  // compute alpha, beta;
  float eps = batchnorm_param_->epsilon_;
  int32_t zp_in = input->quant_params().at(0).zeroPoint;
  int32_t zp_scale = scale->quant_params().at(0).zeroPoint;
  int32_t zp_offset = offset->quant_params().at(0).zeroPoint;
  int32_t zp_mean = mean->quant_params().at(0).zeroPoint;
  int32_t zp_var = variance->quant_params().at(0).zeroPoint;
  int32_t zp_out = output->quant_params().at(0).zeroPoint;
  auto s_in = static_cast<float>(input->quant_params().at(0).scale);
  auto s_scale = static_cast<float>(scale->quant_params().at(0).scale);
  auto s_offset = static_cast<float>(offset->quant_params().at(0).scale);
  auto s_mean = static_cast<float>(mean->quant_params().at(0).scale);
  auto s_var = static_cast<float>(variance->quant_params().at(0).scale);
  auto s_out = static_cast<float>(output->quant_params().at(0).scale);

  float mul_12 = s_in * s_scale;
  float mul_24 = s_scale * s_mean;
  float div_36 = s_offset / s_out;
  for (int i = 0; i < batchnorm_param_->channel_; ++i) {
    float tmp = s_out * sqrt(eps + s_var * (var_ptr[i] - zp_var));
    float tmp_a = (mul_12 * (scale_ptr[i] - zp_scale)) / tmp;
    float tmp_b = zp_out + div_36 * (offset_ptr[i] - zp_offset) - tmp_a * zp_in -
                  (mul_24 * (scale_ptr[i] - zp_scale) * (mean_ptr[i] - zp_mean)) / tmp;
    alpha_addr_[i] = tmp_a;
    beta_addr_[i] = tmp_b;
  }

  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_BatchNorm, CPUOpCoderCreator<BatchNormInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
