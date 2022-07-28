/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/nnacl/fp32_grad/adam_coder.h"
#include "nnacl/fp32_grad/optimizer.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_Adam;

namespace mindspore::lite::micro::nnacl {
namespace {
constexpr int kWeightIdx = 0;
constexpr int kMomentVector1stIdx = 1;
constexpr int kMomentVector2stIdx = 2;
constexpr int kBeta1PowerIdx = 3;
constexpr int kBeta2PowerIdx = 4;
constexpr int kLearningRateIdx = 5;
constexpr int kBeta1Idx = 6;
constexpr int kBeta2Idx = 7;
constexpr int kEpsilonIdx = 8;
constexpr int kGradientIdx = 9;
}  // namespace
int AdamCoder::DoCode(CoderContext *const context) {
  MS_CHECK_TRUE(input_tensors_.size() >= DIMENSION_10D, "inputs size is less than 10");
  auto weight = input_tensors_.at(kWeightIdx);
  auto m = input_tensors_.at(kMomentVector1stIdx);
  auto v = input_tensors_.at(kMomentVector2stIdx);
  auto beta1_power = input_tensors_.at(kBeta1PowerIdx);
  auto beta2_power = input_tensors_.at(kBeta2PowerIdx);
  auto learning_rate = reinterpret_cast<float *>(
    input_tensors_.at(kLearningRateIdx)->MutableData())[0];  // use model origin lr, unsupported to config
  auto beta1 = reinterpret_cast<float *>(input_tensors_.at(kBeta1Idx)->MutableData())[0];
  auto beta2 = reinterpret_cast<float *>(input_tensors_.at(kBeta2Idx)->MutableData())[0];
  auto eps = reinterpret_cast<float *>(input_tensors_.at(kEpsilonIdx)->MutableData())[0];
  auto gradient = input_tensors_.at(kGradientIdx);
  int length = input_tensors_.at(kWeightIdx)->ElementsNum();

  // attribute
  auto *adam_param = reinterpret_cast<AdamParameter *>(parameter_);
  Collect(context,
          {
            "nnacl/fp32/adam_fp32.h",
          },
          {
            "adam_fp32.c",
          });
  NNaclFp32Serializer code;
  code.CodeFunction("DoAdam", m, v, gradient, weight, beta1, beta2, beta1_power, beta2_power, eps, learning_rate,
                    adam_param->use_nesterov_, 0, length);
  context->AppendCode(code.str());
  return lite::RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Adam, CPUOpCoderCreator<AdamCoder>)
}  // namespace mindspore::lite::micro::nnacl
