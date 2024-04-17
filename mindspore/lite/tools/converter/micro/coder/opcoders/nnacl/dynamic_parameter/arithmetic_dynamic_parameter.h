/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_DYNAMIC_PARAMETER_ARITHMETIC_DYNAMIC_PARAMETER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_DYNAMIC_PARAMETER_ARITHMETIC_DYNAMIC_PARAMETER_H_
#include <string>

typedef struct ArithmeticDynamicParameter {
  std::string in_shape0_;
  std::string in_elements_num0_;
  std::string in_shape1_;
  std::string in_elements_num1_;

  std::string out_shape_;
  std::string out_elements_num_;

  std::string in_strides0_;
  std::string in_strides1_;
  std::string out_strides_;

  std::string multiples0_;
  std::string multiples1_;
} ArithmeticDynamicParameter;

typedef struct BroadcastDynamicShapeInfo {
  std::string input_shape_;
  std::string output_shape_;
} BroadcastDynamicShapeInfo;

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_OPCODERS_NNACL_DYNAMIC_PARAMETER_ARITHMETIC_DYNAMIC_PARAMETER_H_
