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

#include <sstream>
#include <string>
#include "nnacl/pooling_parameter.h"
#include "nnacl/slice_parameter.h"
#include "nnacl/softmax_parameter.h"
#include "nnacl/int8/add_int8.h"
#include "nnacl/int8/quantize.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_stream_utils.h"

namespace mindspore::lite::micro {

std::ostream &operator<<(std::ostream &code, const ::QuantArg &quant_arg) {
  code << "{" << static_cast<float>(quant_arg.scale_) << ", " << quant_arg.zp_ << "}";
  return code;
}

std::ostream &operator<<(std::ostream &code, const OpParameter &parameter) {
  code << "{ \"\""
       << ", " << std::boolalpha << parameter.infer_flag_ << ", " << parameter.type_ << ", " << gThreadNum << ", "
       << parameter.quant_type_ << "}";
  return code;
}

std::ostream &operator<<(std::ostream &code, const AddQuantQrgs &args) {
  code << "{" << args.zp_ << ", " << args.left_shift_ << ", " << args.right_shift_ << ", " << args.multiplier_ << "}";
  return code;
}

std::ostream &operator<<(std::ostream &code, const SliceQuantArg &arg) {
  code << "{" << arg.in_args_ << ", " << arg.out_args_ << ", " << arg.output_activation_min_ << ", "
       << arg.output_activation_max_ << "}";
  return code;
}

std::ostream &operator<<(std::ostream &code, PoolMode pool_mode) {
  code << "(PoolMode)"
       << "(" << static_cast<int>(pool_mode) << ")";
  return code;
}

std::ostream &operator<<(std::ostream &code, RoundMode round_mode) {
  code << "(RoundMode)"
       << "(" << static_cast<int>(round_mode) << ")";
  return code;
}

std::ostream &operator<<(std::ostream &code, RoundingMode rounding_mode) {
  code << "(RoundingMode)"
       << "(" << static_cast<int>(rounding_mode) << ")";
  return code;
}

std::ostream &operator<<(std::ostream &code, PadMode pad_mode) {
  code << "(PadMode)"
       << "(" << static_cast<int>(pad_mode) << ")";
  return code;
}

std::ostream &operator<<(std::ostream &code, ActType act_type) {
  code << "(ActType)"
       << "(" << static_cast<int>(act_type) << ")";
  return code;
}

std::ostream &operator<<(std::ostream &code, DataOrder data_order) {
  if (data_order == RowMajor) {
    code << "RowMajor";
  } else {
    code << "ColMajor";
  }
  return code;
}

}  // namespace mindspore::lite::micro
