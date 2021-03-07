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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_SERIALIZERS_NNACL_STREAM_UTILS_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_SERIALIZERS_NNACL_STREAM_UTILS_H_
#include <sstream>
#include <string>
#include "nnacl/pooling_parameter.h"
#include "nnacl/slice_parameter.h"
#include "nnacl/softmax_parameter.h"
#include "nnacl/int8/add_int8.h"
#include "nnacl/int8/quantize.h"

namespace mindspore::lite::micro {

std::ostream &operator<<(std::ostream &code, const ::QuantArg &quant_arg);

std::ostream &operator<<(std::ostream &code, const OpParameter &tile);

std::ostream &operator<<(std::ostream &code, const AddQuantQrgs &args);

std::ostream &operator<<(std::ostream &code, const SliceQuantArg &arg);

std::ostream &operator<<(std::ostream &code, PoolMode pool_mode);

std::ostream &operator<<(std::ostream &code, RoundMode round_mode);

std::ostream &operator<<(std::ostream &code, RoundingMode rounding_mode);

std::ostream &operator<<(std::ostream &code, PadMode pad_mode);

std::ostream &operator<<(std::ostream &code, ActType act_type);

std::ostream &operator<<(std::ostream &code, DataOrder data_order);
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_SERIALIZERS_NNACL_STREAM_UTILS_H_
