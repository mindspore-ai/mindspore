/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_C_OPS_CONV_UTILS_H
#define MINDSPORE_CORE_C_OPS_CONV_UTILS_H
#include <string>
#include <set>
#include <vector>
#include <algorithm>
#include <memory>
#include "abstract/primitive_infer_map.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
constexpr auto kKernelSize = "kernel_size";
constexpr auto kStride = "stride";
constexpr auto kDilation = "dilation";
constexpr auto kPadMode = "pad_mode";
constexpr auto kPad = "pad";
constexpr auto kPads = "pads";
constexpr auto kMode = "mode";
constexpr auto kGroup = "group";
constexpr auto kOutputChannel = "output_channel";
constexpr auto kPadList = "pad_list";
constexpr auto kAxis = "axis";

const std::set<TypeId> common_valid_types = {
  kNumberTypeInt8,   kNumberTypeInt16,  kNumberTypeInt32,   kNumberTypeInt64,   kNumberTypeUInt8,  kNumberTypeUInt16,
  kNumberTypeUInt32, kNumberTypeUInt64, kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64};

abstract::ShapePtr BroadCastInferShape(const std::string &op_name, const std::vector<AbstractBasePtr> &input_args);
}  // namespace mindspore
#endif  // MINDSPORE_CORE_C_OPS_CONV_UTILS_H
