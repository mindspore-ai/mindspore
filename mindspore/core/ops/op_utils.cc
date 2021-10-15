/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <string>
#include <set>
#include <vector>
#include <algorithm>
#include <memory>
#include "ops/op_utils.h"
#include "abstract/primitive_infer_map.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
std::vector<int64_t> CalBroadCastShape(std::vector<int64_t> x_shape, std::vector<int64_t> y_shape,
                                       const std::string &op_name, const std::string &op_x_name,
                                       const std::string &op_y_name) {
  if (x_shape == y_shape) {
    return x_shape;
  }
  auto x_length = static_cast<int64_t>(x_shape.size());
  auto y_length = static_cast<int64_t>(y_shape.size());
  auto length = x_length < y_length ? x_length : y_length;
  std::vector<int64_t> broadcast_shape;
  if (x_length == length) {
    (void)std::copy(y_shape.begin(), y_shape.end() - length, std::back_inserter(broadcast_shape));
  } else {
    (void)std::copy(x_shape.begin(), x_shape.end() - length, std::back_inserter(broadcast_shape));
  }
  for (int64_t i = -length; i < 0; i++) {
    if (x_shape[LongToSize(x_length + i)] == 1) {
      broadcast_shape.push_back(y_shape[LongToSize(y_length + i)]);
    } else if (y_shape[LongToSize(y_length + i)] == 1) {
      broadcast_shape.push_back(x_shape[LongToSize(x_length + i)]);
    } else if (x_shape[LongToSize(x_length + i)] == y_shape[LongToSize(y_length + i)]) {
      broadcast_shape.push_back(x_shape[LongToSize(x_length + i)]);
    } else {
      MS_EXCEPTION(ValueError) << "For op " << op_name << ", the two input '" << op_x_name << "' and '" << op_y_name
                               << "' can not broadcast";
    }
  }
  return broadcast_shape;
}
abstract::ShapePtr BroadCastInferShape(const std::string &op_name, const std::vector<AbstractBasePtr> &input_args) {
  MS_LOG(INFO) << "Do infer shape for op " << op_name;
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack());
  auto y_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShapeTrack());
  auto x_shape = x_shape_map[kShape];
  auto y_shape = y_shape_map[kShape];
  auto x_min_shape = x_shape_map[kMinShape];
  auto x_max_shape = x_shape_map[kMaxShape];
  auto y_min_shape = y_shape_map[kMinShape];
  auto y_max_shape = y_shape_map[kMaxShape];
  if (x_shape == y_shape) {
    return std::make_shared<abstract::Shape>(x_shape, x_min_shape, x_max_shape);
  }
  auto broadcast_shape = CalBroadCastShape(x_shape, y_shape, op_name);
  auto min_broadcast_shape = CalBroadCastShape(x_min_shape, y_min_shape, op_name);
  auto max_broadcast_shape = CalBroadCastShape(x_max_shape, y_max_shape, op_name);
  return std::make_shared<abstract::Shape>(broadcast_shape, min_broadcast_shape, max_broadcast_shape);
}
}  // namespace ops
}  // namespace mindspore
