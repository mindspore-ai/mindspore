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
abstract::ShapePtr BroadCastInferShape(const std::string &op_name, const std::vector<AbstractBasePtr> &input_args) {
  MS_LOG(INFO) << "Do infer shape for op " << op_name;
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->GetShapeTrack(), op_name);
  auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShape("y_shape", input_args[1]->GetShapeTrack(), op_name);
  if (x_shape == y_shape) {
    return std::make_shared<abstract::Shape>(x_shape);
  }

  auto x_length = x_shape.size();
  auto y_length = y_shape.size();
  auto length = x_length < y_length ? x_length : y_length;
  std::vector<int64_t> broadcast_shape;
  if (x_length == length) {
    std::copy(y_shape.begin(), y_shape.end() - length, std::back_inserter(broadcast_shape));
  } else {
    std::copy(x_shape.begin(), x_shape.end() - length, std::back_inserter(broadcast_shape));
  }
  for (int64_t i = -length; i < 0; i++) {
    if (x_shape[x_length + i] == 1) {
      broadcast_shape.push_back(y_shape[y_length + i]);
    } else if (y_shape[y_length + i] == 1) {
      broadcast_shape.push_back(x_shape[x_length + i]);
    } else if (x_shape[x_length + i] == y_shape[y_length + i]) {
      broadcast_shape.push_back(x_shape[x_length + i]);
    } else {
      MS_EXCEPTION(ValueError) << "For op " << op_name << ", the two input can not broadcast";
    }
  }
  return std::make_shared<abstract::Shape>(broadcast_shape);
}
}  // namespace ops
}  // namespace mindspore
