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
#include <iostream>
#include <iterator>
#include <sstream>
#include <algorithm>
#include "device/ascend/profiling/reporter/profiling_desc.h"

namespace mindspore {
namespace device {
namespace ascend {
std::string TaskDesc::ToString() {
  std::string out = op_name_;
  out.append(" ")
    .append(std::to_string(block_dim_))
    .append(" ")
    .append(std::to_string(task_id_))
    .append(" ")
    .append(std::to_string(stream_id_))
    .append("\n");
  return out;
}

std::string GraphDesc::ToString() {
  std::string desc;
  desc.append("op_name:").append(op_name_).append(" op_type:").append(op_type_);
  int input_id = 0;
  for (const auto &element : input_data_list_) {
    desc.append(" input_id:")
      .append(std::to_string(input_id++))
      .append(" input_format:")
      .append(element.data_format_)
      .append(" input_data_type:")
      .append(std::to_string(element.data_type_))
      .append(" input_shape:")
      .append(DataShapeToString(element.data_shape_));
  }

  input_id = 0;
  for (const auto &element : output_data_list_) {
    desc.append(" output_id:")
      .append(std::to_string(input_id++))
      .append(" output_format:")
      .append(element.data_format_)
      .append(" output_data_type:")
      .append(std::to_string(element.data_type_))
      .append(" output_shape:")
      .append((DataShapeToString(element.data_shape_)));
  }

  desc.append("\n");

  return desc;
}

std::string PointDesc::ToString() {
  std::string desc;
  desc.append(std::to_string(point_id_)).append(" ").append(op_name_).append("\n");
  return desc;
}

std::string GraphDesc::DataShapeToString(const std::vector<size_t> &shape) {
  std::ostringstream oss;
  oss << "\"";
  if (!shape.empty()) {
    std::copy(shape.begin(), shape.end() - 1, std::ostream_iterator<size_t>(oss, ","));
    oss << shape.back();
  }
  oss << "\"";
  return oss.str();
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
