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
#include "minddata/dataset/audio/ir/validators.h"

namespace mindspore {
namespace dataset {
/* ####################################### Validator Functions ############################################ */
Status CheckFloatScalarPositive(const std::string &op_name, const std::string &scalar_name, float scalar) {
  RETURN_IF_NOT_OK(CheckScalar(op_name, scalar_name, scalar, {0}, true));
  return Status::OK();
}

Status CheckFloatScalarNotNan(const std::string &op_name, const std::string &scalar_name, float scalar) {
  if (std::isnan(scalar)) {
    std::string err_msg = op_name + ":" + scalar_name + " should be specified, got: Nan.";
    MS_LOG(ERROR) << err_msg;
    return Status(StatusCode::kMDSyntaxError, __LINE__, __FILE__, err_msg);
  }
  return Status::OK();
}

Status CheckFloatScalarNonNegative(const std::string &op_name, const std::string &scalar_name, float scalar) {
  RETURN_IF_NOT_OK(CheckScalar(op_name, scalar_name, scalar, {0}, false));
  return Status::OK();
}

Status CheckIntScalarPositive(const std::string &op_name, const std::string &scalar_name, int32_t scalar) {
  RETURN_IF_NOT_OK(CheckScalar(op_name, scalar_name, scalar, {0}, true));
  return Status::OK();
}

Status CheckStringScalarInList(const std::string &op_name, const std::string &scalar_name, const std::string &scalar,
                               const std::vector<std::string> &str_vec) {
  auto ret = std::find(str_vec.begin(), str_vec.end(), scalar);
  if (ret == str_vec.end()) {
    std::string interval_description = "[";
    for (int m = 0; m < str_vec.size(); m++) {
      std::string word = str_vec[m];
      interval_description = interval_description + word;
      if (m != str_vec.size() - 1) interval_description = interval_description + ", ";
    }
    interval_description = interval_description + "]";

    std::string err_msg = op_name + ": " + scalar_name + " must be one of " + interval_description + ", got: " + scalar;
    MS_LOG(ERROR) << err_msg;
    return Status(StatusCode::kMDSyntaxError, __LINE__, __FILE__, err_msg);
  }
  return Status::OK();
}

template <typename T>
Status CheckScalar(const std::string &op_name, const std::string &scalar_name, const T scalar,
                   const std::vector<T> &range, bool left_open_interval, bool right_open_interval) {
  if (range.empty() || range.size() > 2) {
    std::string err_msg = "Range check expecting size 1 or 2, but got: " + std::to_string(range.size());
    MS_LOG(ERROR) << err_msg;
    return Status(StatusCode::kMDSyntaxError, __LINE__, __FILE__, err_msg);
  }
  if ((left_open_interval && scalar <= range[0]) || (!left_open_interval && scalar < range[0])) {
    std::string interval_description = left_open_interval ? " greater than " : " greater than or equal to ";
    std::string err_msg = op_name + ":" + scalar_name + " must be" + interval_description + std::to_string(range[0]) +
                          ", got: " + std::to_string(scalar);
    MS_LOG(ERROR) << err_msg;
    return Status(StatusCode::kMDSyntaxError, __LINE__, __FILE__, err_msg);
  }
  if (range.size() == 2) {
    if ((right_open_interval && scalar >= range[1]) || (!right_open_interval && scalar > range[1])) {
      std::string left_bracket = left_open_interval ? "(" : "[";
      std::string right_bracket = right_open_interval ? ")" : "]";
      std::string err_msg = op_name + ":" + scalar_name + " is out of range " + left_bracket +
                            std::to_string(range[0]) + ", " + std::to_string(range[1]) + right_bracket +
                            ", got: " + std::to_string(scalar);
      MS_LOG(ERROR) << err_msg;
      return Status(StatusCode::kMDSyntaxError, __LINE__, __FILE__, err_msg);
    }
  }
  return Status::OK();
}
template Status CheckScalar(const std::string &op_name, const std::string &scalar_name, const float scalar,
                            const std::vector<float> &range, bool left_open_interval, bool right_open_interval);

template Status CheckScalar(const std::string &op_name, const std::string &scalar_name, const int32_t scalar,
                            const std::vector<int32_t> &range, bool left_open_interval, bool right_open_interval);
}  // namespace dataset
}  // namespace mindspore
