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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_VALIDATORS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_VALIDATORS_H_

#include <limits>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// validator Parameter in json file
inline Status ValidateParamInJson(const nlohmann::json &json_obj, const std::string &param_name,
                                  const std::string &operator_name) {
  if (json_obj.find(param_name) == json_obj.end()) {
    std::string err_msg = "Failed to find key '" + param_name + "' in " + operator_name +
                          "' JSON file or input dict, check input content of deserialize().";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

inline Status ValidateTensorShape(const std::string &op_name, bool cond, const std::string &expected_shape = "",
                                  const std::string &actual_dim = "") {
  if (!cond) {
    std::string err_msg = op_name + ": the shape of input tensor does not match the requirement of operator.";
    if (expected_shape != "") {
      err_msg += " Expecting tensor in shape of " + expected_shape + ".";
    }
    if (actual_dim != "") {
      err_msg += " But got tensor with dimension " + actual_dim + ".";
    }
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

inline Status ValidateLowRank(const std::string &op_name, const std::shared_ptr<Tensor> &input, dsize_t threshold = 0,
                              const std::string &expected_shape = "") {
  dsize_t dim = input->shape().Size();
  return ValidateTensorShape(op_name, dim >= threshold, expected_shape, std::to_string(dim));
}

inline Status ValidateTensorType(const std::string &op_name, bool cond, const std::string &expected_type = "",
                                 const std::string &actual_type = "") {
  if (!cond) {
    std::string err_msg = op_name + ": the data type of input tensor does not match the requirement of operator.";
    if (expected_type != "") {
      err_msg += " Expecting tensor in type of " + expected_type + ".";
    }
    if (actual_type != "") {
      err_msg += " But got type " + actual_type + ".";
    }
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

inline Status ValidateTensorNumeric(const std::string &op_name, const std::shared_ptr<Tensor> &input) {
  return ValidateTensorType(op_name, input->type().IsNumeric(), "[int, float, double]", input->type().ToString());
}

inline Status ValidateTensorFloat(const std::string &op_name, const std::shared_ptr<Tensor> &input) {
  return ValidateTensorType(op_name, input->type().IsFloat(), "[float, double]", input->type().ToString());
}

template <typename T>
inline Status ValidateEqual(const std::string &op_name, const std::string &param_name, T param_value,
                            const std::string &other_name, T other_value) {
  if (param_value != other_value) {
    std::string err_msg = op_name + ": invalid parameter, '" + param_name + "' should be equal to '" + other_name +
                          "', but got: " + param_name + " " + std::to_string(param_value) + " while " + other_name +
                          " " + std::to_string(other_value) + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

template <typename T>
inline Status ValidateNotEqual(const std::string &op_name, const std::string &param_name, T param_value,
                               const std::string &other_name, T other_value) {
  if (param_value == other_value) {
    std::string err_msg = op_name + ": invalid parameter, '" + param_name + "' can not be equal to '" + other_name +
                          "', but got: " + param_name + " " + std::to_string(param_value) + " while " + other_name +
                          " " + std::to_string(other_value) + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

template <typename T>
inline Status ValidateGreaterThan(const std::string &op_name, const std::string &param_name, T param_value,
                                  const std::string &other_name, T other_value) {
  if (param_value <= other_value) {
    std::string err_msg = op_name + ": invalid parameter, '" + param_name + "' should be greater than '" + other_name +
                          "', but got: " + param_name + " " + std::to_string(param_value) + " while " + other_name +
                          " " + std::to_string(other_value) + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

template <typename T>
inline Status ValidateLessThan(const std::string &op_name, const std::string &param_name, T param_value,
                               const std::string &other_name, T other_value) {
  if (param_value >= other_value) {
    std::string err_msg = op_name + ": invalid parameter, '" + param_name + "' should be less than '" + other_name +
                          "', but got: " + param_name + " " + std::to_string(param_value) + " while " + other_name +
                          " " + std::to_string(other_value) + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

template <typename T>
inline Status ValidateNoGreaterThan(const std::string &op_name, const std::string &param_name, T param_value,
                                    const std::string &other_name, T other_value) {
  if (param_value > other_value) {
    std::string err_msg = op_name + ": invalid parameter, '" + param_name + "' should be no greater than '" +
                          other_name + "', but got: " + param_name + " " + std::to_string(param_value) + " while " +
                          other_name + " " + std::to_string(other_value) + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

template <typename T>
inline Status ValidateNoLessThan(const std::string &op_name, const std::string &param_name, T param_value,
                                 const std::string &other_name, T other_value) {
  if (param_value < other_value) {
    std::string err_msg = op_name + ": invalid parameter, '" + param_name + "' should be no less than '" + other_name +
                          "', but got: " + param_name + " " + std::to_string(param_value) + " while " + other_name +
                          " " + std::to_string(other_value) + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

template <typename T>
inline Status ValidatePositive(const std::string &op_name, const std::string &param_name, T param_value) {
  if (param_value <= 0) {
    std::string err_msg = op_name + ": invalid parameter, '" + param_name +
                          "' should be positive, but got: " + std::to_string(param_value) + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

template <typename T>
inline Status ValidateNegative(const std::string &op_name, const std::string &param_name, T param_value) {
  if (param_value >= 0) {
    std::string err_msg = op_name + ": invalid parameter, '" + param_name +
                          "' should be negative, but got: " + std::to_string(param_value) + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

template <typename T>
inline Status ValidateNonPositive(const std::string &op_name, const std::string &param_name, T param_value) {
  if (param_value > 0) {
    std::string err_msg = op_name + ": invalid parameter, '" + param_name +
                          "' should be non positive, but got: " + std::to_string(param_value) + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

template <typename T>
inline Status ValidateNonNegative(const std::string &op_name, const std::string &param_name, T param_value) {
  if (param_value < 0) {
    std::string err_msg = op_name + ": invalid parameter, '" + param_name +
                          "' should be non negative, but got: " + std::to_string(param_value) + ".";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  return Status::OK();
}

inline std::string DataTypeSetToString(const std::set<uint8_t> &valid_dtype) {
  std::string init;
  std::string err_msg =
    std::accumulate(valid_dtype.begin(), valid_dtype.end(), init, [](const std::string &str, uint8_t dtype) {
      if (str.empty()) {
        return DataType(DataType::Type(dtype)).ToString();
      } else {
        return str + ", " + DataType(DataType::Type(dtype)).ToString();
      }
    });
  return "(" + err_msg + ")";
}

template <typename T>
std::string NumberSetToString(const std::set<T> &valid_value) {
  std::string init;
  std::string err_msg =
    std::accumulate(valid_value.begin(), valid_value.end(), init, [](const std::string &str, T value) {
      if (str.empty()) {
        return std::to_string(value);
      } else {
        return str + ", " + std::to_string(value);
      }
    });
  return "(" + err_msg + ")";
}
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_VALIDATORS_H_
