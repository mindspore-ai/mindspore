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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VALIDATORS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VALIDATORS_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"
#include "minddata/dataset/util/status.h"

constexpr int64_t size_two = 2;
constexpr int64_t size_three = 3;
constexpr int64_t size_four = 4;

namespace mindspore {
namespace dataset {
// Helper function to validate probability
Status ValidateProbability(const std::string &op_name, const double probability);

// Helper function to positive int scalar
Status ValidateIntScalarPositive(const std::string &op_name, const std::string &scalar_name, int32_t scalar);

// Helper function to positive int scalar
Status ValidateIntScalarNonNegative(const std::string &op_name, const std::string &scalar_name, int32_t scalar);

// Helper function to positive float scalar
Status ValidateFloatScalarPositive(const std::string &op_name, const std::string &scalar_name, float scalar);

// Helper function to non-negative float scalar
Status ValidateFloatScalarNonNegative(const std::string &op_name, const std::string &scalar_name, float scalar);

// Helper function to validate scalar
template <typename T>
Status ValidateScalar(const std::string &op_name, const std::string &scalar_name, const T scalar,
                      const std::vector<T> &range, bool left_open_interval = false, bool right_open_interval = false) {
  if (range.empty() || range.size() > size_two) {
    std::string err_msg = op_name + ": expecting range size 1 or 2, but got: " + std::to_string(range.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_SYNTAX_ERROR(err_msg);
  }
  if ((left_open_interval && scalar <= range[0]) || (!left_open_interval && scalar < range[0])) {
    std::string interval_description = left_open_interval ? " greater than " : " greater than or equal to ";
    std::string err_msg = op_name + ": '" + scalar_name + "' must be" + interval_description +
                          std::to_string(range[0]) + ", got: " + std::to_string(scalar);
    MS_LOG(ERROR) << err_msg;
    RETURN_SYNTAX_ERROR(err_msg);
  }
  if (range.size() == size_two) {
    if ((right_open_interval && scalar >= range[1]) || (!right_open_interval && scalar > range[1])) {
      std::string left_bracket = left_open_interval ? "(" : "[";
      std::string right_bracket = right_open_interval ? ")" : "]";
      std::string err_msg = op_name + ":" + scalar_name + " is out of range " + left_bracket +
                            std::to_string(range[0]) + ", " + std::to_string(range[1]) + right_bracket +
                            ", got: " + std::to_string(scalar);
      MS_LOG(ERROR) << err_msg;
      RETURN_SYNTAX_ERROR(err_msg);
    }
  }
  return Status::OK();
}

// Helper function to validate enum
template <typename T>
Status ValidateEnum(const std::string &op_name, const std::string &enum_name, const T enumeration,
                    const std::vector<T> &enum_list) {
  auto existed = std::find(enum_list.begin(), enum_list.end(), enumeration);
  std::string err_msg = op_name + ": Invalid " + enum_name + ", check input value of enum.";
  if (existed != enum_list.end()) {
    return Status::OK();
  }
  RETURN_SYNTAX_ERROR(err_msg);
}

// Helper function to validate color attribute
Status ValidateVectorColorAttribute(const std::string &op_name, const std::string &attr_name,
                                    const std::vector<float> &attr, const std::vector<float> &range);

// Helper function to validate fill value
Status ValidateVectorFillvalue(const std::string &op_name, const std::vector<uint8_t> &fill_value);

// Helper function to validate mean/std value
Status ValidateVectorMeanStd(const std::string &op_name, const std::vector<float> &mean, const std::vector<float> &std);

// Helper function to validate odd value
Status ValidateVectorOdd(const std::string &op_name, const std::string &vec_name, const std::vector<int32_t> &value);

// Helper function to validate padding
Status ValidateVectorPadding(const std::string &op_name, const std::vector<int32_t> &padding);

// Helper function to validate positive value
Status ValidateVectorPositive(const std::string &op_name, const std::string &vec_name, const std::vector<int32_t> &vec);

// Helper function to validate non-negative value
Status ValidateVectorNonNegative(const std::string &op_name, const std::string &vec_name,
                                 const std::vector<int32_t> &vec);

// Helper function to validate size of sigma
Status ValidateVectorSigma(const std::string &op_name, const std::vector<float> &sigma);

// Helper function to validate size of size
Status ValidateVectorSize(const std::string &op_name, const std::vector<int32_t> &size);

// Helper function to validate scale
Status ValidateVectorScale(const std::string &op_name, const std::vector<float> &scale);

// Helper function to validate ratio
Status ValidateVectorRatio(const std::string &op_name, const std::vector<float> &ratio);

// Helper function to validate transforms
Status ValidateVectorTransforms(const std::string &op_name,
                                const std::vector<std::shared_ptr<TensorOperation>> &transforms);

// Helper function to compare float value
bool CmpFloat(const float a, const float b, float epsilon = 0.0000000001f);
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IR_VALIDATORS_H_
