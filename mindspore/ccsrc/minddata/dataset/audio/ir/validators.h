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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_VALIDATORS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_VALIDATORS_H_

#include <string>
#include <vector>
#include "minddata/dataset/kernels/ir/validators.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

// Helper function to non-nan float scalar
Status CheckFloatScalarNotNan(const std::string &op_name, const std::string &scalar_name, float scalar);

// Helper function to positive float scalar
Status CheckFloatScalarPositive(const std::string &op_name, const std::string &scalar_name, float scalar);

// Helper function to positive int scalar
Status CheckIntScalarPositive(const std::string &op_name, const std::string &scalar_name, int32_t scalar);

template <typename T>
// Helper function to check scalar is not equal to zero
Status CheckScalarNotZero(const std::string &op_name, const std::string &scalar_name, const T scalar) {
  if (scalar == 0) {
    std::string err_msg = op_name + ":" + scalar_name + " can't be 0" + ", got: " + std::to_string(scalar);
    MS_LOG(ERROR) << err_msg;
    return Status(StatusCode::kMDSyntaxError, __LINE__, __FILE__, err_msg);
  }
  return Status::OK();
}

// Helper function to positive float scalar
Status CheckFloatScalarPositive(const std::string &op_name, const std::string &scalar_name, float scalar);

// Helper function to non-negative float scalar
Status CheckFloatScalarNonNegative(const std::string &op_name, const std::string &scalar_name, float scalar);

// Helper function to check string scalar
Status CheckStringScalarInList(const std::string &op_name, const std::string &scalar_name, const std::string &scalar,
                               const std::vector<std::string> &str_vec);

// Helper function to validate scalar
template <typename T>
Status CheckScalar(const std::string &op_name, const std::string &scalar_name, const T scalar,
                   const std::vector<T> &range, bool left_open_interval = false, bool right_open_interval = false);
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ADUIO_IR_VALIDATORS_H_
