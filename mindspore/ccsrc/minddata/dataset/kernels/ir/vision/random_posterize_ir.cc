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
#include <algorithm>

#include "minddata/dataset/kernels/ir/vision/random_posterize_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_posterize_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandomPosterizeOperation
RandomPosterizeOperation::RandomPosterizeOperation(const std::vector<uint8_t> &bit_range)
    : TensorOperation(true), bit_range_(bit_range) {}

RandomPosterizeOperation::~RandomPosterizeOperation() = default;

std::string RandomPosterizeOperation::Name() const { return kRandomPosterizeOperation; }

Status RandomPosterizeOperation::ValidateParams() {
  constexpr size_t dimension_zero = 0;
  constexpr size_t dimension_one = 1;
  constexpr size_t size_two = 2;
  constexpr uint8_t kMinimumBitValue = 1;
  constexpr uint8_t kMaximumBitValue = 8;

  if (bit_range_.size() != size_two) {
    std::string err_msg =
      "RandomPosterize: bit_range needs to be of size 2 but is of size: " + std::to_string(bit_range_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (bit_range_[dimension_zero] < kMinimumBitValue || bit_range_[dimension_zero] > kMaximumBitValue) {
    std::string err_msg =
      "RandomPosterize: min_bit value is out of range [1-8]: " + std::to_string(bit_range_[dimension_zero]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (bit_range_[dimension_one] < kMinimumBitValue || bit_range_[dimension_one] > kMaximumBitValue) {
    std::string err_msg =
      "RandomPosterize: max_bit value is out of range [1-8]: " + std::to_string(bit_range_[dimension_one]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (bit_range_[dimension_one] < bit_range_[dimension_zero]) {
    std::string err_msg =
      "RandomPosterize: max_bit value is less than min_bit: max =" + std::to_string(bit_range_[dimension_one]) +
      ", min = " + std::to_string(bit_range_[dimension_zero]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomPosterizeOperation::Build() {
  std::shared_ptr<RandomPosterizeOp> tensor_op = std::make_shared<RandomPosterizeOp>(bit_range_);
  return tensor_op;
}

Status RandomPosterizeOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["bits"] = bit_range_;
  return Status::OK();
}

Status RandomPosterizeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("bits") != op_params.end(), "Failed to find bits");
  std::vector<uint8_t> bit_range = op_params["bits"];
  *operation = std::make_shared<vision::RandomPosterizeOperation>(bit_range);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
