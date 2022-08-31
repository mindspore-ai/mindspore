/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/ir/vision/rand_augment_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/rand_augment_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandAugmentOperation
RandAugmentOperation::RandAugmentOperation(int32_t num_ops, int32_t magnitude, int32_t num_magnitude_bins,
                                           InterpolationMode interpolation, const std::vector<uint8_t> &fill_value)
    : num_ops_(num_ops),
      magnitude_(magnitude),
      num_magnitude_bins_(num_magnitude_bins),
      interpolation_(interpolation),
      fill_value_(fill_value) {}

RandAugmentOperation::~RandAugmentOperation() = default;

std::shared_ptr<TensorOp> RandAugmentOperation::Build() {
  std::shared_ptr<RandAugmentOp> tensor_op =
    std::make_shared<RandAugmentOp>(num_ops_, magnitude_, num_magnitude_bins_, interpolation_, fill_value_);
  return tensor_op;
}

Status RandAugmentOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("RandAugment", "num_ops", num_ops_));
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("RandAugment", "magnitude", magnitude_));
  RETURN_IF_NOT_OK(ValidateScalar("RandAugment", "num_magnitude_bins", num_magnitude_bins_, {1}, true));
  CHECK_FAIL_RETURN_UNEXPECTED(magnitude_ < num_magnitude_bins_,
                               "RandAugment: magnitude should be smaller than num_magnitude_bins.");

  if (interpolation_ != InterpolationMode::kLinear && interpolation_ != InterpolationMode::kNearestNeighbour &&
      interpolation_ != InterpolationMode::kCubic && interpolation_ != InterpolationMode::kArea) {
    std::string err_msg =
      "RandAugment: Invalid InterpolationMode. Use InterpolationMode::kLinear, InterpolationMode::kNearestNeighbour,"
      " InterpolationMode::kCubic or InterpolationMode::kArea.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  RETURN_IF_NOT_OK(ValidateVectorFillvalue("RandAugment", fill_value_));
  return Status::OK();
}

std::string RandAugmentOperation::Name() const { return kRandAugmentOperation; }

Status RandAugmentOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["num_ops"] = num_ops_;
  args["magnitude"] = magnitude_;
  args["num_magnitude_bins"] = num_magnitude_bins_;
  args["interpolation"] = interpolation_;
  args["fill_value"] = fill_value_;
  *out_json = args;
  return Status::OK();
}

Status RandAugmentOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "num_ops", kRandAugmentOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "magnitude", kRandAugmentOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "num_magnitude_bins", kRandAugmentOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "interpolation", kRandAugmentOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "fill_value", kRandAugmentOperation));
  int32_t num_ops = op_params["num_ops"];
  int32_t magnitude = op_params["magnitude"];
  int32_t num_magnitude_bins = op_params["num_magnitude_bins"];
  InterpolationMode interpolation = op_params["interpolation"];
  std::vector<uint8_t> fill_value = op_params["fill_value"];
  *operation =
    std::make_shared<vision::RandAugmentOperation>(num_ops, magnitude, num_magnitude_bins, interpolation, fill_value);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
