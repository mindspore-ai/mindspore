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
#include "minddata/dataset/kernels/ir/vision/trivial_augment_wide_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/trivial_augment_wide_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// TrivialAugmentWideOperation
TrivialAugmentWideOperation::TrivialAugmentWideOperation(int32_t num_magnitude_bins, InterpolationMode interpolation,
                                                         const std::vector<uint8_t> &fill_value)
    : num_magnitude_bins_(num_magnitude_bins), interpolation_(interpolation), fill_value_(fill_value) {}

TrivialAugmentWideOperation::~TrivialAugmentWideOperation() = default;

std::string TrivialAugmentWideOperation::Name() const { return kTrivialAugmentWideOperation; }

Status TrivialAugmentWideOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalar("TrivialAugmentWide", "num_magnitude_bins", num_magnitude_bins_, {1}, true));
  if (interpolation_ != InterpolationMode::kLinear && interpolation_ != InterpolationMode::kNearestNeighbour &&
      interpolation_ != InterpolationMode::kCubic && interpolation_ != InterpolationMode::kArea) {
    std::string err_msg = "TrivialAugmentWide: Invalid InterpolationMode, check input value of enum.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  RETURN_IF_NOT_OK(ValidateVectorFillvalue("TrivialAugmentWide", fill_value_));
  return Status::OK();
}

std::shared_ptr<TensorOp> TrivialAugmentWideOperation::Build() {
  std::shared_ptr<TrivialAugmentWideOp> tensor_op =
    std::make_shared<TrivialAugmentWideOp>(num_magnitude_bins_, interpolation_, fill_value_);
  return tensor_op;
}

Status TrivialAugmentWideOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["num_magnitude_bins"] = num_magnitude_bins_;
  args["interpolation"] = interpolation_;
  args["fill_value"] = fill_value_;
  *out_json = args;
  return Status::OK();
}

Status TrivialAugmentWideOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "num_magnitude_bins", kTrivialAugmentWideOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "interpolation", kTrivialAugmentWideOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "fill_value", kTrivialAugmentWideOperation));
  float num_magnitude_bins = op_params["num_magnitude_bins"];
  InterpolationMode interpolation = op_params["interpolation"];
  std::vector<uint8_t> fill_value = op_params["fill_value"];
  *operation = std::make_shared<vision::TrivialAugmentWideOperation>(num_magnitude_bins, interpolation, fill_value);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
