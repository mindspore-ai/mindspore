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
#include "minddata/dataset/kernels/ir/vision/random_crop_ir.h"

#include <algorithm>

#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_crop_op.h"
#endif

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandomCropOperation
RandomCropOperation::RandomCropOperation(const std::vector<int32_t> &size, const std::vector<int32_t> &padding,
                                         bool pad_if_needed, const std::vector<uint8_t> &fill_value,
                                         BorderType padding_mode)
    : TensorOperation(true),
      size_(size),
      padding_(padding),
      pad_if_needed_(pad_if_needed),
      fill_value_(fill_value),
      padding_mode_(padding_mode) {
  random_op_ = true;
}

RandomCropOperation::~RandomCropOperation() = default;

std::string RandomCropOperation::Name() const { return kRandomCropOperation; }

Status RandomCropOperation::ValidateParams() {
  // size
  RETURN_IF_NOT_OK(ValidateVectorSize("RandomCrop", size_));
  // padding
  RETURN_IF_NOT_OK(ValidateVectorPadding("RandomCrop", padding_));
  // fill_value
  RETURN_IF_NOT_OK(ValidateVectorFillvalue("RandomCrop", fill_value_));
  // padding_mode
  if (padding_mode_ != BorderType::kConstant && padding_mode_ != BorderType::kEdge &&
      padding_mode_ != BorderType::kReflect && padding_mode_ != BorderType::kSymmetric) {
    std::string err_msg = "RandomCrop: Invalid BorderType, check input value of enum.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomCropOperation::Build() {
  constexpr size_t dimension_zero = 0;
  constexpr size_t dimension_one = 1;
  constexpr size_t dimension_two = 2;
  constexpr size_t dimension_three = 3;
  constexpr size_t size_one = 1;
  constexpr size_t size_two = 2;
  constexpr size_t size_three = 3;

  int32_t crop_height = size_[dimension_zero];
  int32_t crop_width = size_[dimension_zero];

  // User has specified the crop_width value.
  if (size_.size() == size_two) {
    crop_width = size_[dimension_one];
  }

  int32_t pad_top, pad_bottom, pad_left, pad_right;
  switch (padding_.size()) {
    case size_one:
      pad_left = padding_[dimension_zero];
      pad_top = padding_[dimension_zero];
      pad_right = padding_[dimension_zero];
      pad_bottom = padding_[dimension_zero];
      break;
    case size_two:
      pad_left = padding_[dimension_zero];
      pad_right = padding_[dimension_zero];
      pad_top = padding_[dimension_one];
      pad_bottom = padding_[dimension_one];
      break;
    default:
      pad_left = padding_[dimension_zero];
      pad_top = padding_[dimension_one];
      pad_right = padding_[dimension_two];
      pad_bottom = padding_[dimension_three];
  }

  uint8_t fill_r, fill_g, fill_b;
  fill_r = fill_value_[dimension_zero];
  fill_g = fill_value_[dimension_zero];
  fill_b = fill_value_[dimension_zero];

  if (fill_value_.size() == size_three) {
    fill_r = fill_value_[dimension_zero];
    fill_g = fill_value_[dimension_one];
    fill_b = fill_value_[dimension_two];
  }

  auto tensor_op = std::make_shared<RandomCropOp>(crop_height, crop_width, pad_top, pad_bottom, pad_left, pad_right,
                                                  pad_if_needed_, padding_mode_, fill_r, fill_g, fill_b);
  return tensor_op;
}

Status RandomCropOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["size"] = size_;
  args["padding"] = padding_;
  args["pad_if_needed"] = pad_if_needed_;
  args["fill_value"] = fill_value_;
  args["padding_mode"] = padding_mode_;
  *out_json = args;
  return Status::OK();
}

Status RandomCropOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "size", kRandomCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "padding", kRandomCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "pad_if_needed", kRandomCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "fill_value", kRandomCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "padding_mode", kRandomCropOperation));
  std::vector<int32_t> size = op_params["size"];
  std::vector<int32_t> padding = op_params["padding"];
  bool pad_if_needed = op_params["pad_if_needed"];
  std::vector<uint8_t> fill_value = op_params["fill_value"];
  BorderType padding_mode = static_cast<BorderType>(op_params["padding_mode"]);
  *operation = std::make_shared<vision::RandomCropOperation>(size, padding, pad_if_needed, fill_value, padding_mode);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
