/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/ir/vision/pad_to_size_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/pad_to_size_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// PadOperation
PadToSizeOperation::PadToSizeOperation(const std::vector<int32_t> &size, const std::vector<int32_t> &offset,
                                       const std::vector<uint8_t> &fill_value, BorderType padding_mode)
    : size_(size), offset_(offset), fill_value_(fill_value), padding_mode_(padding_mode) {}

PadToSizeOperation::~PadToSizeOperation() = default;

std::string PadToSizeOperation::Name() const { return kPadToSizeOperation; }

Status PadToSizeOperation::ValidateParams() {
  constexpr size_t kTargetSize = 2;
  CHECK_FAIL_RETURN_SYNTAX_ERROR(
    !size_.empty() && size_.size() <= kTargetSize,
    "PadToSize: size should be in length of 1 or 2, but got: " + std::to_string(size_.size()));
  RETURN_IF_NOT_OK(ValidateVectorPositive("PadToSize", "size", size_));
  CHECK_FAIL_RETURN_SYNTAX_ERROR(
    offset_.size() <= kTargetSize,
    "PadToSize: offset should be empty or in length of 1 or 2, but got: " + std::to_string(offset_.size()));
  RETURN_IF_NOT_OK(ValidateVectorNonNegative("PadToSize", "offset", offset_));
  RETURN_IF_NOT_OK(ValidateVectorFillvalue("PadToSize", fill_value_));
  if (padding_mode_ != BorderType::kConstant && padding_mode_ != BorderType::kEdge &&
      padding_mode_ != BorderType::kReflect && padding_mode_ != BorderType::kSymmetric) {
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR("PadToSize: invalid BorderType, check input value of enum.");
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> PadToSizeOperation::Build() {
  // If fill_value size is 1, use it to fill all R, G, B channels.
  if (fill_value_.size() == 1) {
    fill_value_.push_back(fill_value_[0]);
    fill_value_.push_back(fill_value_[0]);
  }
  return std::make_shared<PadToSizeOp>(size_, offset_, fill_value_, padding_mode_);
}

Status PadToSizeOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["size"] = size_;
  args["offset"] = offset_;
  args["fill_value"] = fill_value_;
  args["padding_mode"] = padding_mode_;
  *out_json = args;
  return Status::OK();
}

Status PadToSizeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "size", kPadToSizeOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "offset", kPadToSizeOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "fill_value", kPadToSizeOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "padding_mode", kPadToSizeOperation));
  std::vector<int32_t> size = op_params["size"];
  std::vector<int32_t> offset = op_params["offset"];
  std::vector<uint8_t> fill_value = op_params["fill_value"];
  auto padding_mode = static_cast<BorderType>(op_params["padding_mode"]);
  *operation = std::make_shared<vision::PadToSizeOperation>(size, offset, fill_value, padding_mode);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
