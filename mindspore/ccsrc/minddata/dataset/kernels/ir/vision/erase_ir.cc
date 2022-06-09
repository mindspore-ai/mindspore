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

#include "minddata/dataset/kernels/ir/vision/erase_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/kernels/image/erase_op.h"
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// EraseOperation
EraseOperation::EraseOperation(int32_t top, int32_t left, int32_t height, int32_t width,
                               const std::vector<uint8_t> &value, bool inplace)
    : top_(top), left_(left), height_(height), width_(width), value_(value), inplace_(inplace) {}

EraseOperation::~EraseOperation() = default;

std::string EraseOperation::Name() const { return kEraseOperation; }

Status EraseOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("Erase", "top", top_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("Erase", "left", left_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("Erase", "height", height_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("Erase", "width", width_));
  RETURN_IF_NOT_OK(ValidateVectorFillvalue("Erase", value_));
  return Status::OK();
}

std::shared_ptr<TensorOp> EraseOperation::Build() {
  std::shared_ptr<EraseOp> tensor_op = std::make_shared<EraseOp>(top_, left_, height_, width_, value_, inplace_);
  return tensor_op;
}

Status EraseOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["top"] = top_;
  args["left"] = left_;
  args["height"] = height_;
  args["width"] = width_;
  args["value"] = value_;
  args["inplace"] = inplace_;
  *out_json = args;
  return Status::OK();
}

Status EraseOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "top", kEraseOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "left", kEraseOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "height", kEraseOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "width", kEraseOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "value", kEraseOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "inplace", kEraseOperation));

  int32_t top = op_params["top"];
  int32_t left = op_params["left"];
  int32_t height = op_params["height"];
  int32_t width = op_params["width"];
  std::vector<uint8_t> value = op_params["value"];
  bool inplace = op_params["inplace"];
  *operation = std::make_shared<vision::EraseOperation>(top, left, height, width, value, inplace);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
