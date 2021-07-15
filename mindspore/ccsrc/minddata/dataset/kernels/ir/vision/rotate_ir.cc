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
#include "minddata/dataset/kernels/ir/vision/rotate_ir.h"

#include "minddata/dataset/kernels/image/rotate_op.h"
#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
// RotateOperation
RotateOperation::RotateOperation() { rotate_op_ = std::make_shared<RotateOp>(0); }

RotateOperation::RotateOperation(float degrees, InterpolationMode resample, bool expand, std::vector<float> center,
                                 std::vector<uint8_t> fill_value)
    : degrees_(degrees), interpolation_mode_(resample), expand_(expand), center_(center), fill_value_(fill_value) {}

RotateOperation::~RotateOperation() = default;

std::string RotateOperation::Name() const { return kRotateOperation; }

Status RotateOperation::ValidateParams() {
#ifndef ENABLE_ANDROID
  // center
  if (center_.size() != 0 && center_.size() != 2) {
    std::string err_msg =
      "Rotate: center must be a vector of two values or empty, got: " + std::to_string(center_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // fill_value
  RETURN_IF_NOT_OK(ValidateVectorFillvalue("Rotate", fill_value_));
#endif
  return Status::OK();
}

std::shared_ptr<TensorOp> RotateOperation::Build() {
#ifndef ENABLE_ANDROID
  uint8_t fill_r, fill_g, fill_b;
  fill_r = fill_value_[0];
  fill_g = fill_value_[0];
  fill_b = fill_value_[0];

  if (fill_value_.size() == 3) {
    fill_r = fill_value_[0];
    fill_g = fill_value_[1];
    fill_b = fill_value_[2];
  }

  std::shared_ptr<RotateOp> tensor_op =
    std::make_shared<RotateOp>(degrees_, interpolation_mode_, expand_, center_, fill_r, fill_g, fill_b);
  return tensor_op;
#else
  return rotate_op_;
#endif
}

Status RotateOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
#ifndef ENABLE_ANDROID
  args["degree"] = degrees_;
  args["resample"] = interpolation_mode_;
  args["expand"] = expand_;
  args["center"] = center_;
  args["fill_value"] = fill_value_;
#else
  args["angle_id"] = angle_id_;
#endif
  *out_json = args;
  return Status::OK();
}

void RotateOperation::setAngle(uint64_t angle_id) {
  std::dynamic_pointer_cast<RotateOp>(rotate_op_)->setAngle(angle_id);
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
