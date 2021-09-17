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

#include "minddata/dataset/audio/ir/kernels/time_masking_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/time_masking_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
TimeMaskingOperation::TimeMaskingOperation(bool iid_masks, int32_t time_mask_param, int32_t mask_start,
                                           float mask_value)
    : iid_masks_(iid_masks), time_mask_param_(time_mask_param), mask_start_(mask_start), mask_value_(mask_value) {}

TimeMaskingOperation::~TimeMaskingOperation() = default;

Status TimeMaskingOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("TimeMasking", "time_mask_param", time_mask_param_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("TimeMasking", "mask_start", mask_start_));

  return Status::OK();
}

std::shared_ptr<TensorOp> TimeMaskingOperation::Build() {
  std::shared_ptr<TimeMaskingOp> tensor_op =
    std::make_shared<TimeMaskingOp>(iid_masks_, time_mask_param_, mask_start_, mask_value_);
  return tensor_op;
}

std::string TimeMaskingOperation::Name() const { return kTimeMaskingOperation; }

Status TimeMaskingOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["time_mask_param"] = time_mask_param_;
  args["mask_start"] = mask_start_;
  args["iid_masks"] = iid_masks_;
  args["mask_value"] = mask_value_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
