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

#include "minddata/dataset/audio/ir/kernels/dc_shift_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/dc_shift_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// DCShiftOperation
DCShiftOperation::DCShiftOperation(float shift, float limiter_gain) : shift_(shift), limiter_gain_(limiter_gain) {}

Status DCShiftOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalar("DCShift", "shift", shift_, {-2.0, 2.0}, false, false));
  return Status::OK();
}

std::shared_ptr<TensorOp> DCShiftOperation::Build() {
  std::shared_ptr<DCShiftOp> tensor_op = std::make_shared<DCShiftOp>(shift_, limiter_gain_);
  return tensor_op;
}

Status DCShiftOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["shift"] = shift_;
  args["limiter_gain"] = limiter_gain_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
