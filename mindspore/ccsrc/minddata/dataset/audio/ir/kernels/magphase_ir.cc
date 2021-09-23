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

#include "minddata/dataset/audio/ir/kernels/magphase_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/magphase_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
MagphaseOperation::MagphaseOperation(float power) : power_(power) {}

Status MagphaseOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("Magphase", "power", power_));
  return Status::OK();
}

std::shared_ptr<TensorOp> MagphaseOperation::Build() { return std::make_shared<MagphaseOp>(power_); }

Status MagphaseOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["power"] = power_;
  *out_json = args;
  return Status::OK();
}

Status MagphaseOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("power") != op_params.end(), "Fail to find power");
  float power = op_params["power"];
  *operation = std::make_shared<audio::MagphaseOperation>(power);
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
