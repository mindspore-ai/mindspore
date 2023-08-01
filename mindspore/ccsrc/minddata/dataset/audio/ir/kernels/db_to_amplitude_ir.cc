/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/audio/ir/kernels/db_to_amplitude_ir.h"

#include "minddata/dataset/audio/kernels/db_to_amplitude_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// DBToAmplitudeOperation
DBToAmplitudeOperation::DBToAmplitudeOperation(float ref, float power) : ref_(ref), power_(power) {}

Status DBToAmplitudeOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> DBToAmplitudeOperation::Build() {
  std::shared_ptr<DBToAmplitudeOp> tensor_op = std::make_shared<DBToAmplitudeOp>(ref_, power_);
  return tensor_op;
}

Status DBToAmplitudeOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["ref"] = ref_;
  args["power"] = power_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
