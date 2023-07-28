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

#include "minddata/dataset/audio/ir/kernels/overdrive_ir.h"

#include "minddata/dataset/audio/kernels/overdrive_op.h"
#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace audio {
OverdriveOperation::OverdriveOperation(float gain, float color) : gain_(gain), color_(color) {}

Status OverdriveOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalar("Overdrive", "gain", gain_, {0.0f, 100.0f}, false, false));
  RETURN_IF_NOT_OK(ValidateScalar("Overdrive", "color", color_, {0.0f, 100.0f}, false, false));
  return Status::OK();
}

std::shared_ptr<TensorOp> OverdriveOperation::Build() {
  std::shared_ptr<OverdriveOp> tensor_op = std::make_shared<OverdriveOp>(gain_, color_);
  return tensor_op;
}

Status OverdriveOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["gain"] = gain_;
  args["color"] = color_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
