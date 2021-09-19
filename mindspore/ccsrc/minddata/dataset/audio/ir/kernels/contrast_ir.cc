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

#include "minddata/dataset/audio/ir/kernels/contrast_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/contrast_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// ContrastOperation
ContrastOperation::ContrastOperation(float enhancement_amount) : enhancement_amount_(enhancement_amount) {}

Status ContrastOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalar("Contrast", "enhancement_amount", enhancement_amount_, {0, 100.0}, false, false));
  return Status::OK();
}

std::shared_ptr<TensorOp> ContrastOperation::Build() {
  std::shared_ptr<ContrastOp> tensor_op = std::make_shared<ContrastOp>(enhancement_amount_);
  return tensor_op;
}

Status ContrastOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["enhancement_amount"] = enhancement_amount_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
