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

#include "minddata/dataset/audio/ir/kernels/gain_ir.h"

#include "minddata/dataset/audio/kernels/gain_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// GainOperation
GainOperation::GainOperation(float gain_db) : gain_db_(gain_db) {}

Status GainOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> GainOperation::Build() {
  std::shared_ptr<GainOp> tensor_op = std::make_shared<GainOp>(gain_db_);
  return tensor_op;
}

Status GainOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["gain_db"] = gain_db_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
