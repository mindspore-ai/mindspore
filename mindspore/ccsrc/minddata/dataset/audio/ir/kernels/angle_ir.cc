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

#include "minddata/dataset/audio/ir/kernels/angle_ir.h"

#include "minddata/dataset/audio/kernels/angle_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// AngleOperation
AngleOperation::AngleOperation() {}

Status AngleOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> AngleOperation::Build() {
  std::shared_ptr<AngleOp> tensor_op = std::make_shared<AngleOp>();
  return tensor_op;
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
