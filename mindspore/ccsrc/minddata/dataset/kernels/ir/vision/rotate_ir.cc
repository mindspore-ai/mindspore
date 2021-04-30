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
#include <algorithm>

#include "minddata/dataset/kernels/ir/vision/rotate_ir.h"

#include "minddata/dataset/kernels/image/rotate_op.h"

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {

namespace vision {

// RotateOperation
RotateOperation::RotateOperation() { rotate_op = std::make_shared<RotateOp>(0); }

RotateOperation::~RotateOperation() = default;

std::string RotateOperation::Name() const { return kRotateOperation; }

Status RotateOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> RotateOperation::Build() { return rotate_op; }

void RotateOperation::setAngle(uint64_t angle_id) {
  std::dynamic_pointer_cast<RotateOp>(rotate_op)->setAngle(angle_id);
}

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
