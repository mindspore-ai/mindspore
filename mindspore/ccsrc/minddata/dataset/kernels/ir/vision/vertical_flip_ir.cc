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
#include "minddata/dataset/kernels/ir/vision/vertical_flip_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/vertical_flip_op.h"
#endif

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID

// VerticalFlipOperation
VerticalFlipOperation::VerticalFlipOperation() {}

VerticalFlipOperation::~VerticalFlipOperation() = default;

std::string VerticalFlipOperation::Name() const { return kVerticalFlipOperation; }

Status VerticalFlipOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> VerticalFlipOperation::Build() {
  std::shared_ptr<VerticalFlipOp> tensor_op = std::make_shared<VerticalFlipOp>();
  return tensor_op;
}

Status VerticalFlipOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  *operation = std::make_shared<vision::VerticalFlipOperation>();
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
