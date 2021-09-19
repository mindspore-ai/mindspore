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
#include "minddata/dataset/kernels/ir/vision/horizontal_flip_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/horizontal_flip_op.h"
#endif

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID

// VerticalFlipOperation
HorizontalFlipOperation::HorizontalFlipOperation() {}

HorizontalFlipOperation::~HorizontalFlipOperation() = default;

std::string HorizontalFlipOperation::Name() const { return kHorizontalFlipOperation; }

Status HorizontalFlipOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> HorizontalFlipOperation::Build() {
  std::shared_ptr<HorizontalFlipOp> tensor_op = std::make_shared<HorizontalFlipOp>();
  return tensor_op;
}

Status HorizontalFlipOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  *operation = std::make_shared<vision::HorizontalFlipOperation>();
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
