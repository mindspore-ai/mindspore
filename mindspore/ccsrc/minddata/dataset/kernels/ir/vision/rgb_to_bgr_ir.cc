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
#include "minddata/dataset/kernels/ir/vision/rgb_to_bgr_ir.h"

#include "minddata/dataset/kernels/image/rgb_to_bgr_op.h"

namespace mindspore {
namespace dataset {
namespace vision {
RgbToBgrOperation::RgbToBgrOperation() = default;

// RGB2BGROperation
RgbToBgrOperation::~RgbToBgrOperation() = default;

std::string RgbToBgrOperation::Name() const { return kRgbToBgrOperation; }

Status RgbToBgrOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> RgbToBgrOperation::Build() { return std::make_shared<RgbToBgrOp>(); }

Status RgbToBgrOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  *operation = std::make_shared<vision::RgbToBgrOperation>();
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
