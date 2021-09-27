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
#include "minddata/dataset/kernels/ir/vision/rgb_to_gray_ir.h"

#include "minddata/dataset/kernels/image/rgb_to_gray_op.h"

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
RgbToGrayOperation::RgbToGrayOperation() = default;

// RGB2GRAYOperation
RgbToGrayOperation::~RgbToGrayOperation() = default;

std::string RgbToGrayOperation::Name() const { return kRgbToGrayOperation; }

Status RgbToGrayOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> RgbToGrayOperation::Build() { return std::make_shared<RgbToGrayOp>(); }

Status RgbToGrayOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  *operation = std::make_shared<vision::RgbToGrayOperation>();
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
