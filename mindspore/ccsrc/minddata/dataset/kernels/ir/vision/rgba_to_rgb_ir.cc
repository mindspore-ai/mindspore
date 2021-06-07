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

#include "minddata/dataset/kernels/ir/vision/rgba_to_rgb_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/rgba_to_rgb_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RgbaToRgbOperation.
RgbaToRgbOperation::RgbaToRgbOperation() {}

RgbaToRgbOperation::~RgbaToRgbOperation() = default;

std::string RgbaToRgbOperation::Name() const { return kRgbaToRgbOperation; }

Status RgbaToRgbOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> RgbaToRgbOperation::Build() {
  std::shared_ptr<RgbaToRgbOp> tensor_op = std::make_shared<RgbaToRgbOp>();
  return tensor_op;
}

Status RgbaToRgbOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  *operation = std::make_shared<vision::RgbaToRgbOperation>();
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
