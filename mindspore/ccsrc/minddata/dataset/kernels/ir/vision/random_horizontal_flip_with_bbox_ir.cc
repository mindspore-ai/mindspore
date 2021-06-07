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

#include "minddata/dataset/kernels/ir/vision/random_horizontal_flip_with_bbox_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_horizontal_flip_with_bbox_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandomHorizontalFlipWithBBoxOperation
RandomHorizontalFlipWithBBoxOperation::RandomHorizontalFlipWithBBoxOperation(float probability)
    : TensorOperation(true), probability_(probability) {}

RandomHorizontalFlipWithBBoxOperation::~RandomHorizontalFlipWithBBoxOperation() = default;

std::string RandomHorizontalFlipWithBBoxOperation::Name() const { return kRandomHorizontalFlipWithBBoxOperation; }

Status RandomHorizontalFlipWithBBoxOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateProbability("RandomHorizontalFlipWithBBox", probability_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomHorizontalFlipWithBBoxOperation::Build() {
  std::shared_ptr<RandomHorizontalFlipWithBBoxOp> tensor_op =
    std::make_shared<RandomHorizontalFlipWithBBoxOp>(probability_);
  return tensor_op;
}

Status RandomHorizontalFlipWithBBoxOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["prob"] = probability_;
  return Status::OK();
}

Status RandomHorizontalFlipWithBBoxOperation::from_json(nlohmann::json op_params,
                                                        std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("prob") != op_params.end(), "Failed to find prob");
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::RandomHorizontalFlipWithBBoxOperation>(prob);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
