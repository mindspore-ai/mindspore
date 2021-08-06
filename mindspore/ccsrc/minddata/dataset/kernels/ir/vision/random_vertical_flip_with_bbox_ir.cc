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

#include "minddata/dataset/kernels/ir/vision/random_vertical_flip_with_bbox_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_vertical_flip_with_bbox_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandomVerticalFlipWithBBoxOperation
RandomVerticalFlipWithBBoxOperation::RandomVerticalFlipWithBBoxOperation(float prob)
    : TensorOperation(true), probability_(prob) {}

RandomVerticalFlipWithBBoxOperation::~RandomVerticalFlipWithBBoxOperation() = default;

std::string RandomVerticalFlipWithBBoxOperation::Name() const { return kRandomVerticalFlipWithBBoxOperation; }

Status RandomVerticalFlipWithBBoxOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateProbability("RandomVerticalFlipWithBBox", probability_));

  return Status::OK();
}

std::shared_ptr<TensorOp> RandomVerticalFlipWithBBoxOperation::Build() {
  std::shared_ptr<RandomVerticalFlipWithBBoxOp> tensor_op =
    std::make_shared<RandomVerticalFlipWithBBoxOp>(probability_);
  return tensor_op;
}

Status RandomVerticalFlipWithBBoxOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["prob"] = probability_;
  return Status::OK();
}

Status RandomVerticalFlipWithBBoxOperation::from_json(nlohmann::json op_params,
                                                      std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("prob") != op_params.end(), "Failed to find prob");
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::RandomVerticalFlipWithBBoxOperation>(prob);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
