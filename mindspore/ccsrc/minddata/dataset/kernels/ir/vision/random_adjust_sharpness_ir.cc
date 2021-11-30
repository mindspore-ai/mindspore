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
#include "minddata/dataset/kernels/ir/vision/random_adjust_sharpness_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_adjust_sharpness_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandomAdjustSharpnessOperation
RandomAdjustSharpnessOperation::RandomAdjustSharpnessOperation(float degree, float prob)
    : degree_(degree), probability_(prob) {}

RandomAdjustSharpnessOperation::~RandomAdjustSharpnessOperation() = default;

std::string RandomAdjustSharpnessOperation::Name() const { return kRandomAdjustSharpnessOperation; }

Status RandomAdjustSharpnessOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("RandomAdjustSharpness", "degree", degree_));
  RETURN_IF_NOT_OK(ValidateProbability("RandomAdjustSharpness", probability_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomAdjustSharpnessOperation::Build() {
  std::shared_ptr<RandomAdjustSharpnessOp> tensor_op = std::make_shared<RandomAdjustSharpnessOp>(degree_, probability_);
  return tensor_op;
}

Status RandomAdjustSharpnessOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["degree"] = degree_;
  args["prob"] = probability_;
  *out_json = args;
  return Status::OK();
}

Status RandomAdjustSharpnessOperation::from_json(nlohmann::json op_params,
                                                 std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "degree", kRandomAdjustSharpnessOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "prob", kRandomAdjustSharpnessOperation));
  float degree = op_params["degree"];
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::RandomAdjustSharpnessOperation>(degree, prob);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
