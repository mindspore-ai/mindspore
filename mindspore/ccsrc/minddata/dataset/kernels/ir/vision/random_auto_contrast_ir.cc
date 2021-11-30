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
#include "minddata/dataset/kernels/ir/vision/random_auto_contrast_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_auto_contrast_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandomAutoContrastOperation
RandomAutoContrastOperation::RandomAutoContrastOperation(float cutoff, const std::vector<uint32_t> &ignore, float prob)
    : cutoff_(cutoff), ignore_(ignore), probability_(prob) {}

RandomAutoContrastOperation::~RandomAutoContrastOperation() = default;

std::string RandomAutoContrastOperation::Name() const { return kRandomAutoContrastOperation; }

Status RandomAutoContrastOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalar("RandomAutoContrast", "cutoff", cutoff_, {0, 50}, false, true));

  for (auto i = 0; i < ignore_.size(); i++) {
    RETURN_IF_NOT_OK(ValidateScalar("RandomAutoContrast", "ignore[" + std::to_string(i) + "]", ignore_[i], {0, 255}));
  }

  RETURN_IF_NOT_OK(ValidateProbability("RandomAutoContrast", probability_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomAutoContrastOperation::Build() {
  std::shared_ptr<RandomAutoContrastOp> tensor_op =
    std::make_shared<RandomAutoContrastOp>(cutoff_, ignore_, probability_);
  return tensor_op;
}

Status RandomAutoContrastOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["cutoff"] = cutoff_;
  args["ignore"] = ignore_;
  args["prob"] = probability_;
  *out_json = args;
  return Status::OK();
}

Status RandomAutoContrastOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "cutoff", kRandomAutoContrastOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "ignore", kRandomAutoContrastOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "prob", kRandomAutoContrastOperation));
  float cutoff = op_params["cutoff"];
  std::vector<uint32_t> ignore = op_params["ignore"];
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::RandomAutoContrastOperation>(cutoff, ignore, prob);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
