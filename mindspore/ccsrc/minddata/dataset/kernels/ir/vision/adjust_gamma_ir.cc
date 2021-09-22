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
#include "minddata/dataset/kernels/ir/vision/adjust_gamma_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/adjust_gamma_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID

// AdjustGammaOperation
AdjustGammaOperation::AdjustGammaOperation(float gamma, float gain) : gamma_(gamma), gain_(gain) {}

Status AdjustGammaOperation::ValidateParams() {
  // gamma
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("AdjustGamma", "gamma", gamma_));
  return Status::OK();
}

std::shared_ptr<TensorOp> AdjustGammaOperation::Build() {
  std::shared_ptr<AdjustGammaOp> tensor_op = std::make_shared<AdjustGammaOp>(gamma_, gain_);
  return tensor_op;
}

Status AdjustGammaOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["gamma"] = gamma_;
  args["gain"] = gain_;
  *out_json = args;
  return Status::OK();
}

Status AdjustGammaOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("gamma") != op_params.end(), "Failed to find gamma");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("gain") != op_params.end(), "Failed to find gain");
  float gamma = op_params["gamma"];
  float gain = op_params["gain"];
  *operation = std::make_shared<vision::AdjustGammaOperation>(gamma, gain);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
