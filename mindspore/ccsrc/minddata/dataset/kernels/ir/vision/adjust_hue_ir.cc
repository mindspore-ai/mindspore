/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/ir/vision/adjust_hue_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/adjust_hue_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// AdjustHueOperation
AdjustHueOperation::AdjustHueOperation(float hue_factor) : hue_factor_(hue_factor) {}

Status AdjustHueOperation::ValidateParams() {
  // hue_factor
  RETURN_IF_NOT_OK(ValidateScalar("AdjustHue", "hue_factor", hue_factor_, {-0.5, 0.5}, false, false));
  return Status::OK();
}

std::shared_ptr<TensorOp> AdjustHueOperation::Build() {
  std::shared_ptr<AdjustHueOp> tensor_op = std::make_shared<AdjustHueOp>(hue_factor_);
  return tensor_op;
}

Status AdjustHueOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["hue_factor"] = hue_factor_;
  *out_json = args;
  return Status::OK();
}

Status AdjustHueOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "hue_factor", kAdjustHueOperation));
  float hue_factor = op_params["hue_factor"];
  *operation = std::make_shared<vision::AdjustHueOperation>(hue_factor);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
