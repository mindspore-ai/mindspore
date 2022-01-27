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

#include "minddata/dataset/audio/ir/kernels/mask_along_axis_iid_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/mask_along_axis_iid_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
MaskAlongAxisIIDOperation::MaskAlongAxisIIDOperation(int32_t mask_param, float mask_value, int32_t axis)
    : mask_param_(mask_param), mask_value_(mask_value), axis_(axis) {
  random_op_ = true;
}

MaskAlongAxisIIDOperation::~MaskAlongAxisIIDOperation() = default;

Status MaskAlongAxisIIDOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("MaskAlongAxisIID", "mask_param", mask_param_));
  RETURN_IF_NOT_OK(ValidateScalarValue("MaskAlongAxisIID", "axis", axis_, {1, 2}));
  return Status::OK();
}

std::string MaskAlongAxisIIDOperation::Name() const { return kMaskAlongAxisIIDOperation; }

std::shared_ptr<TensorOp> MaskAlongAxisIIDOperation::Build() {
  std::shared_ptr<MaskAlongAxisIIDOp> tensor_op = std::make_shared<MaskAlongAxisIIDOp>(mask_param_, mask_value_, axis_);
  return tensor_op;
}

Status MaskAlongAxisIIDOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["mask_param"] = mask_param_;
  args["mask_value"] = mask_value_;
  args["axis"] = axis_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
