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

#include "minddata/dataset/audio/ir/kernels/complex_norm_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/complex_norm_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
ComplexNormOperation::ComplexNormOperation(float power) : power_(power) {}

ComplexNormOperation::~ComplexNormOperation() = default;

Status ComplexNormOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("ComplexNorm", "power", power_));
  return Status::OK();
}

Status ComplexNormOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["power"] = power_;
  *out_json = args;
  return Status::OK();
}

std::shared_ptr<TensorOp> ComplexNormOperation::Build() {
  std::shared_ptr<ComplexNormOp> tensor_op = std::make_shared<ComplexNormOp>(power_);
  return tensor_op;
}

std::string ComplexNormOperation::Name() const { return kComplexNormOperation; }
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
