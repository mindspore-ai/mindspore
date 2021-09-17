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

#include "minddata/dataset/audio/ir/kernels/biquad_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/biquad_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// BiquadOperation
BiquadOperation::BiquadOperation(float b0, float b1, float b2, float a0, float a1, float a2)
    : b0_(b0), b1_(b1), b2_(b2), a0_(a0), a1_(a1), a2_(a2) {}

Status BiquadOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalarNotZero("Biquad", "a0", a0_));
  return Status::OK();
}

std::shared_ptr<TensorOp> BiquadOperation::Build() {
  std::shared_ptr<BiquadOp> tensor_op = std::make_shared<BiquadOp>(b0_, b1_, b2_, a0_, a1_, a2_);
  return tensor_op;
}

Status BiquadOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["b0"] = b0_;
  args["b1"] = b1_;
  args["b2"] = b2_;
  args["a0"] = a0_;
  args["a1"] = a1_;
  args["a2"] = a2_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
