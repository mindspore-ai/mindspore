/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/audio/ir/kernels/phase_vocoder_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/phase_vocoder_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
PhaseVocoderOperation::PhaseVocoderOperation(float rate, const std::shared_ptr<Tensor> &phase_advance)
    : rate_(rate), phase_advance_(phase_advance) {}

PhaseVocoderOperation::~PhaseVocoderOperation() = default;

Status PhaseVocoderOperation::ValidateParams() {
  const int kPhaseAdvanceRank = 2;
  const int kLastDim = -1;
  const int kLastDimSize = 1;
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("PhaseVocoder", "rate", rate_));
  CHECK_FAIL_RETURN_SYNTAX_ERROR(
    phase_advance_->Rank() == kPhaseAdvanceRank && phase_advance_->shape()[kLastDim] == kLastDimSize,
    "PhaseVocoder: invalid parameter, 'phase_advance' should be in shape of <freq, 1>.");
  return Status::OK();
}

std::string PhaseVocoderOperation::Name() const { return kPhaseVocoderOperation; }

std::shared_ptr<TensorOp> PhaseVocoderOperation::Build() {
  std::shared_ptr<PhaseVocoderOp> tensor_op = std::make_shared<PhaseVocoderOp>(rate_, phase_advance_);
  return tensor_op;
}

Status PhaseVocoderOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["rate"] = rate_;
  nlohmann::json phase_advance;
  RETURN_IF_NOT_OK(phase_advance_->to_json(&phase_advance));
  args["phase_advance"] = phase_advance;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
