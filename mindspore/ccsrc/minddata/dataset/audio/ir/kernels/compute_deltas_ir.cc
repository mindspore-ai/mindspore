/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/audio/ir/kernels/compute_deltas_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/compute_deltas_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
ComputeDeltasOperation::ComputeDeltasOperation(int32_t win_length, BorderType pad_mode)
    : win_length_(win_length), pad_mode_(pad_mode) {}

std::shared_ptr<TensorOp> ComputeDeltasOperation::Build() {
  return std::make_shared<ComputeDeltasOp>(win_length_, pad_mode_);
}

Status ComputeDeltasOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["win_length"] = win_length_;
  args["pad_mode"] = pad_mode_;
  *out_json = args;
  return Status::OK();
}

Status ComputeDeltasOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalar("ComputeDeltas", "win_length", win_length_, {3}, false));
  if (pad_mode_ != BorderType::kConstant && pad_mode_ != BorderType::kEdge && pad_mode_ != BorderType::kReflect &&
      pad_mode_ != BorderType::kSymmetric) {
    std::string err_msg = "ComputeDeltas: invalid pad_mode value, check the optional value of BorderType.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
