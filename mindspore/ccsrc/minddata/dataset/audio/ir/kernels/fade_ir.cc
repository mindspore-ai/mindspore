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
#include "minddata/dataset/audio/ir/kernels/fade_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/fade_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
FadeOperation::FadeOperation(int32_t fade_in_len, int32_t fade_out_len, FadeShape fade_shape)
    : fade_in_len_(fade_in_len), fade_out_len_(fade_out_len), fade_shape_(fade_shape) {}

Status FadeOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("Fade", "fade_in_len", fade_in_len_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("Fade", "fade_out_len", fade_out_len_));
  return Status::OK();
}

std::shared_ptr<TensorOp> FadeOperation::Build() {
  std::shared_ptr<FadeOp> tensor_op = std::make_shared<FadeOp>(fade_in_len_, fade_out_len_, fade_shape_);
  return tensor_op;
}

Status FadeOperation::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  args["fade_in_len"] = fade_in_len_;
  args["fade_out_len"] = fade_out_len_;
  args["fade_shape"] = fade_shape_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
