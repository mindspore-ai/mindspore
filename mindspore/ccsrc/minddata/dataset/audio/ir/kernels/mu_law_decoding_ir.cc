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
#include "minddata/dataset/audio/ir/kernels/mu_law_decoding_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/mu_law_decoding_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
MuLawDecodingOperation::MuLawDecodingOperation(int quantization_channels)
    : quantization_channels_(quantization_channels) {}

MuLawDecodingOperation::~MuLawDecodingOperation() = default;

Status MuLawDecodingOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("MuLawEncoding", "quantization_channels", quantization_channels_));
  return Status::OK();
}

Status MuLawDecodingOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["quantization_channels"] = quantization_channels_;
  *out_json = args;
  return Status::OK();
}

std::shared_ptr<TensorOp> MuLawDecodingOperation::Build() {
  std::shared_ptr<MuLawDecodingOp> tensor_op = std::make_shared<MuLawDecodingOp>(quantization_channels_);
  return tensor_op;
}

std::string MuLawDecodingOperation::Name() const { return kMuLawDecodingOperation; }
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
