/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/ir/vision/decode_video_ir.h"

#include "minddata/dataset/kernels/image/decode_video_op.h"
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
// DecodeVideoOperation
DecodeVideoOperation::DecodeVideoOperation() {}

DecodeVideoOperation::~DecodeVideoOperation() = default;

std::string DecodeVideoOperation::Name() const { return kDecodeVideoOperation; }

Status DecodeVideoOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> DecodeVideoOperation::Build() { return std::make_shared<DecodeVideoOp>(); }

Status DecodeVideoOperation::to_json(nlohmann::json *out_json) { return Status::OK(); }

Status DecodeVideoOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  *operation = std::make_shared<vision::DecodeVideoOperation>();
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
