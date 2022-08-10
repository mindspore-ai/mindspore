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
#include "minddata/dataset/kernels/ir/vision/posterize_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/posterize_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// PosterizeOperation
PosterizeOperation::PosterizeOperation(uint8_t bits) : bits_(bits) {}

PosterizeOperation::~PosterizeOperation() = default;

Status PosterizeOperation::ValidateParams() {
  constexpr uint8_t kMinimumBitValue = 0;
  constexpr uint8_t kMaximumBitValue = 8;

  if (bits_ < kMinimumBitValue || bits_ > kMaximumBitValue) {
    std::string err_msg = "Posterize: bits is out of range [0, 8], got: " + std::to_string(bits_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> PosterizeOperation::Build() {
  std::shared_ptr<PosterizeOp> tensor_op = std::make_shared<PosterizeOp>(bits_);
  return tensor_op;
}

Status PosterizeOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["bits"] = bits_;
  return Status::OK();
}

Status PosterizeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "bits", kPosterizeOperation));
  uint8_t bits_ = op_params["bits"];
  *operation = std::make_shared<vision::PosterizeOperation>(bits_);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
