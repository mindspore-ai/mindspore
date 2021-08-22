/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>

#include "minddata/dataset/kernels/ir/vision/mixup_batch_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/mixup_batch_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// MixUpOperation
MixUpBatchOperation::MixUpBatchOperation(float alpha) : alpha_(alpha) {}

MixUpBatchOperation::~MixUpBatchOperation() = default;

std::string MixUpBatchOperation::Name() const { return kMixUpBatchOperation; }

Status MixUpBatchOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("MixUpBatch", "alpha", alpha_));
  return Status::OK();
}

std::shared_ptr<TensorOp> MixUpBatchOperation::Build() { return std::make_shared<MixUpBatchOp>(alpha_); }

Status MixUpBatchOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["alpha"] = alpha_;
  return Status::OK();
}

Status MixUpBatchOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("alpha") != op_params.end(), "Failed to find alpha");
  float alpha = op_params["alpha"];
  *operation = std::make_shared<vision::MixUpBatchOperation>(alpha);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
