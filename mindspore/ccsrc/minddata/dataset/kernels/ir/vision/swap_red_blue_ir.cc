/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/ir/vision/swap_red_blue_ir.h"

#if !defined(ENABLE_ANDROID) || defined(ENABLE_CLOUD_FUSION_INFERENCE)
#include "minddata/dataset/kernels/image/swap_red_blue_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#if !defined(ENABLE_ANDROID) || defined(ENABLE_CLOUD_FUSION_INFERENCE)
// SwapRedBlueOperation.
SwapRedBlueOperation::SwapRedBlueOperation() {}

SwapRedBlueOperation::~SwapRedBlueOperation() = default;

std::string SwapRedBlueOperation::Name() const { return kSwapRedBlueOperation; }

Status SwapRedBlueOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> SwapRedBlueOperation::Build() {
  std::shared_ptr<SwapRedBlueOp> tensor_op = std::make_shared<SwapRedBlueOp>();
  return tensor_op;
}

Status SwapRedBlueOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  *operation = std::make_shared<vision::SwapRedBlueOperation>();
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
