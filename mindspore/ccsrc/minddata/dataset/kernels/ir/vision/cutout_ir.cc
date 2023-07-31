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

#include "minddata/dataset/kernels/ir/vision/cutout_ir.h"

#include <algorithm>
#include <vector>

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/cut_out_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
CutOutOperation::CutOutOperation(int32_t length, int32_t num_patches, bool is_hwc)
    : length_(length), num_patches_(num_patches), is_hwc_(is_hwc) {}

CutOutOperation::~CutOutOperation() = default;

std::string CutOutOperation::Name() const { return kCutOutOperation; }

Status CutOutOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("CutOut", "length", length_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("CutOut", "num_patches", num_patches_));
  return Status::OK();
}

std::shared_ptr<TensorOp> CutOutOperation::Build() {
  std::vector<uint8_t> fill;
  std::shared_ptr<CutOutOp> tensor_op =
    std::make_shared<CutOutOp>(length_, length_, num_patches_, false, fill, is_hwc_);
  return tensor_op;
}

Status CutOutOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["length"] = length_;
  args["num_patches"] = num_patches_;
  args["is_hwc"] = is_hwc_;
  *out_json = args;
  return Status::OK();
}

Status CutOutOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "length", kCutOutOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "num_patches", kCutOutOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "is_hwc", kCutOutOperation));
  int32_t length = op_params["length"];
  int32_t num_patches = op_params["num_patches"];
  bool is_hwc = op_params["is_hwc"];
  *operation = std::make_shared<vision::CutOutOperation>(length, num_patches, is_hwc);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
