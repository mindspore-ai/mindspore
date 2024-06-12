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
#include "minddata/dataset/kernels/ir/vision/invert_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/invert_op.h"
#endif
#if !defined(BUILD_LITE) && defined(ENABLE_D)
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_invert_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// InvertOperation
InvertOperation::InvertOperation(const std::string &device_target) : device_target_(device_target) {}

InvertOperation::~InvertOperation() = default;

std::string InvertOperation::Name() const { return kInvertOperation; }

Status InvertOperation::ValidateParams() {
  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "Invert: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> InvertOperation::Build() {
  if (device_target_ == "CPU") {
    return std::make_shared<InvertOp>();
#if !defined(BUILD_LITE) && defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    std::shared_ptr<DvppInvertOp> dvpp_tensor_op = std::make_shared<DvppInvertOp>();
    return dvpp_tensor_op;
#endif
  } else {
    MS_LOG(ERROR) << "Invert: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

Status InvertOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  (*out_json)["device_target"] = device_target_;
  return Status::OK();
}

Status InvertOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kInvertOperation));
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<vision::InvertOperation>(device_target);
  return Status::OK();
}

MapTargetDevice InvertOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "Invert: Invalid device target. It's not CPU or Ascend.";
  }
  return MapTargetDevice::kInvalid;
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
