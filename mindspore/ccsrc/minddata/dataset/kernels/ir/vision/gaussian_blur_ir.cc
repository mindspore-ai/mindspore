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
#include "minddata/dataset/kernels/ir/vision/gaussian_blur_ir.h"

#include "minddata/dataset/kernels/image/gaussian_blur_op.h"
#if !defined(BUILD_LITE) && defined(ENABLE_D)
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_gaussian_blur_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
constexpr int sigma_size = 2;

GaussianBlurOperation::GaussianBlurOperation(const std::vector<int32_t> &kernel_size, const std::vector<float> &sigma,
                                             const std::string &device_target)
    : kernel_size_(kernel_size), sigma_(sigma), device_target_(device_target) {}

GaussianBlurOperation::~GaussianBlurOperation() = default;

std::string GaussianBlurOperation::Name() const { return kGaussianBlurOperation; }

Status GaussianBlurOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorSize("GaussianBlur", kernel_size_));
  RETURN_IF_NOT_OK(ValidateVectorOdd("GaussianBlur", "kernel_size", kernel_size_));
  RETURN_IF_NOT_OK(ValidateVectorSigma("GaussianBlur", sigma_));
  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "GaussianBlur: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> GaussianBlurOperation::Build() {
  int32_t kernel_x = kernel_size_[0];
  int32_t kernel_y = kernel_size_[0];
  // User has specified kernel_y.
  if (kernel_size_.size() == 2) {
    kernel_y = kernel_size_[1];
  }

  float sigma_x = sigma_[0] <= 0.0 ? static_cast<float>(kernel_x) * 0.15F + 0.35F : sigma_[0];
  float sigma_y = sigma_x;

  // User has specified sigma_y.
  if (sigma_.size() == sigma_size) {
    sigma_y = sigma_[1] <= 0.0 ? static_cast<float>(kernel_y) * 0.15F + 0.35F : sigma_[1];
  }

  if (device_target_ == "CPU") {
    std::shared_ptr<GaussianBlurOp> tensor_op = std::make_shared<GaussianBlurOp>(kernel_x, kernel_y, sigma_x, sigma_y);
    return tensor_op;
#if !defined(BUILD_LITE) && defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    std::shared_ptr<DvppGaussianBlurOp> dvpp_tensor_op =
      std::make_shared<DvppGaussianBlurOp>(kernel_x, kernel_y, sigma_x, sigma_y);
    return dvpp_tensor_op;
#endif
  } else {
    MS_LOG(ERROR) << "GaussianBlur: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

Status GaussianBlurOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["kernel_size"] = kernel_size_;
  args["sigma"] = sigma_;
  args["device_target"] = device_target_;
  *out_json = args;
  return Status::OK();
}

Status GaussianBlurOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "kernel_size", kGaussianBlurOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "sigma", kGaussianBlurOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kGaussianBlurOperation));
  std::vector<int32_t> kernel_size = op_params["kernel_size"];
  std::vector<float> sigma = op_params["sigma"];
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<vision::GaussianBlurOperation>(kernel_size, sigma, device_target);
  return Status::OK();
}

MapTargetDevice GaussianBlurOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "GaussianBlur: Invalid device target. It's not CPU or Ascend.";
    return MapTargetDevice::kInvalid;
  }
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
