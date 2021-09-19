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
#include "minddata/dataset/kernels/ir/vision/gaussian_blur_ir.h"

#include "minddata/dataset/kernels/image/gaussian_blur_op.h"
#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
GaussianBlurOperation::GaussianBlurOperation(const std::vector<int32_t> kernel_size, const std::vector<float> sigma)
    : kernel_size_(kernel_size), sigma_(sigma) {}

GaussianBlurOperation::~GaussianBlurOperation() = default;

std::string GaussianBlurOperation::Name() const { return kGaussianBlurOperation; }

Status GaussianBlurOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorSize("GaussianBlur", kernel_size_));
  RETURN_IF_NOT_OK(ValidateVectorOdd("GaussianBlur", "kernel_size", kernel_size_));
  RETURN_IF_NOT_OK(ValidateVectorSigma("GaussianBlur", sigma_));
  return Status::OK();
}

std::shared_ptr<TensorOp> GaussianBlurOperation::Build() {
  int32_t kernel_x = kernel_size_[0];
  int32_t kernel_y = kernel_size_[0];
  // User has specified kernel_y.
  if (kernel_size_.size() == 2) {
    kernel_y = kernel_size_[1];
  }

  float sigma_x = sigma_[0] <= 0 ? kernel_x * 0.15 + 0.35 : sigma_[0];
  float sigma_y = sigma_x;

  // User has specified sigma_y.
  if (sigma_.size() == 2) {
    sigma_y = sigma_[1] <= 0 ? kernel_y * 0.15 + 0.35 : sigma_[1];
  }
  std::shared_ptr<GaussianBlurOp> tensor_op = std::make_shared<GaussianBlurOp>(kernel_x, kernel_y, sigma_x, sigma_y);
  return tensor_op;
}

Status GaussianBlurOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["kernel_size"] = kernel_size_;
  args["sigma"] = sigma_;
  *out_json = args;
  return Status::OK();
}

Status GaussianBlurOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("kernel_size") != op_params.end(), "Failed to find kernel_size");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("sigma") != op_params.end(), "Failed to find sigma");
  std::vector<int32_t> kernel_size = op_params["kernel_size"];
  std::vector<float> sigma = op_params["sigma"];
  *operation = std::make_shared<vision::GaussianBlurOperation>(kernel_size, sigma);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
