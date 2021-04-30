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

#include "minddata/dataset/kernels/ir/vision/random_sharpness_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_sharpness_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {

namespace vision {

#ifndef ENABLE_ANDROID

// Function to create RandomSharpness.
RandomSharpnessOperation::RandomSharpnessOperation(std::vector<float> degrees)
    : TensorOperation(true), degrees_(degrees) {}

RandomSharpnessOperation::~RandomSharpnessOperation() = default;

std::string RandomSharpnessOperation::Name() const { return kRandomSharpnessOperation; }

Status RandomSharpnessOperation::ValidateParams() {
  if (degrees_.size() != 2 || degrees_[0] < 0 || degrees_[1] < 0) {
    std::string err_msg = "RandomSharpness: degrees must be a vector of two values and greater than or equal to 0.";
    MS_LOG(ERROR) << "RandomSharpness: degrees must be a vector of two values and greater than or equal to 0, got: "
                  << degrees_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (degrees_[1] < degrees_[0]) {
    std::string err_msg = "RandomSharpness: degrees must be in the format of (min, max).";
    MS_LOG(ERROR) << "RandomSharpness: degrees must be in the format of (min, max), got: " << degrees_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomSharpnessOperation::Build() {
  std::shared_ptr<RandomSharpnessOp> tensor_op = std::make_shared<RandomSharpnessOp>(degrees_[0], degrees_[1]);
  return tensor_op;
}

Status RandomSharpnessOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["degrees"] = degrees_;
  return Status::OK();
}

#endif

}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
