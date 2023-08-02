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

#include "minddata/dataset/kernels/ir/vision/random_color_ir.h"

#include <vector>

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_color_op.h"
#endif
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandomColorOperation.
RandomColorOperation::RandomColorOperation(float t_lb, float t_ub) : t_lb_(t_lb), t_ub_(t_ub) { random_op_ = true; }

RandomColorOperation::~RandomColorOperation() = default;

std::string RandomColorOperation::Name() const { return kRandomColorOperation; }

Status RandomColorOperation::ValidateParams() {
  if (t_lb_ < 0.0 || t_ub_ < 0.0) {
    std::string err_msg =
      "RandomColor: lower bound or upper bound must be greater than or equal to 0, got 'degree'(t_lb): " +
      std::to_string(t_lb_) + ", 'degree'(t_ub): " + std::to_string(t_ub_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (t_lb_ > t_ub_) {
    std::string err_msg =
      "RandomColor: lower bound must be less or equal to upper bound, got 'degree'(t_lb): " + std::to_string(t_lb_) +
      ", 'degree'(t_ub): " + std::to_string(t_ub_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomColorOperation::Build() {
  std::shared_ptr<RandomColorOp> tensor_op = std::make_shared<RandomColorOp>(t_lb_, t_ub_);
  return tensor_op;
}

Status RandomColorOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  (*out_json)["degrees"] = std::vector<float>{t_lb_, t_ub_};
  return Status::OK();
}

Status RandomColorOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "degrees", kRandomColorOperation));
  std::vector<float> degrees = op_params["degrees"];
  CHECK_FAIL_RETURN_UNEXPECTED(degrees.size() == 2, "The number of degrees should be 2");
  float t_lb = degrees[0];
  float t_ub = degrees[1];
  *operation = std::make_shared<vision::RandomColorOperation>(t_lb, t_ub);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
