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

#include "minddata/dataset/kernels/ir/vision/cutmix_batch_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/cutmix_batch_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// CutMixBatchOperation
CutMixBatchOperation::CutMixBatchOperation(ImageBatchFormat image_batch_format, float alpha, float prob)
    : image_batch_format_(image_batch_format), alpha_(alpha), prob_(prob) {}

CutMixBatchOperation::~CutMixBatchOperation() = default;

std::string CutMixBatchOperation::Name() const { return kCutMixBatchOperation; }

Status CutMixBatchOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("CutMixBatch", "alpha", alpha_));
  RETURN_IF_NOT_OK(ValidateProbability("CutMixBatch", prob_));
  if (image_batch_format_ != ImageBatchFormat::kNHWC && image_batch_format_ != ImageBatchFormat::kNCHW) {
    std::string err_msg = "CutMixBatch: Invalid ImageBatchFormat, check input value of enum.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> CutMixBatchOperation::Build() {
  std::shared_ptr<CutMixBatchOp> tensor_op = std::make_shared<CutMixBatchOp>(image_batch_format_, alpha_, prob_);
  return tensor_op;
}

Status CutMixBatchOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["image_batch_format"] = image_batch_format_;
  args["alpha"] = alpha_;
  args["prob"] = prob_;
  *out_json = args;
  return Status::OK();
}

Status CutMixBatchOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("image_batch_format") != op_params.end(),
                               "Failed to find image_batch_format");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("alpha") != op_params.end(), "Failed to find alpha");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("prob") != op_params.end(), "Failed to find prob");
  ImageBatchFormat image_batch = static_cast<ImageBatchFormat>(op_params["image_batch_format"]);
  float alpha = op_params["alpha"];
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::CutMixBatchOperation>(image_batch, alpha, prob);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
