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
#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
/* ####################################### Validator Functions ############################################ */
Status ValidateProbability(const std::string &op_name, const float probability) {
  if (probability < 0.0 || probability > 1.0) {
    std::string err_msg = op_name + ": probability must be between 0.0 and 1.0, got: " + std::to_string(probability);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

Status ValidateIntScalarPositive(const std::string &op_name, const std::string &scalar_name, int32_t scalar) {
  RETURN_IF_NOT_OK(ValidateScalar(op_name, scalar_name, scalar, {0}, true));
  return Status::OK();
}

Status ValidateFloatScalarPositive(const std::string &op_name, const std::string &scalar_name, float scalar) {
  RETURN_IF_NOT_OK(ValidateScalar(op_name, scalar_name, scalar, {0}, true));
  return Status::OK();
}

Status ValidateVectorFillvalue(const std::string &op_name, const std::vector<uint8_t> &fill_value) {
  if (fill_value.empty() || (fill_value.size() != 1 && fill_value.size() != 3)) {
    std::string err_msg =
      op_name + ": fill_value expecting size 1 or 3, got fill_value.size(): " + std::to_string(fill_value.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // Note that fill_value need to be in range [0, 255],
  // but we omit the check since its type is uint8_t
  return Status::OK();
}

Status ValidateVectorColorAttribute(const std::string &op_name, const std::string &attr_name,
                                    const std::vector<float> &attr, const std::vector<float> &range) {
  if (attr.empty() || attr.size() > 2) {
    std::string err_msg = op_name + ":" + attr_name + " expecting size 1 or 2, but got: " + std::to_string(attr.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (auto &attr_val : attr) {
    RETURN_IF_NOT_OK(ValidateScalar(op_name, attr_name, attr_val, range, false, false));
  }
  if (attr.size() == 2 && (attr[0] > attr[1])) {
    std::string err_msg = op_name + ":" + attr_name +
                          " lower bound must be less or equal to upper bound, got lb: " + std::to_string(attr[0]) +
                          ", ub: " + std::to_string(attr[1]);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

Status ValidateVectorMeanStd(const std::string &op_name, const std::vector<float> &mean,
                             const std::vector<float> &std) {
  if (mean.size() != 3) {
    std::string err_msg = op_name + ": mean expecting size 3, got size: " + std::to_string(mean.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (std.size() != 3) {
    std::string err_msg = op_name + ": std expecting size 3, got size: " + std::to_string(std.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // check std/mean value
  for (int32_t i = 0; i < std.size(); ++i) {
    RETURN_IF_NOT_OK(ValidateScalar(op_name, "mean", mean[i], {0.0, 255.0}, false, false));
    RETURN_IF_NOT_OK(ValidateScalar(op_name, "std", std[i], {0.0, 255.0}, true, false));
  }

  return Status::OK();
}

Status ValidateVectorPadding(const std::string &op_name, const std::vector<int32_t> &padding) {
  if (padding.empty() || padding.size() == 3 || padding.size() > 4) {
    std::string err_msg = op_name + ": padding expecting size 1, 2 or 4, got size: " + std::to_string(padding.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (const auto &pad_val : padding) {
    RETURN_IF_NOT_OK(ValidateScalar(op_name, "padding", pad_val, {0, INT_MAX}, false, false));
  }

  return Status::OK();
}

Status ValidateVectorPositive(const std::string &op_name, const std::string &vec_name,
                              const std::vector<int32_t> &vec) {
  for (const auto &vec_val : vec) {
    RETURN_IF_NOT_OK(ValidateScalar(op_name, vec_name, vec_val, {0}, true));
  }

  return Status::OK();
}

Status ValidateVectorNonNegative(const std::string &op_name, const std::string &vec_name,
                                 const std::vector<int32_t> &vec) {
  for (const auto &vec_val : vec) {
    RETURN_IF_NOT_OK(ValidateScalar(op_name, vec_name, vec_val, {0}, false));
  }

  return Status::OK();
}

Status ValidateVectorSize(const std::string &op_name, const std::vector<int32_t> &size) {
  if (size.empty() || size.size() > 2) {
    std::string err_msg = op_name + ": size expecting size 2, got size.size(): " + std::to_string(size.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (const auto &size_val : size) {
    RETURN_IF_NOT_OK(ValidateScalar(op_name, "size", size_val, {0, INT_MAX}, true, false));
  }

  return Status::OK();
}

Status ValidateVectorScale(const std::string &op_name, const std::vector<float> &scale) {
  if (scale.size() != 2) {
    std::string err_msg = op_name + ": scale expecting size 2, got scale.size(): " + std::to_string(scale.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  RETURN_IF_NOT_OK(ValidateScalar(op_name, "scale", scale[0], {0}, false));
  RETURN_IF_NOT_OK(ValidateScalar(op_name, "scale", scale[1], {0}, true));
  if (scale[1] < scale[0]) {
    std::string err_msg = op_name + ": scale must be in the format of (min, max).";
    MS_LOG(ERROR) << op_name + ": scale must be in the format of (min, max), but got: " << scale;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

Status ValidateVectorRatio(const std::string &op_name, const std::vector<float> &ratio) {
  if (ratio.size() != 2) {
    std::string err_msg = op_name + ": ratio expecting size 2, got ratio.size(): " + std::to_string(ratio.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  RETURN_IF_NOT_OK(ValidateScalar(op_name, "ratio", ratio[0], {0}, true));
  RETURN_IF_NOT_OK(ValidateScalar(op_name, "ratio", ratio[1], {0}, true));
  if (ratio[1] < ratio[0]) {
    std::string err_msg = op_name + ": ratio must be in the format of (min, max).";
    MS_LOG(ERROR) << op_name + ": ratio must be in the format of (min, max), but got: " << ratio;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

Status ValidateVectorTransforms(const std::string &op_name,
                                const std::vector<std::shared_ptr<TensorOperation>> &transforms) {
  if (transforms.empty()) {
    std::string err_msg = op_name + ": transform list must not be empty.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (int32_t i = 0; i < transforms.size(); ++i) {
    if (transforms[i] == nullptr) {
      std::string err_msg =
        op_name + ": transform ops must not be null, got transform[" + std::to_string(i) + "] == nullptr.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    } else {
      RETURN_IF_NOT_OK(transforms[i]->ValidateParams());
    }
  }

  return Status::OK();
}

bool CmpFloat(const float a, const float b, float epsilon) { return (std::fabs(a - b) < epsilon); }
}  // namespace dataset
}  // namespace mindspore
