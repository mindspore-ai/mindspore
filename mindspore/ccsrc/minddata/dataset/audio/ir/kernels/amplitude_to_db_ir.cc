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

#include "minddata/dataset/audio/ir/kernels/amplitude_to_db_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/amplitude_to_db_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// AmplitudeToDBOperation
AmplitudeToDBOperation::AmplitudeToDBOperation(ScaleType stype, float ref_value, float amin, float top_db)
    : stype_(stype), ref_value_(ref_value), amin_(amin), top_db_(top_db) {}

AmplitudeToDBOperation::~AmplitudeToDBOperation() = default;

std::string AmplitudeToDBOperation::Name() const { return kAmplitudeToDBOperation; }

Status AmplitudeToDBOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("AmplitudeToDB", "top_db", top_db_));
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("AmplitudeToDB", "amin", amin_));
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("AmplitudeToDB", "ref_value", ref_value_));

  return Status::OK();
}

std::shared_ptr<TensorOp> AmplitudeToDBOperation::Build() {
  std::shared_ptr<AmplitudeToDBOp> tensor_op = std::make_shared<AmplitudeToDBOp>(stype_, ref_value_, amin_, top_db_);
  return tensor_op;
}

Status AmplitudeToDBOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["stype"] = stype_;
  args["ref_value"] = ref_value_;
  args["amin"] = amin_;
  args["top_db"] = top_db_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
