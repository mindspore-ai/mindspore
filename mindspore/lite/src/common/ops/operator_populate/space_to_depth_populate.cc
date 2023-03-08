/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/space_to_depth_parameter.h"
#include "ops/space_to_depth.h"
using mindspore::ops::kNameSpaceToDepth;
using mindspore::schema::PrimitiveType_SpaceToDepth;

namespace mindspore {
namespace lite {
OpParameter *PopulateSpaceToDepthOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<SpaceToDepthParameter *>(PopulateOpParameter<SpaceToDepthParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new SpaceToDepthParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::SpaceToDepth *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not TopKFusion.";
    return nullptr;
  }

  param->block_size_ = op->get_block_size();
  if (param->block_size_ < C2NUM) {
    MS_LOG(ERROR) << "invalid block_size value: " << param->block_size_;
    free(param);
    return nullptr;
  }
  if (op->get_format() != NHWC) {
    MS_LOG(ERROR) << "Currently only NHWC format is supported.";
    free(param);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameSpaceToDepth, PrimitiveType_SpaceToDepth, PopulateSpaceToDepthOpParameter)
}  // namespace lite
}  // namespace mindspore
