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
#include "nnacl/depth_to_space_parameter.h"
#include "ops/depth_to_space.h"
using mindspore::ops::kNameDepthToSpace;
using mindspore::schema::PrimitiveType_DepthToSpace;
namespace mindspore {
namespace lite {
OpParameter *PopulateDepthToSpaceOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<DepthToSpaceParameter *>(PopulateOpParameter<DepthToSpaceParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new DepthToSpaceParameter failed.";
    return nullptr;
  }

  auto op = dynamic_cast<ops::DepthToSpace *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to DepthToSpace failed";
    free(param);
    return nullptr;
  }

  param->mode_ = 0;
  auto mode = op->get_mode();
  if (mode == "CRD") {
    param->mode_ = 1;
  }

  auto block_size = static_cast<int>(op->get_block_size());
  if (block_size < C2NUM) {
    MS_LOG(ERROR) << "invalid block_size value: " << block_size;
    free(param);
    return nullptr;
  }
  param->block_size_ = block_size;
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameDepthToSpace, PrimitiveType_DepthToSpace, PopulateDepthToSpaceOpParameter)
}  // namespace lite
}  // namespace mindspore
