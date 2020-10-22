/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/depth_to_space.h"
#include "src/common/common.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/depth_to_space_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateDepthToSpaceParameter(const mindspore::lite::PrimitiveC *primitive) {
  DepthToSpaceParameter *depth_space_param =
    reinterpret_cast<DepthToSpaceParameter *>(malloc(sizeof(DepthToSpaceParameter)));
  if (depth_space_param == nullptr) {
    MS_LOG(ERROR) << "malloc DepthToSpaceParameter failed.";
    return nullptr;
  }
  memset(depth_space_param, 0, sizeof(DepthToSpaceParameter));
  auto param = reinterpret_cast<mindspore::lite::DepthToSpace *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  depth_space_param->op_parameter_.type_ = primitive->Type();
  depth_space_param->block_size_ = param->GetBlockSize();
  return reinterpret_cast<OpParameter *>(depth_space_param);
}

Registry DepthToSpaceParameterRegistry(schema::PrimitiveType_DepthToSpace, PopulateDepthToSpaceParameter);

}  // namespace lite

}  // namespace mindspore
