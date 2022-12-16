/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/mapper/reshape_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/log_util.h"
#include "ops/op_utils.h"
#include "nnacl/op_base.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "ops/reshape.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kInputNum = 3;
}  // namespace

STATUS ReshapeMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  if (cnode->inputs().size() != kInputNum) {
    MS_LOG(ERROR) << "Reshape input num should be " << kInputNum << ", real size: " << cnode->inputs().size();
    return RET_ERROR;
  }
  if (AttrAdjust(src_prim, value_node, cnode) != RET_OK) {
    MS_LOG(ERROR) << "Reshape attr adjust failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS ReshapeMapper::AttrAdjust(const PrimitivePtr &src_prim, const ValueNodePtr &val_node, const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(src_prim != nullptr, RET_ERROR, "src_prim is nullptr.");
  MS_CHECK_TRUE_MSG(val_node != nullptr, RET_ERROR, "val_node is nullptr.");
  ValuePtr attr_val = nullptr;
  if (src_prim->HasAttr("shape")) {
    return RET_OK;
  }

  // attr shape has been erased during input adjust
  auto shape_input = cnode->input(kInputNum - 1);
  MS_CHECK_TRUE_MSG(shape_input != nullptr, lite::RET_ERROR, "shape_input is nullptr.");
  if (!utils::isa<ParameterPtr>(shape_input)) {
    MS_LOG(INFO) << "The shape input node is not parameter, no need attr adjust.";
    return lite::RET_OK;
  }
  MS_LOG(INFO) << "Add shape attr in reshape.";
  ParameterPtr shape_param = shape_input->cast<ParameterPtr>();
  MS_CHECK_TRUE_MSG(shape_param != nullptr, lite::RET_ERROR, "ParameterPtr casts failed.");
  std::vector<int> shape = acl::GetIntParameterData(shape_param);
  ops::Reshape reshape;
  auto dst_prim = reshape.GetPrim();
  dst_prim->AddAttr("shape", MakeValue(shape));
  val_node->set_value(dst_prim);
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameReshape, ReshapeMapper)
}  // namespace lite
}  // namespace mindspore
