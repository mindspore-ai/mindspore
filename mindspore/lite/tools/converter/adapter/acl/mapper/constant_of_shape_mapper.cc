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

#include "tools/converter/adapter/acl/mapper/constant_of_shape_mapper.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/op_utils.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
STATUS ConstantOfShapeMapper::Mapper(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto func_graph = cnode->func_graph();
  CHECK_NULL_RETURN(func_graph);
  auto prim = ops::GetOperator<ops::ConstantOfShape>(cnode->input(0));
  CHECK_NULL_RETURN(prim);
  ParameterPtr value_param = nullptr;
  auto values = prim->get_value();
  auto data_type = prim->get_data_type();
  switch (data_type) {
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      if (values.size() == 1) {
        value_param =
          opt::BuildFloatValueParameterNode(func_graph, values[0], cnode->fullname_with_scope() + "_value", true);
      } else {
        value_param = opt::BuildFloatVecParameterNode(func_graph, values, cnode->fullname_with_scope() + "_values");
      }
      break;
    case kNumberTypeInt:
    case kNumberTypeInt32: {
      if (values.size() == 1) {
        value_param =
          opt::BuildIntValueParameterNode(func_graph, values[0], cnode->fullname_with_scope() + "_value", true);
      } else {
        std::vector<int> dst_values;
        std::transform(values.begin(), values.end(), std::back_inserter(dst_values),
                       [](float ele) { return static_cast<int>(ele); });
        value_param = opt::BuildIntVecParameterNode(func_graph, dst_values, cnode->fullname_with_scope() + "_values");
      }
    } break;
    default:
      MS_LOG(ERROR) << "Unsupported data type: " << data_type;
      return RET_ERROR;
  }
  if (value_param == nullptr) {
    MS_LOG(ERROR) << "Build parameter node failed.";
    return RET_ERROR;
  }
  auto dst_prim = std::make_shared<acl::FillV1>();
  CHECK_NULL_RETURN(dst_prim);
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  CHECK_NULL_RETURN(value_node);
  value_node->set_value(dst_prim);
  auto inputs = cnode->inputs();
  inputs.push_back(value_param);
  cnode->set_inputs(inputs);
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameConstantOfShape, ConstantOfShapeMapper)
}  // namespace lite
}  // namespace mindspore
