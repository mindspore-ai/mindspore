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

#define USE_DEPRECATED_API
#include "tools/optimizer/graph/input_data_type_trans_pass.h"
#include <vector>
#include <memory>
#include "ops/op_utils.h"
#include "ops/cast.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/tensor.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"
#include "src/common/utils.h"

namespace mindspore::opt {
namespace {
constexpr size_t kNotEqualMinIndex = 3;
}  // namespace

CNodePtr InputDTypeTransPass::GenCastNode(const FuncGraphPtr &graph, const AnfNodePtr &input_node,
                                          const std::string &cnode_name) {
  MS_CHECK_TRUE_RET(graph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(input_node != nullptr, nullptr);
  // auto new_cast = std::make_shared<mindspore::ops::Cast>();
  ops::Cast cast_node;
  auto new_cast_c = cast_node.GetPrim();
  MS_CHECK_TRUE_MSG(new_cast_c != nullptr, nullptr, "new_cast_c is nullptr");
  ValueNodePtr value_node = NewValueNode(new_cast_c);
  MS_CHECK_TRUE_MSG(value_node != nullptr, nullptr, "NewValueNode Failed");

  auto param_node =
    opt::BuildIntValueParameterNode(graph, static_cast<int32_t>(src_input_data_type_), cnode_name + "_type");

  auto cast_cnode = graph->NewCNode({value_node});
  MS_CHECK_TRUE_MSG(cast_cnode != nullptr, nullptr, "new_cnode is nullptr");
  cast_cnode->set_fullname_with_scope(cnode_name);
  auto manager = Manage(graph);
  (void)manager->Replace(input_node, cast_cnode);
  manager->AddEdge(cast_cnode, input_node);
  manager->AddEdge(cast_cnode, param_node);
  return cast_cnode;
}

STATUS InputDTypeTransPass::HandleGraphInput(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto manager = graph->manager();
  MS_ASSERT(manager != nullptr);
  auto graph_inputs = graph->get_inputs();
  for (const auto &input : graph_inputs) {
    TypeId input_data_type;
    if (GetDataTypeFromAnfNode(input, &input_data_type) != RET_OK) {
      MS_LOG(ERROR) << "get input node data type failed." << input->fullname_with_scope();
      return RET_ERROR;
    }
    if (input_data_type == static_cast<TypeId>(src_input_data_type_)) {
      auto input_node = input->cast<ParameterPtr>();
      MS_ASSERT(input_node != nullptr);
      auto abstract = input_node->abstract();
      MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_NULL_PTR, "abstract is nullptr");

      if (!utils::isa<abstract::AbstractTensorPtr>(abstract)) {
        MS_LOG(ERROR) << "abstract is not AbstractTensor";
        return RET_ERROR;
      }
      auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract);
      MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, RET_ERROR, "Cast to abstract tensor failed!");
      auto element = abstract_tensor->element();
      element->set_type(TypeIdToType(kNumberTypeInt32));

      auto cast_cnode = GenCastNode(graph, input, input->fullname_with_scope() + "_post_cast");
      cast_cnode->set_abstract(abstract->Clone());
      auto cast_abstract = cast_cnode->abstract();
      MS_ASSERT(cast_abstract != nullptr);
      cast_abstract->set_value(std::make_shared<AnyValue>());
    }
  }
  return lite::RET_OK;
}

bool InputDTypeTransPass::Run(const FuncGraphPtr &graph) {
  if (dst_input_data_type_ == src_input_data_type_) {
    return true;
  }
  MS_ASSERT(graph != nullptr);
  auto manager = Manage(graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  if (HandleGraphInput(graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "transfer graph input format from nhwc to nchw failed.";
    return false;
  }
  return true;
}
}  // namespace mindspore::opt
