/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/const_input_to_attr.h"

#include <vector>
#include "include/backend/common/pass_manager/helper.h"
#include "include/utils/utils.h"
#include "ops_utils/op_utils.h"
#include "utils/anf_utils.h"
#include "utils/log_adapter.h"
#include "include/utils/anfalgo.h"

namespace mindspore {
namespace opt {
CNodePtr ConstInputToAttr(const CNodePtr &cnode, const mindspore::HashSet<size_t> &input_attrs) {
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> new_inputs;
  auto primitive = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive = primitive->Clone();
  auto inputs = cnode->inputs();
  new_inputs.push_back(inputs[0]);
  bool need_update = false;
  auto input_names = primitive->GetAttr(kAttrInputNames);
  std::vector<std::string> input_names_vec;
  if (input_names != nullptr) {
    input_names_vec = GetValue<std::vector<std::string>>(input_names);
  }
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  for (size_t i = 0; i < inputs.size() - 1; ++i) {
    auto input_node = inputs[i + 1];
    MS_EXCEPTION_IF_NULL(input_node);
    if (IsPrimitiveCNode(input_node, prim::kPrimDepend)) {
      input_node = AnfUtils::VisitKernel(input_node, 0).first;
    }
    if (input_attrs.find(i) != input_attrs.end() && input_node->isa<ValueNode>() && !HasAbstractMonad(input_node)) {
      auto input_name = ops::GetInputNameByIndex(op_name, i);
      if (op_name == "Cast" && i == 1) {
        input_name = "dst_type";
      }
      if (input_name == "") {
        // operators that are not developed by yaml
        if (input_names_vec.empty()) {
          MS_LOG(DEBUG) << "input_names are nullptr in cnode[" + cnode->DebugString() + "]";
          return cnode;
        }
        if (i >= input_names_vec.size()) {
          MS_LOG(EXCEPTION) << "Index " << i << " is larger than input names size [" << input_names_vec.size() << "]";
        }
        input_name = input_names_vec[i];
      }
      auto value_node = input_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      MS_LOG(DEBUG) << "start erase input[" << i << "] of cnode[" + cnode->DebugString() + "]";
      auto value = value_node->value();
      if (value->isa<tensor::Tensor>()) {
        auto tensor = value->cast<tensor::TensorPtr>();
        if (tensor->unsafe_data() == nullptr && !tensor->has_user_data(kTensorValueIsEmpty)) {
          need_update = false;
          break;
        }
      }
      primitive->set_attr(input_name, value);
      need_update = true;
    } else {
      new_inputs.push_back(inputs[i + 1]);
    }
  }
  if (need_update) {
    // Update cnode's inputs
    new_inputs[0] = NewValueNode(primitive);
    auto graph = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(graph);
    auto new_cnode = NewCNode(new_inputs, graph, {cnode});
    MS_EXCEPTION_IF_NULL(new_cnode);
    new_cnode->set_abstract(cnode->abstract());
    new_cnode->set_scope(cnode->scope());
    new_cnode->set_primal_attrs(cnode->primal_attrs());
    new_cnode->set_attrs(cnode->attrs());
    auto kernel_graph = graph->cast<KernelGraphPtr>();
    if (kernel_graph != nullptr) {
      kernel_graph->FrontBackendlMapUpdate(cnode, new_cnode);
    }
    return new_cnode;
  }
  return cnode;
}
}  // namespace opt
}  // namespace mindspore
