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
#include "plugin/device/ascend/optimizer/create_node_helper.h"

#include <memory>
#include <vector>
#include <set>
#include "plugin/device/ascend/optimizer/mindir/reg_ascend_vm_op_adaptation_info.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"

namespace mindspore::opt {
AnfNodePtr CreateNodeHelper::CreateNodeWithCheck(const AnfNodePtr &node) {
  if (!node || !AnfUtils::IsRealCNodeKernel(node)) {
    return node;
  }
  auto op_name = common::AnfAlgo::GetCNodeName(node);
  auto op_adaptation_info = OpAdaptationInfoRegister::GetInstance().GetOpAdaptationInfo(op_name, kAscendDevice, true);
  if (!op_adaptation_info) {
    return node;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto ret_node = ConvertToTargetOp(cnode, op_adaptation_info);
  if ((ret_node != nullptr) && (ret_node != cnode)) {
    MS_LOG(INFO) << "Replace op " << cnode->fullname_with_scope() << " debug string:" << cnode->DebugString()
                 << " with " << ret_node->fullname_with_scope() << " debug string:" << ret_node->DebugString()
                 << ", is dynamic shape:" << common::AnfAlgo::IsDynamicShape(node);
  }
  return ret_node;
}

CNodePtr CreateNodeHelper::ConvertToTargetOp(const CNodePtr &origin_op, OpAdaptationInfo *op_adaptation_info) {
  MS_EXCEPTION_IF_NULL(origin_op);
  MS_EXCEPTION_IF_NULL(op_adaptation_info);
  auto me_op_name = op_adaptation_info->me_op_name();
  auto default_op_name = op_adaptation_info->backend_op_name();
  auto target_op_name = op_adaptation_info->target_op_name();
  auto input_to_attr_map = op_adaptation_info->input_attr_map();
  // rename to default op name
  RenamePrimitiveName(origin_op, me_op_name, default_op_name);
  // Check supported if the op need
  auto graph = origin_op->func_graph();
  auto kernel_graph = graph->cast<KernelGraphPtr>();

  if (!input_to_attr_map.empty()) {
    auto target_op = CreateTargetOp(origin_op, *op_adaptation_info);
    if (target_op == nullptr) {
      MS_LOG(DEBUG) << "Create target op failed for node " << origin_op->fullname_with_scope();
      return origin_op;
    }
    if (kernel_graph != nullptr) {
      kernel_graph->FrontBackendlMapUpdate(origin_op, target_op);
    }
    return target_op;
  }

  return origin_op;
}

void CreateNodeHelper::RenamePrimitiveName(const CNodePtr &origin_op, const string &me_op_name,
                                           const string &default_op_name) {
  MS_EXCEPTION_IF_NULL(origin_op);
  if (default_op_name == me_op_name) {
    return;
  }
  auto primitive = GetCNodePrimitive(origin_op);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive->set_name(default_op_name);
  // reset full scope name
  origin_op->set_fullname_with_scope("");
  MS_LOG(INFO) << "Rename op type from " << me_op_name << " to " << default_op_name << " for op "
               << origin_op->fullname_with_scope();
  if (default_op_name == kSparseGatherV2OpName) {
    common::AnfAlgo::SetNodeAttr(kAttrIsSparse, MakeValue(true), origin_op);
  }
  common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), origin_op);
}

CNodePtr CreateNodeHelper::CreateTargetOp(const CNodePtr &origin_op, const OpAdaptationInfo &op_adaptation_info) {
  MS_EXCEPTION_IF_NULL(origin_op);
  auto target_op_name = op_adaptation_info.target_op_name();
  auto input_attr_info_map = op_adaptation_info.input_attr_map();

  auto origin_primitive = GetCNodePrimitive(origin_op);
  MS_EXCEPTION_IF_NULL(origin_primitive);
  auto target_primitive = std::make_shared<Primitive>(target_op_name);
  MS_EXCEPTION_IF_NULL(target_primitive);
  (void)target_primitive->SetAttrs(origin_primitive->attrs());
  std::vector<AnfNodePtr> target_inputs;
  auto inputs = origin_op->inputs();
  target_inputs.push_back(inputs[0]);

  for (size_t i = 0; i < inputs.size() - 1; ++i) {
    auto input_node = inputs[i + 1];
    MS_EXCEPTION_IF_NULL(input_node);
    if (IsPrimitiveCNode(input_node, prim::kPrimDepend)) {
      input_node = AnfUtils::VisitKernel(input_node, 0).first;
    }

    auto iter = input_attr_info_map.find(i);
    if (iter != input_attr_info_map.end() && input_node->isa<ValueNode>() && !HasAbstractMonad(input_node)) {
      auto ret = ConvertInputToAttr(origin_op, i, input_node, iter->second, target_primitive);
      if (!ret) {
        return nullptr;
      }
    } else {
      target_inputs.push_back(inputs[i + 1]);
    }
  }

  // Update target_op's inputs
  target_inputs[0] = NewValueNode(target_primitive);
  auto graph = origin_op->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto target_op = opt::NewCNode(target_inputs, graph, {origin_op});
  MS_EXCEPTION_IF_NULL(target_op);
  target_op->set_abstract(origin_op->abstract());
  target_op->set_scope(origin_op->scope());
  target_op->set_primal_attrs(origin_op->primal_attrs());
  target_op->set_attrs(origin_op->attrs());
  common::AnfAlgo::EraseNodeAttr(kAttrIsKernelDynamicImpl, target_op);
  auto is_dynamic = common::AnfAlgo::IsDynamicShape(origin_op);
  MS_LOG(DEBUG) << "Create op " << target_op->fullname_with_scope() << ", debug string:" << target_op->DebugString()
                << ", attr text:" << target_primitive->GetAttrsText() << " from " << origin_op->fullname_with_scope()
                << ", debug string:" << origin_op->DebugString() << ", attr text:" << origin_primitive->GetAttrsText()
                << ", is dynamic shape:" << is_dynamic;
  common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), target_op);
  common::AnfAlgo::SetNodeAttr(kAttrMeOpName, MakeValue(origin_primitive->name()), target_op);
  return target_op;
}

bool CreateNodeHelper::ConvertInputToAttr(const CNodePtr &origin_op, size_t i,
                                          const std::shared_ptr<AnfNode> &input_node, const std::string &attr_data_type,
                                          const std::shared_ptr<Primitive> &target_primitive) {
  auto value_node = input_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  MS_LOG(DEBUG) << "start erase input[" << i
                << "] of cnode[" + origin_op->DebugString() + "], origin value:" << value_node->ToString()
                << ", Type:" << value_node->type_name();

  auto value = value_node->value();
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    if (tensor->data().const_data() == nullptr) {
      MS_LOG(DEBUG) << "Const input data ptr is null from op " << origin_op->fullname_with_scope() << "'s input " << i;
      return false;
    }
    value = CreateValueFromTensor(tensor);
    value = UpdateValueByAttrDataType(value, attr_data_type);
    MS_LOG(DEBUG) << "new attr value:" << value_node->ToString() << ", Type:" << value_node->type_name();
  }

  std::string attr_name = GetInputName(origin_op, i);
  if (attr_name.empty()) {
    return false;
  }

  if (origin_op->HasAttr(attr_name)) {
    auto origin_primitive = GetCNodePrimitive(origin_op);
    MS_EXCEPTION_IF_NULL(origin_primitive);
    MS_LOG(ERROR) << "Origin op already has this attr " << attr_name
                  << ". op attrs:" << origin_primitive->GetAttrsText() << ". DebugString:" << origin_op->DebugString();
    return false;
  }

  target_primitive->set_attr(attr_name, value);
  return true;
}
}  // namespace mindspore::opt
