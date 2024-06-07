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
#include "plugin/device/ascend/optimizer/ge/insert_identity.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_set>
#include "ops/array_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"
#include "ops/nn_op_name.h"

namespace mindspore {
namespace opt {
namespace {
bool NeedInsertIdentity(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    AnfNodePtr in = utils::cast<AnfNodePtr>(n);
    MS_EXCEPTION_IF_NULL(in);
    if (IsValueNode<Primitive>(in)) {
      auto value_node = in->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto value = value_node->value();
      MS_EXCEPTION_IF_NULL(value);
      auto prim_py = value->cast<PrimitivePtr>();
      MS_EXCEPTION_IF_NULL(prim_py);
      return prim_py->HasAttr(kAttrAclSpecialFormat) || prim_py->HasAttr(kAttrAclSpecialInputFormat) ||
             prim_py->HasAttr(kAttrAclInconsistentInputDtype);
    }
    return false;
  }
  return false;
}

void SetBuildInfo(const AnfNodePtr &cnode, const CNodePtr &new_node, const size_t index, bool need_trans = true,
                  bool need_cast = false, const std::string &dst_dev_fmt = kOpFormat_DEFAULT) {
  MS_EXCEPTION_IF_NULL(cnode);
  const std::string dev_fmt = AnfAlgo::GetOutputFormat(cnode, index);
  const auto origin_shape = AnfAlgo::GetOutputDetailShape(cnode, index);
  TypeId output_type = AnfAlgo::GetOutputDeviceDataType(cnode, index);
  TypeId input_type = output_type;
  if (need_cast) {
    input_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(cnode, index);
    output_type = AnfAlgo::GetInputDeviceDataType(cnode, index);
  }
  // set abstract
  abstract::AbstractTensorPtr abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(output_type), origin_shape);
  new_node->set_abstract(abs);
  // set kernel build info
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->IsEnableInferBoost() && context_ptr->ascend_soc_version() == "ascend310p") {
    builder.SetKernelType(KernelType::INTERNAL_KERNEL);
  } else {
    builder.SetKernelType(KernelType::ACL_KERNEL);
  }
  if (need_trans) {
    builder.SetInputsFormat({dev_fmt});
  } else {
    builder.SetInputsFormat({kOpFormat_DEFAULT});
  }
  builder.SetOutputsFormat({dst_dev_fmt});
  builder.SetInputsReshapeType({});
  builder.SetOutputsReshapeType({});
  builder.SetInputsDeviceType({input_type});
  builder.SetOutputsDeviceType({output_type});
  builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), new_node.get());
}

CNodePtr InsertTransIdentityForInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                     const std::vector<size_t> &trans_input, std::unordered_set<size_t> *cast_input) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (size_t i = 0; i < trans_input.size(); ++i) {
    auto index = trans_input[i];
    auto input_kernel = common::AnfAlgo::GetInputNode(cnode, index);
    auto identity_node =
      kernel_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kIdentityOpName)), input_kernel});
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    if (context_ptr->IsEnableInferBoost() && context_ptr->ascend_soc_version() == "ascend310p") {
      identity_node =
        kernel_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kTransDataOpName)), input_kernel});
    }

    identity_node->set_scope(cnode->scope());
    auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, index, false);
    bool need_cast = cast_input->count(index) != 0;
    if (need_cast) {
      cast_input->erase(index);
    }
    if (common::AnfAlgo::HasNodeAttr(kAttrInternalSepcialFormat, cnode)) {
      auto special_inputs = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrInternalSepcialFormat);
      if (special_inputs.size() != trans_input.size()) {
        MS_LOG(EXCEPTION) << cnode->fullname_with_scope()
                          << " special_input size must be equal to trans_input size, but got special_inputs = "
                          << special_inputs << " trans_input = " << trans_input;
      }
      common::AnfAlgo::SetNodeAttr(kAttrInternalSepcialFormat, MakeValue<int64_t>(special_inputs[i]), identity_node);
    }
    std::string dst_dev_fmt = AnfAlgo::GetInputFormat(cnode, index);
    SetBuildInfo(kernel_with_index.first, identity_node, kernel_with_index.second, true, need_cast, dst_dev_fmt);
    common::AnfAlgo::SetNodeInput(cnode, identity_node, index);
  }
  return cnode;
}

CNodePtr InsertCastIdentityForInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                    const std::unordered_set<size_t> &cast_input) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (auto index : cast_input) {
    auto input_kernel = common::AnfAlgo::GetInputNode(cnode, index);
    auto identity_node =
      kernel_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kIdentityOpName)), input_kernel});
    identity_node->set_scope(cnode->scope());
    auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, index, false);
    SetBuildInfo(kernel_with_index.first, identity_node, kernel_with_index.second, false, true);
    common::AnfAlgo::SetNodeInput(cnode, identity_node, index);
  }
  return cnode;
}

CNodePtr InsertIdentityForOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetOutputTensorNum(cnode) != 1) {
    return cnode;
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  CNodePtr new_node = kernel_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kIdentityOpName)), cnode});
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->IsEnableInferBoost() && context_ptr->ascend_soc_version() == "ascend310p") {
    new_node = kernel_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kTransDataOpName)), cnode});
  }
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(cnode->scope());
  SetBuildInfo(cnode, new_node, 0);
  return new_node;
}
}  // namespace

const BaseRef InsertIdentity::DefinePattern() const {
  VarPtr V = std::make_shared<CondVar>(NeedInsertIdentity);
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr InsertIdentity::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                         const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealCNodeKernel(node) || func_graph == nullptr) {
    return nullptr;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  CNodePtr new_cnode = cnode;
  std::vector<size_t> trans_inputs;
  std::unordered_set<size_t> cast_inputs;
  if (common::AnfAlgo::HasNodeAttr(kAttrAclSpecialInputFormat, cnode)) {
    trans_inputs = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(node, kAttrAclSpecialInputFormat);
    common::AnfAlgo::EraseNodeAttr(kAttrAclSpecialInputFormat, node);
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrAclInconsistentInputDtype, cnode)) {
    auto inconsistent_inputs = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(node, kAttrAclInconsistentInputDtype);
    std::for_each(inconsistent_inputs.begin(), inconsistent_inputs.end(),
                  [&cast_inputs](size_t index) { cast_inputs.insert(index); });
    common::AnfAlgo::EraseNodeAttr(kAttrAclInconsistentInputDtype, node);
  }
  if (!trans_inputs.empty()) {
    new_cnode = InsertTransIdentityForInput(func_graph, cnode, trans_inputs, &cast_inputs);
  }
  if (!cast_inputs.empty()) {
    new_cnode = InsertCastIdentityForInput(func_graph, cnode, cast_inputs);
  }

  if (common::AnfAlgo::HasNodeAttr(kAttrAclSpecialFormat, cnode)) {
    common::AnfAlgo::EraseNodeAttr(kAttrAclSpecialFormat, node);
    new_cnode = InsertIdentityForOutput(func_graph, new_cnode);
  }
  return new_cnode;
}
}  // namespace opt
}  // namespace mindspore
