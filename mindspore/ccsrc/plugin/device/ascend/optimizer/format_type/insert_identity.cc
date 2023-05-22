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
#include "plugin/device/ascend/optimizer/format_type/insert_identity.h"

#include <memory>
#include <string>
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
namespace {
bool SpecialOut(const BaseRef &n) {
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
      return prim_py->HasAttr(kAttrAclSpecialFormat);
    }
    return false;
  }
  return false;
}

AnfNodePtr InsertIdentityForOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetOutputTensorNum(cnode) != 1) {
    return cnode;
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  CNodePtr new_node = kernel_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kIdentity)), cnode});
  MS_EXCEPTION_IF_NULL(new_node);

  const std::string dev_fmt = AnfAlgo::GetOutputFormat(cnode, 0);
  const auto origin_shape = AnfAlgo::GetOutputDetailShape(cnode, 0);
  const TypeId device_type = AnfAlgo::GetOutputDeviceDataType(cnode, 0);
  // set abstract
  abstract::AbstractTensorPtr abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(device_type), origin_shape);
  new_node->set_abstract(abs);
  // set kernel build info
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetKernelType(KernelType::ACL_KERNEL);
  builder.SetInputsFormat({dev_fmt});
  builder.SetOutputsFormat({kOpFormat_DEFAULT});
  builder.SetInputsReshapeType({});
  builder.SetOutputsReshapeType({});
  builder.SetInputsDeviceType({device_type});
  builder.SetOutputsDeviceType({device_type});
  builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), new_node.get());

  return new_node;
}
}  // namespace

const BaseRef InsertIdentity::DefinePattern() const {
  VarPtr V = std::make_shared<CondVar>(SpecialOut);
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr InsertIdentity::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                         const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealCNodeKernel(node) || func_graph == nullptr) {
    return nullptr;
  }
  common::AnfAlgo::EraseNodeAttr(kAttrAclSpecialFormat, node);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // process output
  return InsertIdentityForOutput(func_graph, cnode);
}
}  // namespace opt
}  // namespace mindspore
