/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/seed_adapter.h"

#include <string>
#include <vector>
#include <memory>
#include "backend/common/optimizer/helper.h"
#include "kernel/kernel_build_info.h"
#include "include/common/utils/utils.h"
#include "utils/trace_base.h"
#include "backend/common/session/kernel_graph.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/kernel_info.h"
#include "kernel/oplib/oplib.h"

namespace mindspore::opt {
namespace {
const std::set<std::string> kNodeWithSeedOperators = {kGammaOpName,          kPoissonOpName,    kStandardLaplaceOpName,
                                                      kStandardNormalOpName, kUniformIntOpName, kUniformRealOpName,
                                                      kDropoutGenMaskOpName};
template <typename T>
tensor::TensorPtr CreateTensor(T seed) {
  // 1 create seed tensor
  std::vector<int64_t> indices_shape = {1};
  auto type = std::is_same<T, int64_t>::value ? kInt64 : kUInt64;
  TensorTypePtr tensor_type = std::make_shared<TensorType>(kInt64);
  MS_EXCEPTION_IF_NULL(tensor_type);
  tensor::DeviceInfo device_info{kOpFormat_DEFAULT, tensor_type};
  tensor::TensorPtr indices_tensor = std::make_shared<tensor::Tensor>(type->type_id(), indices_shape);
  MS_EXCEPTION_IF_NULL(indices_tensor);
  indices_tensor->set_device_info(device_info);
  // 2 set value of tensor
  auto data_ptr = indices_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  auto ptr = static_cast<T *>(data_ptr);
  *ptr = seed;
  return indices_tensor;
}

template <typename T>
ValueNodePtr CreateValueNode(T seed) {
  tensor::TensorPtr tensor = CreateTensor(seed);
  MS_EXCEPTION_IF_NULL(tensor);
  auto value_node = std::make_shared<ValueNode>(tensor);
  MS_EXCEPTION_IF_NULL(value_node);
  auto abstract = tensor->ToAbstract();
  value_node->set_abstract(abstract);
  auto indices_kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(indices_kernel_info);
  value_node->set_kernel_info(indices_kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetOutputsFormat({kOpFormat_DEFAULT});
  if (std::is_same<T, int64_t>::value) {
    builder.SetOutputsDeviceType({kNumberTypeInt64});
  } else {
    builder.SetOutputsDeviceType({kNumberTypeUInt64});
  }
  builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), value_node.get());
  return value_node;
}

std::vector<ValueNodePtr> ConvertAttrToValueNode(const std::shared_ptr<kernel::OpInfo> &op_info,
                                                 const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(op_info);
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<ValueNodePtr> ret = {};
  // DropoutGenMask only create offset
  if (op_info->op_name() == kDropoutGenMaskOpName) {
    uint64_t offset = 0;
    auto offset0 = CreateValueNode(offset);
    auto offset1 = CreateValueNode(offset);
    if (offset0 == nullptr || offset1 == nullptr) {
      MS_LOG(EXCEPTION) << "Create value node error, node: " << cnode->DebugString() << trace::DumpSourceLines(cnode);
    }
    (void)ret.emplace_back(offset0);
    (void)ret.emplace_back(offset1);
  } else {
    // Get seed to create value node
    auto attrs = op_info->attrs_ptr();
    if (attrs.empty()) {
      MS_LOG(EXCEPTION) << "Node(" << cnode->DebugString() << ") doesn't have any attrs."
                        << trace::DumpSourceLines(cnode);
    }
    for (const auto &attr : attrs) {
      if (!common::AnfAlgo::HasNodeAttr(attr->name(), cnode)) {
        MS_LOG(EXCEPTION) << "Node(" << cnode->DebugString() << ") doesn't have attr(" << attr->name() << ")."
                          << trace::DumpSourceLines(cnode);
      }
      auto attr_value = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, attr->name());
      auto value_node = CreateValueNode(attr_value);
      if (value_node == nullptr) {
        MS_LOG(EXCEPTION) << "Create value node error, node: " << cnode->DebugString() << ", seed value: " << attr_value
                          << trace::DumpSourceLines(cnode);
      }
      (void)ret.emplace_back(value_node);
    }
    if (ret.empty()) {
      MS_LOG(EXCEPTION) << "Node(" << cnode->DebugString() << ") doesn't have any matched attrs."
                        << trace::DumpSourceLines(cnode);
    }
  }
  return ret;
}
}  // namespace

const BaseRef SeedAdapter::DefinePattern() const {
  std::shared_ptr<Var> V = std::make_shared<CondVar>(UnVisited);
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

// This pass in ordr to convert attr seed to value node
// exp: DropoutGenMask
//     |input0   |input1                   |input0     |input1     |s0      |s1
//  DropoutGenMask(seed0/seed1)      --->    DropoutGenMask(seed0/seed1)
//            |                                       |
const AnfNodePtr SeedAdapter::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto cnode_type = common::AnfAlgo::GetCNodeName(node);
  if (kNodeWithSeedOperators.find(cnode_type) == kNodeWithSeedOperators.end()) {
    return nullptr;
  }
  // 1. convert attr seed to value node
  auto op_info = kernel::OpLib::FindOp(cnode_type, kernel::OpImplyType::kImplyAICPU);
  if (!op_info) {
    MS_LOG(EXCEPTION) << "Find op info failed, node type: " << cnode_type << ", node debug: " << cnode->DebugString();
  }
  auto value_nodes = ConvertAttrToValueNode(op_info, cnode);
  for (auto &value_node : value_nodes) {
    cnode->add_input(value_node);
    kernel_graph->AddValueNodeToGraph(value_node);
  }
  // 2. set visited
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  return node;
}
}  // namespace mindspore::opt
