/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/optimizer/combine_cast_fusion.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
bool IsParameter(const mindspore::AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->isa<Parameter>() || (IsPrimitiveCNode(node, prim::kPrimLoad) &&
                                    (common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0))->isa<Parameter>());
}
bool GetDealList(const std::vector<AnfNodePtr> &node_list, std::vector<std::vector<AnfNodePtr>> *deal_list) {
  MS_EXCEPTION_IF_NULL(deal_list);
  std::vector<AnfNodePtr> cast_32to16_list;
  std::vector<AnfNodePtr> cast_16to32_list;
  AnfNodePtr cast_32to16_load_monad = nullptr;
  AnfNodePtr cast_16to32_load_monad = nullptr;
  constexpr size_t second_input_index = 2;
  for (auto &cast_node : node_list) {
    MS_EXCEPTION_IF_NULL(cast_node);
    // currently, we only deal with the construct : [Param->Cast->] to avoid being a cycle.
    // { prim::kPrimCast, { prim::kPrimLoad, Parameter, U }}
    if (!IsPrimitiveCNode(cast_node, prim::kPrimCast)) {
      continue;
    }
    auto input0 = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(cast_node), 0);
    MS_EXCEPTION_IF_NULL(input0);
    if (IsParameter(input0)) {
      auto dst = common::AnfAlgo::GetOutputInferDataType(cast_node, 0);
      auto src = common::AnfAlgo::GetPrevNodeOutputInferDataType(cast_node, 0);
      if (dst == kNumberTypeFloat16 && src == kNumberTypeFloat32) {
        cast_32to16_list.push_back(cast_node);
        if (IsPrimitiveCNode(input0, prim::kPrimLoad)) {
          auto &monad_32to16 = input0->cast<CNodePtr>()->inputs().at(second_input_index);
          if (cast_32to16_load_monad == nullptr) {
            cast_32to16_load_monad = monad_32to16;
          } else if (cast_32to16_load_monad != monad_32to16) {
            return false;
          }
        }
      } else if (dst == kNumberTypeFloat32 && src == kNumberTypeFloat16) {
        cast_16to32_list.push_back(cast_node);
        if (IsPrimitiveCNode(input0, prim::kPrimLoad)) {
          auto &monad_16to32 = input0->cast<CNodePtr>()->inputs().at(second_input_index);
          if (cast_16to32_load_monad == nullptr) {
            cast_16to32_load_monad = monad_16to32;
          } else if (cast_16to32_load_monad != monad_16to32) {
            return false;
          }
        }
      }
    }
  }
  if (cast_32to16_list.size() <= 1 && cast_16to32_list.size() <= 1) {
    return false;
  }
  if (cast_32to16_list.size() > 1) {
    deal_list->push_back(cast_32to16_list);
  }
  if (cast_16to32_list.size() > 1) {
    deal_list->push_back(cast_16to32_list);
  }
  return true;
}
}  // namespace
bool CastAllFusion::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  // 1 get all the cast node
  std::vector<std::vector<AnfNodePtr>> deal_list;
  if (!GetDealList(node_list, &deal_list)) {
    return false;
  }
  for (auto cast_list : deal_list) {
    // 2 create node CastAll
    auto prim = std::make_shared<Primitive>("CastAll");
    std::vector<AnfNodePtr> inputs = {NewValueNode(prim)};
    // set inputs for CastAll
    for (size_t idx = 0; idx < cast_list.size(); ++idx) {
      auto cnode = utils::cast<CNodePtr>(cast_list[idx]);
      MS_EXCEPTION_IF_NULL(cnode);
      inputs.push_back(common::AnfAlgo::GetInputNode(cnode, 0));
    }
    if (cast_list.size() > 0) {
      MS_EXCEPTION_IF_NULL(cast_list[0]);
      TraceGuard guard(std::make_shared<TraceOpt>(cast_list[0]->debug_info()));
      auto cast_all = graph->NewCNode(inputs);
      auto kernel_info = std::make_shared<device::KernelInfo>();
      MS_EXCEPTION_IF_NULL(cast_all);
      MS_EXCEPTION_IF_NULL(kernel_info);
      cast_all->set_kernel_info(kernel_info);
      AbstractBasePtrList abstract_list;
      for (size_t idx = 0; idx < cast_list.size(); ++idx) {
        auto cnode = utils::cast<CNodePtr>(cast_list[idx]);
        MS_EXCEPTION_IF_NULL(cnode);
        abstract_list.push_back(cnode->abstract());
      }
      auto kernel_build_info = GenerateKernelBuildInfo(cast_list);
      AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info, cast_all.get());
      auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
      MS_EXCEPTION_IF_NULL(abstract_tuple);
      cast_all->set_abstract(abstract_tuple);
      common::AnfAlgo::SetNodeAttr("n", MakeValue(cast_list.size()), cast_all);
      // 3 replace all the cast by CastAllv tuplegetitem[castall, idx]
      for (size_t idx = 0; idx < cast_list.size(); ++idx) {
        std::vector<AnfNodePtr> tuple_getitem_input;
        tuple_getitem_input.push_back(NewValueNode(prim::kPrimTupleGetItem));
        tuple_getitem_input.push_back(cast_all);
        auto index = NewValueNode(SizeToLong(idx));
        auto imm = std::make_shared<Int64Imm>(idx);
        auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
        MS_EXCEPTION_IF_NULL(index);
        MS_EXCEPTION_IF_NULL(abstract_scalar);
        index->set_abstract(abstract_scalar);
        tuple_getitem_input.push_back(index);
        AnfNodePtr tuple_getitem = graph->NewCNode(tuple_getitem_input);
        MS_EXCEPTION_IF_NULL(tuple_getitem);
        tuple_getitem->set_abstract(cast_list[idx]->abstract());
        if (!manager->Replace(cast_list[idx], tuple_getitem)) {
          MS_LOG(EXCEPTION) << "manager replace node failed";
        }
      }
    } else {
      MS_LOG(EXCEPTION) << "The size of cast_list is zero.";
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
