/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/reduce_fake_out_mem.h"

#include <memory>
#include <set>
#include <vector>
#include "kernel/common_utils.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"

namespace mindspore::graphkernel {
constexpr auto kFakeOut = "fake_output";
void ReduceFakeOutMem::ModifyAbstract(const AnfNodePtr &composite_node, const std::set<size_t> &fake_real_indices,
                                      const AnfNodePtrList &output_list) const {
  if (fake_real_indices.empty()) {
    return;
  }

  if (output_list.empty()) {
    MS_LOG(EXCEPTION) << "Output size should not be zero while there is at least one fake output in node "
                      << composite_node->fullname_with_scope();
  }

  std::vector<AbstractBasePtr> out_specs;
  for (size_t i = 0; i < output_list.size(); ++i) {
    if (fake_real_indices.count(i) != 0) {
      std::vector<int64_t> shape_vec_shape = {1};
      AbstractBasePtr abstract = std::make_shared<abstract::AbstractTensor>(kInt64, shape_vec_shape);
      out_specs.push_back(abstract);
      if (output_list.size() > 1) {
        output_list[i]->set_abstract(abstract);
      }
    } else {
      out_specs.push_back(output_list[i]->abstract());
    }
  }
  AbstractBasePtr out_spec;
  if (output_list.size() > 1) {
    out_spec = std::make_shared<abstract::AbstractTuple>(out_specs);
  } else {
    out_spec = output_list[0]->abstract();
  }
  composite_node->set_abstract(out_spec);
  auto gk_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(composite_node);
  auto tuple_output = gk_graph->output()->cast<CNodePtr>();
  if (IsPrimitiveCNode(tuple_output, prim::kPrimMakeTuple)) {
    tuple_output->set_abstract(out_spec);
  }
}

bool ReduceFakeOutMem::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = false;

  auto todos = TopoSort(func_graph->get_return());
  for (auto node : todos) {
    if (!common::AnfAlgo::IsGraphKernel(node)) {
      continue;
    }
    auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
    MS_EXCEPTION_IF_NULL(sub_graph);

    AnfNodePtrList output_list;
    kernel::GetFuncGraphOutputNodes(sub_graph, &output_list);
    std::set<size_t> fake_real_indices;
    for (size_t i = 0; i < output_list.size(); ++i) {
      auto &out = output_list[i];
      auto out_cnode = out->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(out_cnode);
      if (common::AnfAlgo::HasNodeAttr(kFakeOut, out_cnode) &&
          common::AnfAlgo::GetNodeAttr<bool>(out_cnode, kFakeOut)) {
        (void)fake_real_indices.insert(i);
      }
    }

    if (fake_real_indices.empty()) {
      continue;
    }

    ModifyAbstract(node, fake_real_indices, output_list);
    changed = true;
  }

  return changed;
}
}  // namespace mindspore::graphkernel
