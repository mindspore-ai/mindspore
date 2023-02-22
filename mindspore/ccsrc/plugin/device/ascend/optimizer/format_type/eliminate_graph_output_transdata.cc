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

#include "plugin/device/ascend/optimizer/format_type/eliminate_graph_output_transdata.h"

#include <set>
#include <string>
#include <tuple>
#include <vector>
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kNchwDimNum = 4;
constexpr size_t kDimC = 1;
const std::set<std::string> kFormatsNeedTransdata = {kOpFormat_ND_RNN_BIAS, kOpFormat_FRACTAL_ZN_RNN,
                                                     kOpFormat_C1HWNCoC0, kOpFormat_FRACTAL_ZN_LSTM};

bool IsDepthwiseCase(const AnfNodePtr &node, size_t index, const std::string &format) {
  if (format != kOpFormat_FRAC_Z) {
    return false;
  }
  abstract::BaseShapePtr base_shape = AnfAlgo::GetOutputDetailShape(node, index);
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::Shape>()) {
    auto shape_ptr = base_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    auto shape_vec = shape_ptr->shape();
    return shape_vec.size() == kNchwDimNum && shape_vec[kDimC] == 1;
  }
  return false;
}

// Here we assume that the TransData operator has only a single input and a single output.
// So the graph structure is like this:
//    TransData   Ops
//        |     /
//      MakeTuple
//          |
//      MakeTuple
// There is no TupleGetItem between TransData and MakeTuple.
// Based on this assumption, this dfs algorithm will be much simpler.
void GetNeedReplaceEdges(const AnfNodePtr &old_make_tuple, const AnfNodePtr &node, size_t tuple_input_index,
                         const HashMap<AnfNodePtr, size_t> &transdata_ref_count,
                         std::vector<std::tuple<AnfNodePtr, size_t, AnfNodePtr>> *edges) {
  std::vector<PrimitivePtr> return_type = {prim::kPrimMakeTuple};
  auto item_with_index = common::AnfAlgo::VisitKernelWithReturnType(node, 0, false, return_type);
  if (common::AnfAlgo::CheckPrimitiveType(item_with_index.first, prim::kPrimMakeTuple)) {
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    auto make_tuple = item_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    for (size_t i = 1; i < make_tuple->inputs().size(); i++) {
      GetNeedReplaceEdges(make_tuple, make_tuple->input(i), i, transdata_ref_count, edges);
    }
  } else {
    if (!item_with_index.first->isa<CNode>()) {
      return;
    }
    // TransData only have one output.
    if (common::AnfAlgo::GetCNodeName(item_with_index.first) == kTransDataOpName) {
      auto iter = transdata_ref_count.find(item_with_index.first);
      // TransData is in use.
      if (iter != transdata_ref_count.end() && iter->second > 0) {
        return;
      }
      // TransData input node
      auto transdata_input = common::AnfAlgo::GetInputNode(item_with_index.first->cast<CNodePtr>(), 0);
      auto [real_input, index] = common::AnfAlgo::VisitKernel(transdata_input, 0);
      auto format = AnfAlgo::GetOutputFormat(real_input, index);
      if (kFormatsNeedTransdata.find(format) == kFormatsNeedTransdata.end() &&
          !IsDepthwiseCase(real_input, index, format)) {
        (void)edges->emplace_back(std::make_tuple(old_make_tuple, tuple_input_index, transdata_input));
      }
    }
  }
}

// We can't eliminate the TransData by SetEdge(MakeTuple, input_index, TransData_input_node).
// Because the 'TransData-MakeTuple' is used by 'other-op2'.
// (TransData is directly linked to 'other-op1', 'SetEdge' will not affect this edge)
//            TransData
//           /      |
//       MakeTuple  other-op1
//       /      |
//  MakeTuple TupleGetItem
//       |           |
//      ...        other-op2
// So we need to establish the reference count of TransData.
HashMap<AnfNodePtr, size_t> GetTransdataRefCount(const FuncGraphPtr &func_graph) {
  auto real_outputs = common::AnfAlgo::GetAllOutput(func_graph->get_return());
  HashMap<AnfNodePtr, size_t> transdata_ref_count;
  for (auto &node : real_outputs) {
    if (!node->isa<CNode>()) {
      continue;
    }
    if (common::AnfAlgo::GetCNodeName(node) == kTransDataOpName) {
      transdata_ref_count[node] = 0;
    }
  }

  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto input_num = common::AnfAlgo::GetInputNum(cnode);
    for (size_t i = 0; i < input_num; ++i) {
      auto input = common::AnfAlgo::GetInputNode(cnode, i);
      MS_EXCEPTION_IF_NULL(input);
      if (!input->isa<CNode>() || common::AnfAlgo::GetCNodeName(input) == kTransDataOpName) {
        continue;
      }
      auto real_input = common::AnfAlgo::VisitKernel(input, 0);
      MS_EXCEPTION_IF_NULL(real_input.first);
      if (!real_input.first->isa<CNode>()) {
        continue;
      }
      if (common::AnfAlgo::GetCNodeName(real_input.first) == kTransDataOpName) {
        // Is graph output transdata
        if (transdata_ref_count.count(real_input.first) > 0) {
          transdata_ref_count[real_input.first] += 1;
        }
      }
    }
  }
  return transdata_ref_count;
}
}  // namespace

bool EliminateGraphOutputTransdata::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<PrimitivePtr> return_type = {prim::kPrimMakeTuple};
  auto item_with_index = common::AnfAlgo::VisitKernelWithReturnType(func_graph->output(), 0, false, return_type);
  if (!common::AnfAlgo::CheckPrimitiveType(item_with_index.first, prim::kPrimMakeTuple)) {
    MS_LOG(INFO) << "Graph output is not a MakeTuple";
    return true;
  }

  auto transdata_ref_count = GetTransdataRefCount(func_graph);

  std::vector<std::tuple<AnfNodePtr, size_t, AnfNodePtr>> edges;
  MS_EXCEPTION_IF_NULL(item_with_index.first);
  auto make_tuple = item_with_index.first->cast<CNodePtr>();
  GetNeedReplaceEdges(make_tuple, make_tuple, 0, transdata_ref_count, &edges);

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &[node, input_index, input] : edges) {
    manager->SetEdge(node, SizeToInt(input_index), input);
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
