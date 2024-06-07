/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ge/add_cast_for_ge.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "mindspore/core/ops/array_ops.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/op_utils.h"
#include "mindspore/core/ops/auto_generate/gen_ops_name.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {

namespace {
struct CastInfo {
  // input_index/output_index
  size_t idx;
  // If `src_dtypes` is empty, it represents all dtypes
  std::unordered_set<TypeId> src_dtypes;
  TypeId dst_dtype;
};

const std::unordered_set<TypeId> int_type_with_bool = {kNumberTypeUInt8,  kNumberTypeUInt16, kNumberTypeUInt32,
                                                       kNumberTypeUInt64, kNumberTypeInt8,   kNumberTypeInt16,
                                                       kNumberTypeInt32,  kNumberTypeInt64,  kNumberTypeBool};

// prim_name | (input_vector, output_vector) vector: {{input_index/output_index, src_dtypes, dst_dtype}}
const std::unordered_map<std::string, std::pair<std::vector<CastInfo>, std::vector<CastInfo>>> kNeedAddCastMap = {
  {ops::kNameReciprocal, {{{0, int_type_with_bool, kNumberTypeFloat32}}, {}}},
  {ops::kNameExp, {{{0, int_type_with_bool, kNumberTypeFloat32}}, {}}},
  {ops::kNameTanh, {{{0, int_type_with_bool, kNumberTypeFloat32}}, {}}},
  {ops::kNameSqrt, {{{0, int_type_with_bool, kNumberTypeFloat32}}, {}}},
  {ops::kNameRsqrt, {{{0, int_type_with_bool, kNumberTypeFloat32}}, {}}},
  {ops::kNameErfinv, {{{0, int_type_with_bool, kNumberTypeFloat32}}, {}}},
  {ops::kNameErf, {{{0, int_type_with_bool, kNumberTypeFloat32}}, {}}},
  {ops::kNameReduceAny, {{{0, {}, kNumberTypeBool}}, {}}},
  {ops::kNameReduceAll, {{{0, {}, kNumberTypeBool}}, {}}},
  {ops::kNameLogicalAnd, {{{0, {}, kNumberTypeBool}, {1, {}, kNumberTypeBool}}, {}}},
  {ops::kNameLogicalOr, {{{0, {}, kNumberTypeBool}, {1, {}, kNumberTypeBool}}, {}}},
  {ops::kNameLogicalNot, {{{0, {}, kNumberTypeBool}}, {}}},
  {ops::kNameDiv, {{{0, int_type_with_bool, kNumberTypeFloat32}, {1, int_type_with_bool, kNumberTypeFloat32}}, {}}},
  {ops::kNameArgMaxWithValue, {{}, {{0, {}, kNumberTypeInt32}}}},
  {ops::kNameArgMinWithValue, {{}, {{0, {}, kNumberTypeInt32}}}}};

bool NeedAddCast(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    auto node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    if (!IsValueNode<Primitive>(node)) {
      return false;
    }
    auto prim = GetValuePtr<Primitive>(node);
    MS_EXCEPTION_IF_NULL(prim);
    return kNeedAddCastMap.find(prim->name()) != kNeedAddCastMap.cend();
  }
  return false;
}

bool NodeNeedCast(const TypeId node_dtype, const std::unordered_set<TypeId> &src_dtypes, const TypeId dst_dtype) {
  if (src_dtypes.empty() || src_dtypes.find(node_dtype) != src_dtypes.end()) {
    return node_dtype != dst_dtype;
  }
  return false;
}

const CNodePtr AddCastForGeInput(const FuncGraphPtr &graph, const CNodePtr &node,
                                 const std::vector<CastInfo> &cast_info_vector) {
  std::vector<AnfNodePtr> new_inputs(node->inputs());
  bool tag = False;
  for (size_t i = 0; i < cast_info_vector.size(); ++i) {
    auto cast_info = cast_info_vector[i];
    auto input_idx = cast_info.idx;
    auto input_type_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, input_idx);
    if (NodeNeedCast(input_type_id, cast_info.src_dtypes, cast_info.dst_dtype)) {
      tag = True;
      auto cast_node = AddCastNode(graph, cast_info.dst_dtype, node, true, input_idx);
      // In the process of calling function `AddCastNode`, it has been verified that
      // `input_id + 1` is less than node->inputs().size().
      new_inputs[input_idx + 1] = cast_node;
    }
  }
  if (tag == False) {
    return node;
  }

  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto new_node = kernel_graph->NewCNodeWithInfos(new_inputs, node);
  new_node->set_scope(node->scope());
  new_node->set_fullname_with_scope(node->fullname_with_scope());
  new_node->set_abstract(node->abstract());
  new_node->set_inputs(new_inputs);
  return new_node;
}

const CNodePtr AddCastForGeOutput(const FuncGraphPtr &graph, const CNodePtr &node,
                                  const std::vector<CastInfo> &cast_info_vector) {
  if (cast_info_vector.size() > 1) {
    MS_LOG(EXCEPTION) << "Vector sizes larger than 1 are not currently not supported.";
  }
  auto cast_info = cast_info_vector[0];
  auto output_idx = cast_info.idx;
  auto output_type_id = common::AnfAlgo::GetOutputInferDataType(node, output_idx);
  if (!NodeNeedCast(output_type_id, cast_info.src_dtypes, cast_info.dst_dtype)) {
    return node;
  }
  // set dst type for node
  std::vector<TypeId> output_types;
  std::vector<abstract::BaseShapePtr> shapes;
  auto out_num = AnfAlgo::GetOutputElementNum(node);
  for (size_t i = 0; i < out_num; ++i) {
    auto ele_dtype = common::AnfAlgo::GetOutputInferDataType(node, i);
    if (i == output_idx) {
      output_types.push_back(cast_info.dst_dtype);
    } else {
      output_types.push_back(ele_dtype);
    }
    shapes.push_back(AnfAlgo::GetOutputDetailShape(node, i));
  }
  common::AnfAlgo::SetOutputTypeAndDetailShape(output_types, shapes, node.get());
  // cast node to original dtype
  if (out_num == 1) {
    return AddCastNode(graph, output_type_id, node, false);
  }
  std::vector<AnfNodePtr> new_outputs;
  for (size_t i = 0; i < out_num; ++i) {
    auto tuple_getitem = CreatTupleGetItemNode(graph, node, i);
    if (i == output_idx) {
      tuple_getitem = AddCastNode(graph, output_type_id, tuple_getitem, false);
    }
    (void)new_outputs.emplace_back(std::move(tuple_getitem));
  }
  auto make_tuple = CreateMakeTupleNode(graph, new_outputs);
  make_tuple->set_scope(node->scope());
  return make_tuple;
}
}  // namespace

const BaseRef AddCastForGe::DefinePattern() const {
  VarPtr convert = std::make_shared<CondVar>(NeedAddCast);
  VarPtr inputs = std::make_shared<SeqVar>();
  return VectorRef({convert, inputs});
}

const AnfNodePtr AddCastForGe::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(node);
  auto cast_info = kNeedAddCastMap.at(op_name);
  if (!cast_info.first.empty()) {
    cnode = AddCastForGeInput(graph, cnode, cast_info.first);
  }
  if (!cast_info.second.empty()) {
    cnode = AddCastForGeOutput(graph, cnode, cast_info.second);
  }
  return cnode;
}
}  // namespace opt
}  // namespace mindspore
