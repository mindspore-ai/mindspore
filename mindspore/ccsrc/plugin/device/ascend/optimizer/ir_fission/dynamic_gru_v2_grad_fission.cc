/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/dynamic_gru_v2_grad_fission.h"
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <algorithm>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kDynamicGRUV2GradInputNum = 12;
constexpr size_t kDynamicGRUV2GradOutputNum = 6;
constexpr size_t kGRUV2HiddenGradCellOutputNum = 3;
constexpr size_t kGateNum = 3;
constexpr size_t k3Dims = 3;
constexpr size_t kConcatNum = 2;
constexpr size_t kSplitVOutputNum = 2;
size_t t_size = 0;
size_t batch_size = 0;
size_t hidden_size = 0;
size_t input_size = 0;

std::map<std::string, size_t> input_index = {
  {"x", kIndex1},           {"weight_input", kIndex2}, {"weight_hidden", kIndex3},
  {"y", kIndex4},           {"init_h", kIndex5},       {"h", kIndex6},
  {"dy", kIndex7},          {"dh", kIndex8},           {"update", kIndex9},
  {"reset", kIndex10},      {"new", kIndex11},         {"hidden_new", kIndex12},
  {"seq_length", kIndex13}, {"mask", kIndex14}};

std::map<std::string, size_t> output_index = {{"dw_input", kIndex0},  {"dw_hidden", kIndex1}, {"db_input", kIndex2},
                                              {"db_hidden", kIndex3}, {"dx", kIndex4},        {"dh_prev", kIndex5}};

std::map<std::string, size_t> hidden_grad_input_index = {
  {"dh_pre_t", kIndex1}, {"h", kIndex2},     {"dy", kIndex3},  {"dh", kIndex4},
  {"update", kIndex5},   {"reset", kIndex6}, {"new", kIndex7}, {"hidden_new", kIndex8}};

std::map<std::string, size_t> hidden_grad_output_index = {
  {"dh_prev", kIndex0}, {"dgate_h", kIndex1}, {"dnt_x", kIndex2}};

static CNodePtr CastNodeToSpecificDType(const FuncGraphPtr &func_graph, const AnfNodePtr &node, TypeId dtype) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);

  auto shape = common::AnfAlgo::GetOutputInferShape(node, 0);
  std::vector<AnfNodePtr> cast_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name()))};
  (void)cast_inputs.emplace_back(node);
  auto cast_cnode = NewCNode(cast_inputs, func_graph);
  if (cast_cnode == nullptr) {
    MS_LOG(EXCEPTION) << "CastNodeToSpecificDType failed. node: " << node->fullname_with_scope();
  }
  common::AnfAlgo::SetOutputInferTypeAndShape({dtype}, {shape}, cast_cnode.get());
  common::AnfAlgo::SetNodeAttr(kAttrDstType, TypeIdToType(dtype), cast_cnode);
  return cast_cnode;
}
}  // namespace

AnfNodePtr DynamicGRUV2GradFission::CreateGRUV2HiddenGradCellNode(const FuncGraphPtr &func_graph,
                                                                  const CNodePtr &dynamic_gru_v2_grad_cnode,
                                                                  const AnfNodePtr &last_gru_hidden_grad_node,
                                                                  const AnfNodePtr &last_matmul_node,
                                                                  const std::string &gate_order,
                                                                  const size_t cur_t) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dynamic_gru_v2_grad_cnode);
  const auto &dynamic_gru_v2_grad_inputs = dynamic_gru_v2_grad_cnode->inputs();
  std::vector<AnfNodePtr> gru_v2_hidden_grad_cell_inputs = {
    NewValueNode(std::make_shared<Primitive>(kGRUV2HiddenGradCellOpName))};
  std::vector<AnfNodePtr> dynamic_gru_grad_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, dynamic_gru_v2_grad_cnode, kDynamicGRUV2GradOutputNum,
                                 &dynamic_gru_grad_outputs);
  if (cur_t == 0) {
    (void)gru_v2_hidden_grad_cell_inputs.emplace_back(dynamic_gru_v2_grad_inputs[input_index["dh"]]);
  } else {
    MS_EXCEPTION_IF_NULL(last_gru_hidden_grad_node);
    std::vector<AnfNodePtr> last_gru_hidden_grad_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, last_gru_hidden_grad_node->cast<CNodePtr>(),
                                   kGRUV2HiddenGradCellOutputNum, &last_gru_hidden_grad_outputs);
    (void)gru_v2_hidden_grad_cell_inputs.emplace_back(
      last_gru_hidden_grad_outputs[hidden_grad_output_index["dh_prev"]]);
  }
  if (cur_t < t_size - 1) {
    (void)gru_v2_hidden_grad_cell_inputs.emplace_back(dynamic_gru_v2_grad_inputs[input_index["h"]]);
  } else {
    (void)gru_v2_hidden_grad_cell_inputs.emplace_back(dynamic_gru_v2_grad_inputs[input_index["init_h"]]);
  }
  (void)gru_v2_hidden_grad_cell_inputs.emplace_back(dynamic_gru_v2_grad_inputs[input_index["dy"]]);
  auto input_dh = dynamic_gru_v2_grad_inputs[input_index["dh"]];
  auto dh_dtype = common::AnfAlgo::GetOutputInferDataType(input_dh, 0);
  if (cur_t == 0) {
    (void)gru_v2_hidden_grad_cell_inputs.emplace_back(input_dh);
  } else {
    MS_EXCEPTION_IF_NULL(last_matmul_node);
    (void)gru_v2_hidden_grad_cell_inputs.emplace_back(last_matmul_node);
  }
  (void)gru_v2_hidden_grad_cell_inputs.emplace_back(dynamic_gru_v2_grad_inputs[input_index["update"]]);
  (void)gru_v2_hidden_grad_cell_inputs.emplace_back(dynamic_gru_v2_grad_inputs[input_index["reset"]]);
  (void)gru_v2_hidden_grad_cell_inputs.emplace_back(dynamic_gru_v2_grad_inputs[input_index["new"]]);
  (void)gru_v2_hidden_grad_cell_inputs.emplace_back(dynamic_gru_v2_grad_inputs[input_index["hidden_new"]]);
  auto gru_v2_hidden_grad_cell_op = NewCNode(gru_v2_hidden_grad_cell_inputs, func_graph);

  auto dh_prev_shape = common::AnfAlgo::GetOutputInferShape(dynamic_gru_grad_outputs[output_index["dh_prev"]], 0);
  ShapeVector dgate_h_shape = {1, SizeToLong(batch_size), SizeToLong(kGateNum * hidden_size)};
  ShapeVector dnt_x_shape = {1, SizeToLong(batch_size), SizeToLong(hidden_size)};
  common::AnfAlgo::SetOutputInferTypeAndShape(
    {dh_dtype, dh_dtype, dh_dtype}, {dh_prev_shape, dgate_h_shape, dnt_x_shape}, gru_v2_hidden_grad_cell_op.get());
  common::AnfAlgo::SetNodeAttr("t_state", MakeValue(SizeToLong(cur_t)), gru_v2_hidden_grad_cell_op);
  common::AnfAlgo::SetNodeAttr("gate_order", MakeValue(gate_order), gru_v2_hidden_grad_cell_op);
  return gru_v2_hidden_grad_cell_op;
}

void DynamicGRUV2GradFission::AddTLoopNode(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_gru_v2_grad_cnode,
                                           std::vector<std::vector<AnfNodePtr>> *result_nodes) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dynamic_gru_v2_grad_cnode);
  MS_EXCEPTION_IF_NULL(result_nodes);
  std::string gate_order = "rzh";
  if (common::AnfAlgo::HasNodeAttr("gate_order", dynamic_gru_v2_grad_cnode)) {
    gate_order = common::AnfAlgo::GetNodeAttr<std::string>(dynamic_gru_v2_grad_cnode, "gate_order");
  }
  std::vector<AnfNodePtr> gru_hidden_grad_cells;
  std::vector<AnfNodePtr> matmul_nodes;
  AnfNodePtr last_hidden_grad_node = nullptr;
  AnfNodePtr last_matmul_node = nullptr;
  const auto &dynamic_gru_v2_grad_inputs = dynamic_gru_v2_grad_cnode->inputs();
  for (size_t i = 0; i < t_size; ++i) {
    // Create gru_hidden_grad_cell
    auto gru_hidden_grad_cell_node = CreateGRUV2HiddenGradCellNode(
      func_graph, dynamic_gru_v2_grad_cnode, last_hidden_grad_node, last_matmul_node, gate_order, i);
    // add matmul node
    std::vector<AnfNodePtr> matmul_inputs = {NewValueNode(std::make_shared<Primitive>(kBatchMatMulOpName))};
    auto gru_hidden_grad_cnode = gru_hidden_grad_cell_node->cast<CNodePtr>();
    std::vector<AnfNodePtr> hidden_grad_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, gru_hidden_grad_cnode, kGRUV2HiddenGradCellOutputNum,
                                   &hidden_grad_outputs);
    auto dgate_h = hidden_grad_outputs[hidden_grad_output_index["dgate_h"]];
    auto cast_node = CastNodeToSpecificDType(func_graph, dgate_h, kNumberTypeFloat16);
    (void)matmul_inputs.emplace_back(cast_node);
    auto weight_hidden = dynamic_gru_v2_grad_inputs[input_index["weight_hidden"]];
    std::vector<AnfNodePtr> reshape_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReshape->name())),
                                              weight_hidden};
    auto reshape = NewCNode(reshape_inputs, func_graph);
    auto weight_hidden_dtype = common::AnfAlgo::GetOutputInferDataType(weight_hidden, input_index["weight_hidden"]);
    ShapeVector reshape_out_shape = {1, common::AnfAlgo::GetOutputInferShape(weight_hidden, 0)[0],
                                     common::AnfAlgo::GetOutputInferShape(weight_hidden, 0)[1]};
    common::AnfAlgo::SetOutputInferTypeAndShape({weight_hidden_dtype}, {reshape_out_shape}, reshape.get());

    auto reshape_cast_node = CastNodeToSpecificDType(func_graph, reshape, kNumberTypeFloat16);
    (void)matmul_inputs.emplace_back(reshape_cast_node);
    auto matmul_node = NewCNode(matmul_inputs, func_graph);
    MS_EXCEPTION_IF_NULL(matmul_node);
    ShapeVector out_shape = {1, SizeToLong(batch_size), SizeToLong(hidden_size)};
    common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {out_shape}, matmul_node.get());
    common::AnfAlgo::SetNodeAttr("transpose_x1", MakeValue(false), matmul_node);
    common::AnfAlgo::SetNodeAttr("transpose_x2", MakeValue(true), matmul_node);

    last_hidden_grad_node = gru_hidden_grad_cell_node;
    last_matmul_node = matmul_node;
    (void)gru_hidden_grad_cells.emplace_back(gru_hidden_grad_cell_node);
    (void)matmul_nodes.emplace_back(matmul_node);
  }
  // Add last GRUV2HiddenGradCell node
  auto gru_hidden_grad_cell_node = CreateGRUV2HiddenGradCellNode(
    func_graph, dynamic_gru_v2_grad_cnode, last_hidden_grad_node, last_matmul_node, gate_order, t_size);
  (void)gru_hidden_grad_cells.emplace_back(gru_hidden_grad_cell_node);
  (void)result_nodes->emplace_back(gru_hidden_grad_cells);
  (void)result_nodes->emplace_back(matmul_nodes);
}

AnfNodePtr DynamicGRUV2GradFission::AddTConcatNode(const FuncGraphPtr &func_graph,
                                                   const std::vector<AnfNodePtr> &gru_hidden_grad_nodes,
                                                   size_t concat_output_index) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name()))};
  for (size_t i = 0; i < t_size; i++) {
    auto gru_hidden_grad_node_i = gru_hidden_grad_nodes[(t_size - 1) - i];
    MS_EXCEPTION_IF_NULL(gru_hidden_grad_node_i);
    std::vector<AnfNodePtr> gru_hidden_grad_node_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, gru_hidden_grad_node_i, kGRUV2HiddenGradCellOutputNum,
                                   &gru_hidden_grad_node_outputs);
    (void)concat_inputs.emplace_back(gru_hidden_grad_node_outputs[concat_output_index]);
  }
  auto concat_t_node = NewCNode(concat_inputs, func_graph);
  auto out_dims = common::AnfAlgo::GetOutputInferShape(gru_hidden_grad_nodes[kIndex0], concat_output_index);
  ShapeVector concat_output_shape = {SizeToLong(t_size), out_dims[kDim1], out_dims[kDim2]};
  auto out_type = common::AnfAlgo::GetOutputInferDataType(gru_hidden_grad_nodes[kIndex0], concat_output_index);
  common::AnfAlgo::SetOutputInferTypeAndShape({out_type}, {concat_output_shape}, concat_t_node.get());
  common::AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(t_size)), concat_t_node);
  common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(std::vector<int64_t>{SizeToLong(t_size)}), concat_t_node);
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(static_cast<int64_t>(0)), concat_t_node);
  return concat_t_node;
}

std::vector<AnfNodePtr> DynamicGRUV2GradFission::AddGRUHiddenGradNode(const FuncGraphPtr &func_graph,
                                                                      const CNodePtr &dynamic_gru_v2_grad_cnode) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dynamic_gru_v2_grad_cnode);
  std::vector<AnfNodePtr> result;
  std::vector<std::vector<AnfNodePtr>> result_nodes;
  // add loop t hidden grad nodes; [[hidden_grad_nodes] [matmul_nodes]]
  AddTLoopNode(func_graph, dynamic_gru_v2_grad_cnode, &result_nodes);
  if (result_nodes.empty() || result_nodes[0].empty()) {
    MS_LOG(EXCEPTION) << "result_node is empty, DynamicGRUGrad fission failed.";
  }
  auto gru_hidden_grad_nodes = result_nodes[kIndex0];
  (void)result.emplace_back(gru_hidden_grad_nodes[gru_hidden_grad_nodes.size() - 1]);
  if (t_size > 1) {
    // add dnt_x concat node [t_size, batch_size, hidden_size]
    auto dnt_x_concat_t_node = AddTConcatNode(func_graph, gru_hidden_grad_nodes, hidden_grad_output_index["dnt_x"]);
    // add dgate_h concat node [t_size, batch_size, 3 * hidden_size]
    auto dgate_h_concat_t_node = AddTConcatNode(func_graph, gru_hidden_grad_nodes, hidden_grad_output_index["dgate_h"]);
    (void)result.emplace_back(dgate_h_concat_t_node);
    (void)result.emplace_back(dnt_x_concat_t_node);
  } else {
    auto node = result_nodes[kIndex0][kIndex0];
    (void)result.emplace_back(node);
    (void)result.emplace_back(node);
  }
  return result;
}

AnfNodePtr DynamicGRUV2GradFission::AddHSplitNode(const FuncGraphPtr &func_graph,
                                                  const CNodePtr &dynamic_gru_v2_grad_cnode) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dynamic_gru_v2_grad_cnode);
  auto input_h = dynamic_gru_v2_grad_cnode->input(input_index["h"]);
  std::vector<AnfNodePtr> splitv_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitV->name())),
                                          input_h};
  auto split_v = NewCNode(splitv_input, func_graph);
  // Set infer data type and shape
  auto dtypes = {common::AnfAlgo::GetOutputInferDataType(input_h, 0),
                 common::AnfAlgo::GetOutputInferDataType(input_h, 0)};
  std::vector<size_t> output1_shape = {t_size - 1, batch_size, hidden_size};
  std::vector<size_t> output2_shape = {1, batch_size, hidden_size};
  std::vector<int64_t> split_list = {SizeToLong(t_size - 1), 1};
  std::vector<ShapeVector> shapes = {Convert2Long(output1_shape), Convert2Long(output2_shape)};
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split_v.get());
  // Set attr
  common::AnfAlgo::SetNodeAttr(kAttrSplitDim, MakeValue(SizeToLong(0)), split_v);
  common::AnfAlgo::SetNodeAttr(kAttrNumSplit, MakeValue(SizeToLong(kSplitVOutputNum)), split_v);
  common::AnfAlgo::SetNodeAttr(kAttrSizeSplits, MakeValue(split_list), split_v);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), split_v);
  return split_v;
}

AnfNodePtr DynamicGRUV2GradFission::CreateHReshape(const FuncGraphPtr &graph, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto ori_shape = common::AnfAlgo::GetOutputInferShape(node, 0);
  ShapeVector shape_tmp;
  if (ori_shape.size() == k3Dims) {
    shape_tmp = {ori_shape};
  } else {
    shape_tmp = {{1, ori_shape[kDim0], ori_shape[kDim1]}};
  }
  auto ori_dtype = {common::AnfAlgo::GetOutputInferDataType(node, 0)};
  // reshape
  std::vector<AnfNodePtr> reshape_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReshape->name())), node};
  auto reshape = NewCNode(reshape_input, graph);
  common::AnfAlgo::SetOutputInferTypeAndShape(ori_dtype, {shape_tmp}, reshape.get());
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), reshape);
  return reshape;
}

AnfNodePtr DynamicGRUV2GradFission::AddHConcatNode(const FuncGraphPtr &func_graph,
                                                   const CNodePtr &dynamic_gru_v2_grad_cnode,
                                                   const AnfNodePtr &splitv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dynamic_gru_v2_grad_cnode);
  MS_EXCEPTION_IF_NULL(splitv);
  // Create node
  std::vector<AnfNodePtr> splitv_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, splitv, kSplitVOutputNum, &splitv_outputs);
  if (splitv_outputs.size() != kSplitVOutputNum) {
    MS_LOG(EXCEPTION) << "Create outputs of node " << splitv->DebugString() << " failed"
                      << trace::DumpSourceLines(splitv);
  }
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name()))};
  auto init_h_reshape = CreateHReshape(func_graph, dynamic_gru_v2_grad_cnode->input(input_index["init_h"]));
  (void)concat_inputs.emplace_back(init_h_reshape);
  (void)concat_inputs.emplace_back(splitv_outputs[kIndex0]);
  auto concat = NewCNode(concat_inputs, func_graph);
  // Set infer data type and shape
  ShapeVector output_shape = Convert2Long({t_size, batch_size, hidden_size});
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(init_h_reshape, 0)},
                                              {output_shape}, concat.get());
  // Set attr
  common::AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(kConcatNum)), concat);
  common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(std::vector<int64_t>{kConcatNum}), concat);
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(SizeToLong(0)), concat);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), concat);
  return concat;
}

AnfNodePtr DynamicGRUV2GradFission::AddDwhMatmulNode(const FuncGraphPtr &func_graph, const AnfNodePtr &dgate_h,
                                                     const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dgate_h);
  MS_EXCEPTION_IF_NULL(node);
  // BatchMatMul
  auto cast_node = CastNodeToSpecificDType(func_graph, node, kNumberTypeFloat16);
  std::vector<AnfNodePtr> matmul_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimBatchMatMul->name()))};
  (void)matmul_inputs.emplace_back(cast_node);

  CNodePtr dgate_h_cast_node;
  if (t_size == 1) {
    std::vector<AnfNodePtr> dgate_h_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, dgate_h, kGRUV2HiddenGradCellOutputNum, &dgate_h_outputs);
    dgate_h_cast_node =
      CastNodeToSpecificDType(func_graph, dgate_h_outputs[hidden_grad_output_index["dgate_h"]], kNumberTypeFloat16);
  } else {
    dgate_h_cast_node = CastNodeToSpecificDType(func_graph, dgate_h, kNumberTypeFloat16);
  }
  matmul_inputs.emplace_back(dgate_h_cast_node);
  auto batch_matmul = NewCNode(matmul_inputs, func_graph);
  ShapeVector shape = Convert2Long({t_size, hidden_size, kGateNum * hidden_size});
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {shape}, batch_matmul.get());
  common::AnfAlgo::SetNodeAttr("transpose_x1", MakeValue(true), batch_matmul);
  common::AnfAlgo::SetNodeAttr("transpose_x2", MakeValue(false), batch_matmul);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), batch_matmul);
  return batch_matmul;
}

AnfNodePtr DynamicGRUV2GradFission::CreateDgateHSplitVDNode(const FuncGraphPtr &func_graph,
                                                            const AnfNodePtr &dgate_h) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dgate_h);
  std::vector<AnfNodePtr> splitvd_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitV->name()))};
  if (t_size == 1) {
    std::vector<AnfNodePtr> dgate_h_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, dgate_h, kGRUV2HiddenGradCellOutputNum, &dgate_h_outputs);
    (void)splitvd_input.emplace_back(dgate_h_outputs[hidden_grad_output_index["dgate_h"]]);
  } else {
    (void)splitvd_input.emplace_back(dgate_h);
  }
  auto split_vd = NewCNode(splitvd_input, func_graph);
  auto dtypes = {common::AnfAlgo::GetOutputInferDataType(dgate_h, 0),
                 common::AnfAlgo::GetOutputInferDataType(dgate_h, 0)};
  auto shape = Convert2Long({t_size, batch_size, hidden_size << 1});
  auto shape2 = Convert2Long({t_size, batch_size, hidden_size});
  std::vector<ShapeVector> shapes = {shape, shape2};
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split_vd.get());
  common::AnfAlgo::SetNodeAttr("split_dim", MakeValue(SizeToLong(kDim2)), split_vd);
  common::AnfAlgo::SetNodeAttr("num_split", MakeValue(SizeToLong(kSplitVOutputNum)), split_vd);
  std::vector<int64_t> size_splits = {SizeToLong(hidden_size << 1), SizeToLong(hidden_size)};
  common::AnfAlgo::SetNodeAttr("size_splits", MakeValue(size_splits), split_vd);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), split_vd);
  return split_vd;
}

AnfNodePtr DynamicGRUV2GradFission::CreateDgateXConcatDNode(const FuncGraphPtr &func_graph, const AnfNodePtr &split,
                                                            const AnfNodePtr &dnt_x) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(split);
  MS_EXCEPTION_IF_NULL(dnt_x);
  std::vector<AnfNodePtr> split_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, split, kSplitVOutputNum, &split_outputs);
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name())),
                                           split_outputs[kIndex0]};
  if (t_size == 1) {
    std::vector<AnfNodePtr> dnt_x_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, dnt_x, kGRUV2HiddenGradCellOutputNum, &dnt_x_outputs);
    (void)concat_inputs.emplace_back(dnt_x_outputs[hidden_grad_output_index["dnt_x"]]);
  } else {
    (void)concat_inputs.emplace_back(dnt_x);
  }
  auto concat_op = NewCNode(concat_inputs, func_graph);
  auto shape = Convert2Long({t_size, batch_size, kGateNum * hidden_size});
  auto types = {common::AnfAlgo::GetOutputInferDataType(dnt_x, 0)};
  common::AnfAlgo::SetOutputInferTypeAndShape(types, {shape}, concat_op.get());
  common::AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(kConcatNum)), concat_op);
  common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(std::vector<int64_t>{kConcatNum}), concat_op);
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(SizeToLong(kDim2)), concat_op);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), concat_op);
  return concat_op;
}

AnfNodePtr DynamicGRUV2GradFission::CreateDwxBatchMatMul(const FuncGraphPtr &graph, const AnfNodePtr &node1,
                                                         const AnfNodePtr &node2) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);

  auto cast_node1 = CastNodeToSpecificDType(graph, node1, kNumberTypeFloat16);
  auto cast_node2 = CastNodeToSpecificDType(graph, node2, kNumberTypeFloat16);
  std::vector<AnfNodePtr> matmul_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimBatchMatMul->name())),
                                           cast_node1, cast_node2};

  // BatchMatMul
  auto batch_matmul = NewCNode(matmul_inputs, graph);
  MS_EXCEPTION_IF_NULL(batch_matmul);
  auto shape = Convert2Long({t_size, input_size, kGateNum * hidden_size});
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {shape}, batch_matmul.get());
  common::AnfAlgo::SetNodeAttr("transpose_x1", MakeValue(true), batch_matmul);
  common::AnfAlgo::SetNodeAttr("transpose_x2", MakeValue(false), batch_matmul);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), batch_matmul);
  return batch_matmul;
}

AnfNodePtr DynamicGRUV2GradFission::CreateDxtBatchMatMul(const FuncGraphPtr &func_graph, const AnfNodePtr &dgate_concat,
                                                         const AnfNodePtr &weight_input, const AnfNodePtr &dx) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dgate_concat);
  MS_EXCEPTION_IF_NULL(weight_input);
  MS_EXCEPTION_IF_NULL(dx);

  auto dgate_concat_cast = CastNodeToSpecificDType(func_graph, dgate_concat, kNumberTypeFloat16);
  auto weight_input_cast = CastNodeToSpecificDType(func_graph, weight_input, kNumberTypeFloat16);
  std::vector<AnfNodePtr> matmul_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimBatchMatMul->name())),
                                           dgate_concat_cast, weight_input_cast};

  auto batch_matmul = NewCNode(matmul_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(batch_matmul);
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {common::AnfAlgo::GetOutputInferShape(dx, 0)},
                                              batch_matmul.get());
  common::AnfAlgo::SetNodeAttr("transpose_x1", MakeValue(false), batch_matmul);
  common::AnfAlgo::SetNodeAttr("transpose_x2", MakeValue(true), batch_matmul);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), batch_matmul);
  return batch_matmul;
}

AnfNodePtr DynamicGRUV2GradFission::CreateWBroadcastToDNode(const FuncGraphPtr &graph, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  // BroadcastTo
  std::vector<AnfNodePtr> braodcast_to_input = {NewValueNode(std::make_shared<Primitive>(kBroadcastToOpName)), node};
  auto broadcast_to_d = NewCNode(braodcast_to_input, graph);
  auto shape = Convert2Long({t_size, input_size, kGateNum * hidden_size});
  auto type = {common::AnfAlgo::GetOutputInferDataType(node, 0)};
  common::AnfAlgo::SetOutputInferTypeAndShape(type, {shape}, broadcast_to_d.get());
  std::vector<int64_t> attr_shape = {SizeToLong(t_size), SizeToLong(input_size), SizeToLong(kGateNum * hidden_size)};
  common::AnfAlgo::SetNodeAttr(kAttrShape, MakeValue(attr_shape), broadcast_to_d);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), broadcast_to_d);
  return broadcast_to_d;
}

AnfNodePtr DynamicGRUV2GradFission::CreateDwReduceSumDNode(const FuncGraphPtr &graph, const AnfNodePtr &matmul,
                                                           const AnfNodePtr &gru_grad) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(matmul);
  MS_EXCEPTION_IF_NULL(gru_grad);
  // ReduceSumD for dw_x and dw_h
  std::vector<AnfNodePtr> reducesum_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceSum->name())),
                                              matmul};
  auto reduce_sumd = NewCNode(reducesum_inputs, graph);
  auto types = {common::AnfAlgo::GetOutputInferDataType(gru_grad, 0)};
  auto shapes = {common::AnfAlgo::GetOutputInferShape(gru_grad, 0)};
  common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, reduce_sumd.get());
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(std::vector<int64_t>{0}), reduce_sumd);
  common::AnfAlgo::SetNodeAttr("keep_dims", MakeValue(false), reduce_sumd);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), reduce_sumd);
  return reduce_sumd;
}

AnfNodePtr DynamicGRUV2GradFission::CreateDbReduceSumDNode(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                           const AnfNodePtr &node2) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node2);
  // ReduceSumD for db_x and db_h
  std::vector<AnfNodePtr> reducesum_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceSum->name())),
                                              node};
  auto reduce_sumd = NewCNode(reducesum_inputs, graph);
  MS_EXCEPTION_IF_NULL(reduce_sumd);
  auto shape = Convert2Long({kGateNum * hidden_size});
  auto types = {common::AnfAlgo::GetOutputInferDataType(node2, 0)};
  common::AnfAlgo::SetOutputInferTypeAndShape(types, {shape}, reduce_sumd.get());
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(std::vector<int64_t>{0, 1}), reduce_sumd);
  common::AnfAlgo::SetNodeAttr("keep_dims", MakeValue(false), reduce_sumd);
  common::AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), reduce_sumd);
  return reduce_sumd;
}

const BaseRef DynamicGRUV2GradFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimDynamicGRUV2Grad, Xs});
}

const AnfNodePtr DynamicGRUV2GradFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto dynamic_gru_v2_grad_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dynamic_gru_v2_grad_cnode);
  if (dynamic_gru_v2_grad_cnode->size() < kDynamicGRUV2GradInputNum + 1) {
    MS_LOG(INFO) << "The node " << dynamic_gru_v2_grad_cnode->DebugString() << " has less than "
                 << kDynamicGRUV2GradInputNum << " inputs";
    return nullptr;
  }
  if (common::AnfAlgo::IsDynamicShape(node)) {
    MS_LOG(INFO) << "DynamicGRUV2Grad is dynamic shape, can not do fission.";
    return nullptr;
  }

  // input_list of dynamic_gru_v2_grad
  const auto &ori_inputs = dynamic_gru_v2_grad_cnode->inputs();
  std::vector<AnfNodePtr> gru_grad_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, dynamic_gru_v2_grad_cnode, kDynamicGRUV2GradOutputNum, &gru_grad_outputs);
  auto input_h = ori_inputs[input_index["h"]];
  auto input_x = ori_inputs[input_index["x"]];
  t_size = LongToSize(common::AnfAlgo::GetOutputInferShape(input_h, 0)[kDim0]);
  batch_size = LongToSize(common::AnfAlgo::GetOutputInferShape(input_h, 0)[kDim1]);
  hidden_size = LongToSize(common::AnfAlgo::GetOutputInferShape(input_h, 0)[kDim2]);
  input_size = LongToSize(common::AnfAlgo::GetOutputInferShape(input_x, 0)[kDim2]);
  MS_LOG(INFO) << "For DynamicGRUV2Grad op, t_size: " << t_size << ", batch_size: " << batch_size
               << ", hidden_size: " << hidden_size << ", input_size: " << input_size;
  // add GRUHiddenGrad {dhPrevNode, dgateHConcatTNode, dntXConcatTNode}
  std::vector<AnfNodePtr> gru_hidden_grad_nodes = AddGRUHiddenGradNode(func_graph, dynamic_gru_v2_grad_cnode);
  AnfNodePtr dwh_matmul_node;
  auto dgate_h = gru_hidden_grad_nodes[hidden_grad_output_index["dgate_h"]];
  if (t_size != 1) {
    // split h
    auto split = AddHSplitNode(func_graph, dynamic_gru_v2_grad_cnode);
    // concat(h, h_split)
    auto h_concat = AddHConcatNode(func_graph, dynamic_gru_v2_grad_cnode, split);
    // add matmul(h_prev.T, dgate_h)
    dwh_matmul_node = AddDwhMatmulNode(func_graph, dgate_h, h_concat);
  } else {
    auto reshape = CreateHReshape(func_graph, ori_inputs[input_index["init_h"]]);
    dwh_matmul_node = AddDwhMatmulNode(func_graph, dgate_h, reshape);
  }
  // split dgate_h to [dit, drt] and [dnt_h]
  auto dgate_h_split = CreateDgateHSplitVDNode(func_graph, dgate_h);
  // concat(dgate_h_split[0], dnt_x) to dgate_x
  auto dgate_x_concat =
    CreateDgateXConcatDNode(func_graph, dgate_h_split, gru_hidden_grad_nodes[hidden_grad_output_index["dnt_x"]]);
  // broadcast weight_input [input_size, 3 * hidden_size] to [t_size, input_size, 3 * hidden_size]
  auto w_input_broadcast = CreateWBroadcastToDNode(func_graph, ori_inputs[input_index["weight_input"]]);
  // batchmatmul(dgate_x_concat, w_input_broadcast.T)
  auto dxt_batch_matmul =
    CreateDxtBatchMatMul(func_graph, dgate_x_concat, w_input_broadcast, gru_grad_outputs[output_index["dx"]]);
  // batchmatmul(x.T, dgate_x_concat)
  auto dwx_batch_matmul = CreateDwxBatchMatMul(func_graph, ori_inputs[input_index["x"]], dgate_x_concat);
  // reducesum dw_x and dw_h
  auto dwx_reduce_sum =
    CreateDwReduceSumDNode(func_graph, dwx_batch_matmul, gru_grad_outputs[output_index["dw_input"]]);
  auto dwh_reduce_sum =
    CreateDwReduceSumDNode(func_graph, dwh_matmul_node, gru_grad_outputs[output_index["dw_hidden"]]);
  // reducesum db_x and db_h
  auto dbx_reduce_sum = CreateDbReduceSumDNode(func_graph, dgate_x_concat, ori_inputs[kIndex5]);
  AnfNodePtr dbh_reduce_sum;
  if (t_size == 1) {
    std::vector<AnfNodePtr> dbh_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, dgate_h, kGRUV2HiddenGradCellOutputNum, &dbh_outputs);
    dbh_reduce_sum = CreateDbReduceSumDNode(func_graph, dbh_outputs[kIndex1], ori_inputs[kIndex5]);
  } else {
    dbh_reduce_sum = CreateDbReduceSumDNode(func_graph, dgate_h, ori_inputs[kIndex5]);
  }
  std::vector<AnfNodePtr> dh_prev_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, gru_hidden_grad_nodes[kIndex0], kGRUV2HiddenGradCellOutputNum,
                                 &dh_prev_outputs);
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple),
                                               dwx_reduce_sum,
                                               dwh_reduce_sum,
                                               dbx_reduce_sum,
                                               dbh_reduce_sum,
                                               dxt_batch_matmul,
                                               dh_prev_outputs[kIndex0]};
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}
}  // namespace opt
}  // namespace mindspore
