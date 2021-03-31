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
#include "backend/optimizer/ascend/ir_fission/dynamic_gru_v2_grad_fission.h"
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kDynamicGRUV2GradInputNum = 12;
constexpr size_t kDynamicGRUV2GradOutputNum = 6;
constexpr size_t kSplitVOutputNum = 2;
constexpr size_t kGRUV2HiddenGradOutputNum = 3;

AnfNodePtr CreateGRUV2HiddenGradNode(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &dynamic_gru_v2_grad_inputs = cnode->inputs();
  std::vector<AnfNodePtr> gru_v2_hidden_grad_inputs = {
    NewValueNode(std::make_shared<Primitive>(kGRUV2HiddenGradOpName)),
    dynamic_gru_v2_grad_inputs[3],
    dynamic_gru_v2_grad_inputs[5],
    dynamic_gru_v2_grad_inputs[6],
    dynamic_gru_v2_grad_inputs[7],
    dynamic_gru_v2_grad_inputs[8],
    dynamic_gru_v2_grad_inputs[9],
    dynamic_gru_v2_grad_inputs[10],
    dynamic_gru_v2_grad_inputs[11],
    dynamic_gru_v2_grad_inputs[12]};

  std::vector<AnfNodePtr> ori_outputs;
  CreateMultipleOutputsOfAnfNode(graph, node, kDynamicGRUV2GradOutputNum, &ori_outputs);
  auto gru_v2_hidden_grad_op = graph->NewCNode(gru_v2_hidden_grad_inputs);
  MS_EXCEPTION_IF_NULL(gru_v2_hidden_grad_op);
  auto h_dtype = AnfAlgo::GetOutputInferDataType(dynamic_gru_v2_grad_inputs[6], 0);
  auto types = {h_dtype, h_dtype, h_dtype};
  std::vector<size_t> dh_preh_shape = AnfAlgo::GetOutputInferShape(ori_outputs[5], 0);
  std::vector<size_t> dgate_h_shape = {AnfAlgo::GetOutputInferShape(dynamic_gru_v2_grad_inputs[6], 0)[0],
                                       AnfAlgo::GetOutputInferShape(dynamic_gru_v2_grad_inputs[6], 0)[1],
                                       3 * AnfAlgo::GetOutputInferShape(dynamic_gru_v2_grad_inputs[6], 0)[2]};
  std::vector<size_t> dnx_t_shape = AnfAlgo::GetOutputInferShape(dynamic_gru_v2_grad_inputs[6], 0);
  auto shapes = {dh_preh_shape, dgate_h_shape, dnx_t_shape};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, gru_v2_hidden_grad_op.get());
  auto gate_order = AnfAlgo::GetNodeAttr<std::string>(cnode, "gate_order");
  AnfAlgo::SetNodeAttr("gate_order", MakeValue(gate_order), gru_v2_hidden_grad_op);
  return gru_v2_hidden_grad_op;
}

AnfNodePtr CreateHSplitVDNode(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  // SplitV
  std::vector<AnfNodePtr> splitvd_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitV->name())), node};
  auto split_vd = graph->NewCNode(splitvd_input);
  MS_EXCEPTION_IF_NULL(split_vd);
  auto dtypes = {AnfAlgo::GetOutputInferDataType(node, 0), AnfAlgo::GetOutputInferDataType(node, 0)};
  size_t t_size = AnfAlgo::GetOutputInferShape(node, 0)[0];
  size_t batch = AnfAlgo::GetOutputInferShape(node, 0)[1];
  size_t hidden_size = AnfAlgo::GetOutputInferShape(node, 0)[2];
  std::vector<size_t> shape = {t_size - IntToSize(1), batch, hidden_size};
  std::vector<size_t> shape2 = {IntToSize(1), batch, hidden_size};
  std::vector<std::vector<size_t>> shapes = {shape, shape2};
  AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split_vd.get());
  AnfAlgo::SetNodeAttr("split_dim", MakeValue(SizeToLong(0)), split_vd);
  AnfAlgo::SetNodeAttr("num_split", MakeValue(SizeToLong(2)), split_vd);
  std::vector<int64_t> size_splits = {SizeToLong(t_size - 1), SizeToLong(1)};
  AnfAlgo::SetNodeAttr("size_splits", MakeValue(size_splits), split_vd);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), split_vd);
  return split_vd;
}

AnfNodePtr CreateHReshape(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto ori_shape = AnfAlgo::GetOutputInferShape(node, 0);
  std::vector<std::vector<size_t>> shape_tmp;
  if (ori_shape.size() == 3) {
    shape_tmp = {ori_shape};
  } else {
    shape_tmp = {{IntToSize(1), ori_shape[0], ori_shape[1]}};
  }
  auto ori_dtype = {AnfAlgo::GetOutputInferDataType(node, 0)};
  // reshape
  std::vector<AnfNodePtr> reshape_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReshape->name())), node};
  auto reshape = graph->NewCNode(reshape_input);
  AnfAlgo::SetOutputInferTypeAndShape(ori_dtype, shape_tmp, reshape.get());
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), reshape);
  return reshape;
}

AnfNodePtr CreateHConcatDNode(const FuncGraphPtr &graph, const AnfNodePtr &node1, const AnfNodePtr &node2) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  std::vector<AnfNodePtr> ori_outputs;
  CreateMultipleOutputsOfAnfNode(graph, node2, 2, &ori_outputs);
  auto reshape = CreateHReshape(graph, node1);

  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name())),
                                           reshape, ori_outputs[0]};
  auto concat_op = graph->NewCNode(concat_inputs);
  MS_EXCEPTION_IF_NULL(concat_op);

  std::vector<size_t> shape = {AnfAlgo::GetOutputInferShape(node2, 0)[0] + 1, AnfAlgo::GetOutputInferShape(node2, 0)[1],
                               AnfAlgo::GetOutputInferShape(node2, 0)[2]};
  auto types = {AnfAlgo::GetOutputInferDataType(node2, 0)};
  AnfAlgo::SetOutputInferTypeAndShape(types, {shape}, concat_op.get());
  AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(2)), concat_op);
  AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(std::vector<int64_t>{2}), concat_op);
  AnfAlgo::SetNodeAttr("axis", MakeValue(SizeToLong(0)), concat_op);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), concat_op);
  return concat_op;
}

AnfNodePtr CreateDgateHSplitVDNode(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  // SplitV
  std::vector<AnfNodePtr> splitvd_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitV->name())), node};
  auto split_vd = graph->NewCNode(splitvd_input);
  MS_EXCEPTION_IF_NULL(split_vd);
  auto dtypes = {AnfAlgo::GetOutputInferDataType(node, 0), AnfAlgo::GetOutputInferDataType(node, 0)};
  size_t t_size = AnfAlgo::GetOutputInferShape(node, 0)[0];
  size_t batch = AnfAlgo::GetOutputInferShape(node, 0)[1];
  size_t hidden_size = AnfAlgo::GetOutputInferShape(node, 0)[2] / 3;
  std::vector<size_t> shape = {t_size, batch, 2 * hidden_size};
  std::vector<size_t> shape2 = {t_size, batch, hidden_size};
  std::vector<std::vector<size_t>> shapes = {shape, shape2};
  AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split_vd.get());
  AnfAlgo::SetNodeAttr("split_dim", MakeValue(SizeToLong(2)), split_vd);
  AnfAlgo::SetNodeAttr("num_split", MakeValue(SizeToLong(2)), split_vd);
  std::vector<int64_t> size_splits = {2 * SizeToLong(hidden_size), SizeToLong(hidden_size)};
  AnfAlgo::SetNodeAttr("size_splits", MakeValue(size_splits), split_vd);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), split_vd);
  return split_vd;
}

AnfNodePtr CreateDgateXConcatDNode(const FuncGraphPtr &graph, const AnfNodePtr &node1, const AnfNodePtr &node2) {
  MS_EXCEPTION_IF_NULL(graph);
  // node1: dgate_h_split
  // node2: dnt_x
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  std::vector<AnfNodePtr> ori_outputs;
  CreateMultipleOutputsOfAnfNode(graph, node1, 2, &ori_outputs);

  // ConcatD
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name())),
                                           ori_outputs[0], node2};
  auto concat_op = graph->NewCNode(concat_inputs);
  MS_EXCEPTION_IF_NULL(concat_op);
  std::vector<size_t> shape = {AnfAlgo::GetOutputInferShape(node2, 0)[0], AnfAlgo::GetOutputInferShape(node2, 0)[1],
                               AnfAlgo::GetOutputInferShape(node1, 0)[2] + AnfAlgo::GetOutputInferShape(node2, 0)[2]};
  auto types = {AnfAlgo::GetOutputInferDataType(node2, 0)};
  AnfAlgo::SetOutputInferTypeAndShape(types, {shape}, concat_op.get());
  AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(2)), concat_op);
  AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(std::vector<int64_t>{2}), concat_op);
  AnfAlgo::SetNodeAttr("axis", MakeValue(SizeToLong(2)), concat_op);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), concat_op);
  return concat_op;
}

AnfNodePtr CreateWBroadcastToDNode(const FuncGraphPtr &graph, const AnfNodePtr &node1, const AnfNodePtr &node2) {
  MS_EXCEPTION_IF_NULL(graph);
  // node1 : input node
  // node2 : orign_input x
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  // BroadcastTo
  std::vector<AnfNodePtr> braodcast_to_input = {NewValueNode(std::make_shared<Primitive>(kBroadcastToOpName)), node1};
  auto broadcast_to_d = graph->NewCNode(braodcast_to_input);
  MS_EXCEPTION_IF_NULL(broadcast_to_d);
  size_t t_size = AnfAlgo::GetOutputInferShape(node2, 0)[0];
  size_t batch = AnfAlgo::GetOutputInferShape(node1, 0)[0];
  size_t gate_size = AnfAlgo::GetOutputInferShape(node1, 0)[1];
  std::vector<size_t> shape = {t_size, batch, gate_size};
  auto type = {AnfAlgo::GetOutputInferDataType(node1, 0)};
  AnfAlgo::SetOutputInferTypeAndShape(type, {shape}, broadcast_to_d.get());

  std::vector<int64_t> attr_shape = {SizeToLong(t_size), SizeToLong(batch), SizeToLong(gate_size)};
  AnfAlgo::SetNodeAttr("shape", MakeValue(attr_shape), broadcast_to_d);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), broadcast_to_d);
  return broadcast_to_d;
}

AnfNodePtr CreateDhxBatchMatMul(const FuncGraphPtr &graph, const AnfNodePtr &node1, const AnfNodePtr &node2) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  // BatchMatMul
  std::vector<AnfNodePtr> matmul_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimBatchMatMul->name())),
                                           node1, node2};
  auto batch_matmul = graph->NewCNode(matmul_inputs);
  MS_EXCEPTION_IF_NULL(batch_matmul);
  std::vector<size_t> shape = {AnfAlgo::GetOutputInferShape(node1, 0)[0], AnfAlgo::GetOutputInferShape(node1, 0)[2],
                               AnfAlgo::GetOutputInferShape(node2, 0)[2]};
  AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat16}, {shape}, batch_matmul.get());
  AnfAlgo::SetNodeAttr("transpose_x1", MakeValue(true), batch_matmul);
  AnfAlgo::SetNodeAttr("transpose_x2", MakeValue(false), batch_matmul);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), batch_matmul);
  return batch_matmul;
}

AnfNodePtr CreateDwhBatchMatMul(const FuncGraphPtr &graph, const AnfNodePtr &node1, const AnfNodePtr &node2) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  // BatchMatMul
  std::vector<AnfNodePtr> matmul_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimBatchMatMul->name())),
                                           node1, node2};
  auto batch_matmul = graph->NewCNode(matmul_inputs);
  MS_EXCEPTION_IF_NULL(batch_matmul);
  std::vector<size_t> shape = {AnfAlgo::GetOutputInferShape(node1, 0)[0], AnfAlgo::GetOutputInferShape(node1, 0)[1],
                               AnfAlgo::GetOutputInferShape(node2, 0)[1]};
  AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat16}, {shape}, batch_matmul.get());
  AnfAlgo::SetNodeAttr("transpose_x1", MakeValue(false), batch_matmul);
  AnfAlgo::SetNodeAttr("transpose_x2", MakeValue(true), batch_matmul);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), batch_matmul);
  return batch_matmul;
}

AnfNodePtr CreateDwReduceSumDNode(const FuncGraphPtr &graph, const AnfNodePtr &node, const AnfNodePtr &node2) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  // ReduceSumD for dw_x and dw_h
  std::vector<AnfNodePtr> reducesum_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceSum->name())),
                                              node};
  auto reduce_sumd = graph->NewCNode(reducesum_inputs);
  MS_EXCEPTION_IF_NULL(reduce_sumd);
  auto types = {AnfAlgo::GetOutputInferDataType(node, 0)};
  auto shapes = {AnfAlgo::GetOutputInferShape(node2, 0)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, reduce_sumd.get());
  AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(std::vector<int64_t>{0}), reduce_sumd);
  AnfAlgo::SetNodeAttr("keep_dims", MakeValue(false), reduce_sumd);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), reduce_sumd);
  return reduce_sumd;
}

AnfNodePtr CreateDbReduceSumDNode(const FuncGraphPtr &graph, const AnfNodePtr &node, const AnfNodePtr &node2) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node2);
  // ReduceSumD for db_x and db_h
  std::vector<AnfNodePtr> reducesum_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceSum->name())),
                                              node};
  auto reduce_sumd = graph->NewCNode(reducesum_inputs);
  MS_EXCEPTION_IF_NULL(reduce_sumd);

  auto types = {AnfAlgo::GetOutputInferDataType(node, 0)};
  std::vector<size_t> shape = {3 * AnfAlgo::GetOutputInferShape(node2, 0)[1]};
  AnfAlgo::SetOutputInferTypeAndShape(types, {shape}, reduce_sumd.get());
  AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(std::vector<int64_t>{0, 1}), reduce_sumd);
  AnfAlgo::SetNodeAttr("keep_dims", MakeValue(false), reduce_sumd);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), reduce_sumd);
  return reduce_sumd;
}
}  // namespace

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

  // input_list of dynamic_gru_v2_grad
  const auto &ori_inputs = dynamic_gru_v2_grad_cnode->inputs();
  // add gru_v2_gru_hidden
  auto gru_v2_gru_hidden = CreateGRUV2HiddenGradNode(func_graph, dynamic_gru_v2_grad_cnode);
  std::vector<AnfNodePtr> gru_hidden_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, gru_v2_gru_hidden, kGRUV2HiddenGradOutputNum, &gru_hidden_outputs);
  size_t step_num = AnfAlgo::GetOutputInferShape(ori_inputs[1], 0)[0];
  AnfNodePtr dwh_batch_matmul = nullptr;
  if (step_num != 1) {
    // split h
    auto h_split = CreateHSplitVDNode(func_graph, ori_inputs[6]);
    // concat(h, h_split)
    auto h_concat = CreateHConcatDNode(func_graph, ori_inputs[5], h_split);
    // batchmatmul(h_concat.T, dgate_h)
    dwh_batch_matmul = CreateDhxBatchMatMul(func_graph, h_concat, gru_hidden_outputs[1]);
  } else {
    auto reshape = CreateHReshape(func_graph, ori_inputs[5]);
    // batchmatmul(init_h.T, dgate_h)
    dwh_batch_matmul = CreateDhxBatchMatMul(func_graph, reshape, gru_hidden_outputs[1]);
  }
  // split dgate_h
  auto dgate_h_split = CreateDgateHSplitVDNode(func_graph, gru_hidden_outputs[1]);
  // concat(dgate_h_split[0], dnt_x) to dgate_x
  auto dgate_x_concat = CreateDgateXConcatDNode(func_graph, dgate_h_split, gru_hidden_outputs[2]);
  // broadcast weight_input [input_size, 3 * hidden_size] to [t_size, input_size, 3 * hidden_size]
  auto w_input_broadcast = CreateWBroadcastToDNode(func_graph, ori_inputs[2], ori_inputs[1]);
  // batchmatmul(x.T, dgate_x_concat)
  auto dwx_batch_matmul = CreateDhxBatchMatMul(func_graph, ori_inputs[1], dgate_x_concat);
  // batchmatmul(dgate_x_concat, w_input_broadcast.T)
  auto dxt_batch_matmul = CreateDwhBatchMatMul(func_graph, dgate_x_concat, w_input_broadcast);
  // reducesum dw_x and dw_h
  auto dwx_reduce_sum = CreateDwReduceSumDNode(func_graph, dwx_batch_matmul, ori_inputs[2]);
  auto dwh_reduce_sum = CreateDwReduceSumDNode(func_graph, dwh_batch_matmul, ori_inputs[3]);
  // reducesum db_x and db_h
  auto dbx_reduce_sum = CreateDbReduceSumDNode(func_graph, dgate_x_concat, ori_inputs[5]);
  auto dbh_reduce_sum = CreateDbReduceSumDNode(func_graph, gru_hidden_outputs[1], ori_inputs[5]);
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple),
                                               dwx_reduce_sum,
                                               dwh_reduce_sum,
                                               dbx_reduce_sum,
                                               dbh_reduce_sum,
                                               dxt_batch_matmul,
                                               gru_hidden_outputs[0]};
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  return make_tuple;
}
}  // namespace opt
}  // namespace mindspore
