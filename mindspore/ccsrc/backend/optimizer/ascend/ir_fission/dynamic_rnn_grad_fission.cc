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
#include "backend/optimizer/ascend/ir_fission/dynamic_rnn_grad_fission.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
constexpr size_t kDynamicRNNGradInputNum = 16;
constexpr size_t kLSTMInputGradOutputNum = 4;
const BaseRef DynamicRNNGradFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimDynamicRNNGrad, Xs});
}

AnfNodePtr CreateSplitVD(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  // SplitV
  std::vector<AnfNodePtr> splitvd_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitV->name())), node};
  auto split_vd = graph->NewCNode(splitvd_input);
  MS_EXCEPTION_IF_NULL(split_vd);
  auto dtypes = {AnfAlgo::GetOutputInferDataType(node, 0), AnfAlgo::GetOutputInferDataType(node, 0)};
  std::vector<size_t> shape = {AnfAlgo::GetOutputInferShape(node, 0)[0] - 1, AnfAlgo::GetOutputInferShape(node, 0)[1],
                               AnfAlgo::GetOutputInferShape(node, 0)[2]};
  auto shape2 = {IntToSize(1), AnfAlgo::GetOutputInferShape(node, 0)[1], AnfAlgo::GetOutputInferShape(node, 0)[2]};
  std::vector<std::vector<size_t>> shapes = {shape, shape2};
  AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split_vd.get());
  AnfAlgo::SetNodeAttr("split_dim", MakeValue(0), split_vd);
  AnfAlgo::SetNodeAttr("num_split", MakeValue(2), split_vd);
  int tmp = SizeToInt(AnfAlgo::GetOutputInferShape(node, 0)[0]) - 1;
  AnfAlgo::SetNodeAttr("size_splits", MakeValue(std::vector<int>{tmp, 1}), split_vd);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), split_vd);
  return split_vd;
}

AnfNodePtr CreateLSTMInputGrad(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &dynamic_rnn_grad_inputs = cnode->inputs();
  std::vector<AnfNodePtr> lstm_input_grad_inputs = {NewValueNode(std::make_shared<Primitive>(kLSTMInputGradOpName)),
                                                    dynamic_rnn_grad_inputs[2],
                                                    dynamic_rnn_grad_inputs[6],
                                                    dynamic_rnn_grad_inputs[8],
                                                    dynamic_rnn_grad_inputs[9],
                                                    dynamic_rnn_grad_inputs[10],
                                                    dynamic_rnn_grad_inputs[11],
                                                    dynamic_rnn_grad_inputs[12],
                                                    dynamic_rnn_grad_inputs[13],
                                                    dynamic_rnn_grad_inputs[14],
                                                    dynamic_rnn_grad_inputs[15],
                                                    dynamic_rnn_grad_inputs[16]};
  std::vector<AnfNodePtr> ori_outputs;
  CreateMultipleOutputsOfAnfNode(graph, node, 5, &ori_outputs);
  auto lstm_op = graph->NewCNode(lstm_input_grad_inputs);
  MS_EXCEPTION_IF_NULL(lstm_op);
  auto ori_type = AnfAlgo::GetOutputInferDataType(dynamic_rnn_grad_inputs[8], 0);
  auto types = {AnfAlgo::GetOutputInferDataType(ori_outputs[2], 0), AnfAlgo::GetOutputInferDataType(ori_outputs[3], 0),
                AnfAlgo::GetOutputInferDataType(ori_outputs[4], 0), ori_type};
  std::vector<size_t> ori_shape = {AnfAlgo::GetOutputInferShape(dynamic_rnn_grad_inputs[8], 0)[0],
                                   AnfAlgo::GetOutputInferShape(dynamic_rnn_grad_inputs[8], 0)[1],
                                   4 * AnfAlgo::GetOutputInferShape(dynamic_rnn_grad_inputs[8], 0)[2]};
  auto shapes = {AnfAlgo::GetOutputInferShape(ori_outputs[2], 0), AnfAlgo::GetOutputInferShape(ori_outputs[3], 0),
                 AnfAlgo::GetOutputInferShape(ori_outputs[4], 0), ori_shape};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, lstm_op.get());
  return lstm_op;
}

AnfNodePtr CreateBatchMatMul(const FuncGraphPtr &graph, const AnfNodePtr &node1, const AnfNodePtr &node2) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  // BatchMatMul
  std::vector<AnfNodePtr> matmul_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimBatchMatMul->name())),
                                           node2, node1};
  auto batch_matmul = graph->NewCNode(matmul_inputs);
  MS_EXCEPTION_IF_NULL(batch_matmul);
  auto types = {AnfAlgo::GetOutputInferDataType(node1, 0)};
  std::vector<size_t> shape = {AnfAlgo::GetOutputInferShape(node2, 0)[0], AnfAlgo::GetOutputInferShape(node2, 0)[2],
                               AnfAlgo::GetOutputInferShape(node1, 0)[2]};
  auto shapes = {shape};
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), batch_matmul);
  AnfAlgo::SetNodeAttr("transpose_x1", MakeValue(true), batch_matmul);
  AnfAlgo::SetNodeAttr("transpose_x2", MakeValue(false), batch_matmul);
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, batch_matmul.get());
  return batch_matmul;
}

AnfNodePtr AddHConcatD(const FuncGraphPtr &graph, const AnfNodePtr &node1, const AnfNodePtr &node2) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  std::vector<AnfNodePtr> ori_outputs;
  CreateMultipleOutputsOfAnfNode(graph, node2, 2, &ori_outputs);
  auto ori_shape = AnfAlgo::GetOutputInferShape(node1, 0);
  std::vector<std::vector<size_t>> shape_tmp;
  if (ori_shape.size() == 3) {
    shape_tmp = {ori_shape};
  } else {
    shape_tmp = {{IntToSize(1), ori_shape[0], ori_shape[1]}};
  }
  auto ori_dtype = {AnfAlgo::GetOutputInferDataType(node1, 0)};
  // reshape
  std::vector<AnfNodePtr> reshape_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReshape->name())),
                                           node1};
  auto reshape = graph->NewCNode(reshape_input);
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), reshape);
  AnfAlgo::SetOutputInferTypeAndShape(ori_dtype, shape_tmp, reshape.get());

  // concatd --> concat
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name())),
                                           reshape, ori_outputs[0]};
  auto concat_op = graph->NewCNode(concat_inputs);
  MS_EXCEPTION_IF_NULL(concat_op);
  std::vector<size_t> input = {AnfAlgo::GetOutputInferShape(node2, 0)[0] + 1, AnfAlgo::GetOutputInferShape(node2, 0)[1],
                               AnfAlgo::GetOutputInferShape(node2, 0)[2]};
  auto types = {AnfAlgo::GetOutputInferDataType(node1, 0)};
  auto shapes = {input};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, concat_op.get());
  AnfAlgo::SetNodeAttr(kAttrN, MakeValue(2), concat_op);
  AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(std::vector<int>{2}), concat_op);
  AnfAlgo::SetNodeAttr("axis", MakeValue(0), concat_op);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), concat_op);
  return concat_op;
}

AnfNodePtr AddConcatD(const FuncGraphPtr &graph, const AnfNodePtr &node1, const AnfNodePtr &node2) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  // concatd --> concat
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name())), node1,
                                           node2};
  auto concat_op = graph->NewCNode(concat_inputs);
  MS_EXCEPTION_IF_NULL(concat_op);
  std::vector<size_t> input = {AnfAlgo::GetOutputInferShape(node1, 0)[0], AnfAlgo::GetOutputInferShape(node1, 0)[1],
                               AnfAlgo::GetOutputInferShape(node1, 0)[2] + AnfAlgo::GetOutputInferShape(node2, 0)[2]};
  auto types = {AnfAlgo::GetOutputInferDataType(node1, 0)};
  auto shapes = {input};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, concat_op.get());
  AnfAlgo::SetNodeAttr(kAttrN, MakeValue(2), concat_op);
  AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(std::vector<int>{2}), concat_op);
  AnfAlgo::SetNodeAttr("axis", MakeValue(2), concat_op);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), concat_op);
  return concat_op;
}

AnfNodePtr AddDwReduceSum(const FuncGraphPtr &graph, const AnfNodePtr &node1, const AnfNodePtr &node2) {
  // node1 : dynamic output
  // node2 : matmul
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  std::vector<AnfNodePtr> ori_outputs;
  CreateMultipleOutputsOfAnfNode(graph, node1, 5, &ori_outputs);
  // ReduceSumd
  std::vector<AnfNodePtr> reducesum_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceSum->name())),
                                              node2};
  auto reduce_sumd = graph->NewCNode(reducesum_inputs);
  MS_EXCEPTION_IF_NULL(reduce_sumd);
  auto types = {AnfAlgo::GetOutputInferDataType(ori_outputs[0], 0)};
  auto shapes = {AnfAlgo::GetOutputInferShape(ori_outputs[0], 0)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, reduce_sumd.get());
  AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(std::vector<int>{0}), reduce_sumd);
  AnfAlgo::SetNodeAttr("keep_dims", MakeValue(false), reduce_sumd);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), reduce_sumd);
  return reduce_sumd;
}

AnfNodePtr AddDbReduceSum(const FuncGraphPtr &graph, const AnfNodePtr &node1, const AnfNodePtr &node2) {
  // node1 lstm output
  // node2 // dynamic output
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node1);
  MS_EXCEPTION_IF_NULL(node2);
  std::vector<AnfNodePtr> ori_outputs;
  CreateMultipleOutputsOfAnfNode(graph, node2, 5, &ori_outputs);
  // ReduceSumd --> ReduceSum
  std::vector<AnfNodePtr> reducerum_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceSum->name())),
                                              node1};
  auto reduce_sumd = graph->NewCNode(reducerum_inputs);
  MS_EXCEPTION_IF_NULL(reduce_sumd);
  auto types = {AnfAlgo::GetOutputInferDataType(ori_outputs[1], 0)};
  auto shapes = {AnfAlgo::GetOutputInferShape(ori_outputs[1], 0)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, reduce_sumd.get());
  AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(std::vector<int>{0, 1}), reduce_sumd);
  AnfAlgo::SetNodeAttr("keep_dims", MakeValue(false), reduce_sumd);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), reduce_sumd);
  return reduce_sumd;
}

const AnfNodePtr DynamicRNNGradFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() < kDynamicRNNGradInputNum + 1) {
    MS_LOG(INFO) << "The input num of DynamicRNNGrad less than" << kDynamicRNNGradInputNum
                 << ". The node should not be changed";
    return nullptr;
  }
  // input_list of dynamic_rnn_grad
  const auto &ori_inputs = cnode->inputs();
  // create split_vd
  auto split_vd = CreateSplitVD(func_graph, ori_inputs[7]);
  // create concat_1
  auto h_concat = AddHConcatD(func_graph, ori_inputs[5], split_vd);
  // create concat_2
  auto concat = AddConcatD(func_graph, ori_inputs[1], h_concat);
  // create lsym_input_grad
  auto lstm_input_grad = CreateLSTMInputGrad(func_graph, cnode);
  std::vector<AnfNodePtr> lstm_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, lstm_input_grad, kLSTMInputGradOutputNum, &lstm_outputs);
  // create matmul
  auto batch_matmul = CreateBatchMatMul(func_graph, lstm_outputs[3], concat);
  // create reduce_sum_1
  auto dw_reduce_sum = AddDwReduceSum(func_graph, node, batch_matmul);
  // create reduce_sum_2
  auto db_reduce_sum = AddDbReduceSum(func_graph, lstm_outputs[3], node);
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple),
                                               dw_reduce_sum,
                                               db_reduce_sum,
                                               lstm_outputs[0],
                                               lstm_outputs[1],
                                               lstm_outputs[2]};
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  return make_tuple;
}
}  // namespace opt
}  // namespace mindspore
