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
#include "backend/optimizer/ascend/ir_fission/dynamic_rnn_grad_fission_v2.h"
#include <vector>
#include <string>
#include <memory>
#include "backend/session/kernel_graph.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/ascend/ascend_helper.h"
#include "utils/trace_base.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kDynamicRNNGradInputNum = 16;
constexpr size_t kSplitVOutputNum = 2;
constexpr size_t kBasicCellOutputNum = 2;
constexpr size_t kBasicLstmCStateGradOutput0DimNum = 3;
constexpr int64_t kAttrNValue = 2;
constexpr int64_t kAttrDynInputSizesValue = 2;
constexpr int64_t kAttrAxis2Value = 2;
constexpr int64_t kAttrNumSplitValue = 2;
constexpr int64_t kAttrSplitDimValue = 2;
constexpr size_t kDimMultiNum = 4;

void SetAttrInputAndHiddenSize(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_rnn_grad_cnode,
                               int64_t input_size, int64_t hidden_size) {
  auto input = dynamic_rnn_grad_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(input);
  // set for input
  while (input->isa<CNode>()) {
    AnfAlgo::SetNodeAttr(kAttrInputSize, MakeValue(input_size), input);
    AnfAlgo::SetNodeAttr(kAttrHiddenSize, MakeValue(hidden_size), input);
    auto input_cnode = input->cast<CNodePtr>();
    input = input_cnode->input(kIndex1);
  }
  if (input->isa<Parameter>()) {
    auto param = input->cast<ParameterPtr>();
    param->set_input_size(input_size);
    param->set_hidden_size(hidden_size);
  }

  // set for output
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto getitem_index : manager->node_users()[dynamic_rnn_grad_cnode]) {
    if (AnfAlgo::CheckPrimitiveType(getitem_index.first, prim::kPrimTupleGetItem)) {
      for (auto node_index : manager->node_users()[getitem_index.first]) {
        AnfAlgo::SetNodeAttr(kAttrInputSize, MakeValue(input_size), node_index.first);
        AnfAlgo::SetNodeAttr(kAttrHiddenSize, MakeValue(hidden_size), node_index.first);
      }
    }
  }
}
}  // namespace

void DynamicRnnGradFissionV2::CreateTLoopNode(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_rnn_grad_cnode,
                                              const RNNShapeSpecs &specs,
                                              std::vector<std::vector<AnfNodePtr>> *result_nodes) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dynamic_rnn_grad_cnode);
  MS_EXCEPTION_IF_NULL(result_nodes);
  std::vector<AnfNodePtr> basic_lstm_cell_c_state_grad_nodes;
  std::vector<AnfNodePtr> matmul_nodes;
  std::vector<AnfNodePtr> split_nodes;
  // Get the size of t
  auto input_i_shape = AnfAlgo::GetOutputInferShape(dynamic_rnn_grad_cnode->input(kIndex12), 0);

  for (size_t i = 0; i < specs.t_size; ++i) {
    // Create basic_lstm_cell_c_state_grad
    std::vector<AnfNodePtr> basic_lstm_cell_c_state_grad_inputs = {
      NewValueNode(std::make_shared<Primitive>(kBasicLSTMCellCStateGradV2OpName))};
    auto basic_lstm_cell_c_state_grad = NewCNode(basic_lstm_cell_c_state_grad_inputs, func_graph);

    std::vector<size_t> output0_dims{specs.batch_size, kDimMultiNum * specs.hidden_nz_size * kCubeSize};
    // batch_size, hidden_size
    std::vector<size_t> output1_dims{input_i_shape[kDim1], input_i_shape[kDim2]};
    AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat16, kNumberTypeFloat32}, {output0_dims, output1_dims},
                                        basic_lstm_cell_c_state_grad.get());
    AnfAlgo::SetNodeAttr("forget_bias", MakeValue(1.0f), basic_lstm_cell_c_state_grad);
    AnfAlgo::SetNodeAttr("activation", MakeValue("Tanh"), basic_lstm_cell_c_state_grad);

    // Create matmul
    std::vector<AnfNodePtr> matmul_inputs;
    if (specs.shape_need_align) {
      matmul_inputs.push_back(NewValueNode(std::make_shared<Primitive>(prim::kPrimBatchMatMulV2->name())));
    } else {
      matmul_inputs.push_back(NewValueNode(std::make_shared<Primitive>(prim::kPrimMatMul->name())));
    }
    auto matmul = NewCNode(matmul_inputs, func_graph);
    AnfAlgo::SetOutputInferTypeAndShape(
      {kNumberTypeFloat32}, {{IntToSize(1), specs.batch_size, specs.input_size + specs.hidden_size}}, matmul.get());
    AnfAlgo::SetNodeAttr("transpose_x1", MakeValue(false), matmul);
    AnfAlgo::SetNodeAttr("transpose_x2", MakeValue(true), matmul);
    if (specs.shape_need_align) {
      AnfAlgo::SetNodeAttr(kAttrInputSize, MakeValue(SizeToLong(specs.input_size)), matmul);
      AnfAlgo::SetNodeAttr(kAttrHiddenSize, MakeValue(SizeToLong(specs.hidden_size)), matmul);
      std::vector<size_t> output_shape = {1, specs.input_nz_size + specs.hidden_nz_size, specs.batch_nz_size, kCubeSize,
                                          kCubeSize};
      AnfAlgo::SetNodeAttr(kAttrFixedOutputDeviceShape,
                           MakeValue(std::vector<std::vector<int64_t>>{Convert2Long(output_shape)}), matmul);
    }

    // Create split
    std::vector<AnfNodePtr> splitv_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitV->name()))};
    auto split_v = NewCNode(splitv_input, func_graph);
    std::vector<size_t> split_v_output0_shape{IntToSize(1), specs.batch_size, specs.input_size};
    std::vector<size_t> split_v_output1_shape{IntToSize(1), specs.batch_size, specs.hidden_size};
    AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32, kNumberTypeFloat32},
                                        {split_v_output0_shape, split_v_output1_shape}, split_v.get());
    AnfAlgo::SetNodeAttr(kAttrSizeSplits,
                         MakeValue(std::vector<int64_t>{SizeToLong(specs.input_nz_size * kCubeSize),
                                                        SizeToLong(specs.hidden_nz_size * kCubeSize)}),
                         split_v);
    AnfAlgo::SetNodeAttr(kAttrSplitDim, MakeValue(static_cast<int64_t>(kAttrSplitDimValue)), split_v);
    AnfAlgo::SetNodeAttr(kAttrNumSplit, MakeValue(static_cast<int64_t>(kAttrNumSplitValue)), split_v);
    if (specs.shape_need_align) {
      AnfAlgo::SetNodeAttr(kAttrFixedInputFormat, MakeValue(std::vector<string>{kOpFormat_FRAC_NZ}), split_v);
      AnfAlgo::SetNodeAttr(kAttrFixedOutputFormat, MakeValue(std::vector<string>{kOpFormat_FRAC_NZ, kOpFormat_FRAC_NZ}),
                           split_v);
      std::vector<size_t> input_shape = {1, specs.input_nz_size + specs.hidden_nz_size, specs.batch_nz_size, kCubeSize,
                                         kCubeSize};
      AnfAlgo::SetNodeAttr(kAttrFixedInputDeviceShape,
                           MakeValue(std::vector<std::vector<int64_t>>{Convert2Long(input_shape)}), split_v);
    }

    basic_lstm_cell_c_state_grad_nodes.emplace_back(basic_lstm_cell_c_state_grad);
    matmul_nodes.emplace_back(matmul);
    split_nodes.emplace_back(split_v);
  }
  result_nodes->emplace_back(basic_lstm_cell_c_state_grad_nodes);
  result_nodes->emplace_back(matmul_nodes);
  result_nodes->emplace_back(split_nodes);
}

AnfNodePtr DynamicRnnGradFissionV2::CreateLSTMSPlitV(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                                     const std::vector<std::vector<size_t>> &split_shapes,
                                                     const std::vector<TypeId> &split_types,
                                                     const std::vector<int64_t> &size_split, size_t num_split_x) const {
  std::vector<AnfNodePtr> lstm_split_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitV->name())),
                                              input};
  auto lstm_split = NewCNode(lstm_split_input, func_graph);
  AnfAlgo::SetOutputInferTypeAndShape(split_types, split_shapes, lstm_split.get());
  AnfAlgo::SetNodeAttr(kAttrSizeSplits, MakeValue(size_split), lstm_split);
  AnfAlgo::SetNodeAttr(kAttrSplitDim, MakeValue(static_cast<int64_t>(0)), lstm_split);
  AnfAlgo::SetNodeAttr(kAttrNumSplit, MakeValue(SizeToLong(num_split_x)), lstm_split);
  return lstm_split;
}

void DynamicRnnGradFissionV2::CreateTLoopNodeWithEdge(const FuncGraphPtr &func_graph,
                                                      const CNodePtr &dynamic_rnn_grad_cnode,
                                                      const std::vector<std::vector<AnfNodePtr>> &result_nodes,
                                                      size_t num_split_x, const RNNShapeSpecs &specs,
                                                      std::vector<std::vector<AnfNodePtr>> *loop_node_outputs) const {
  auto &basic_lstm_cell_c_state_grad_nodes = result_nodes[kIndex0];
  auto &matmul_nodes = result_nodes[kIndex1];
  auto &split_nodes = result_nodes[kIndex2];
  auto &lstm_split_c_outputs = result_nodes[kIndex3];
  auto &lstm_split_dy_outputs = result_nodes[kIndex4];
  auto &lstm_split_i_outputs = result_nodes[kIndex5];
  auto &lstm_split_j_outputs = result_nodes[kIndex6];
  auto &lstm_split_f_outputs = result_nodes[kIndex7];
  auto &lstm_split_o_outputs = result_nodes[kIndex8];
  auto &lstm_split_tanh_outputs = result_nodes[kIndex9];
  std::vector<AnfNodePtr> pre_basic_lstm_cell_c_state_grad_outputs;
  std::vector<AnfNodePtr> pre_split_outputs;
  std::vector<AnfNodePtr> lstm_x_concat_input(num_split_x + 1);
  lstm_x_concat_input[0] = NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name()));
  std::vector<AnfNodePtr> lstm_gage_concat_input(num_split_x + 1);
  lstm_gage_concat_input[0] = NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name()));
  for (size_t i = 0; i < num_split_x; ++i) {
    size_t idx = num_split_x - i - 1;
    // Create basic_lstm_cell_c_state_grad
    std::vector<AnfNodePtr> basic_lstm_cell_c_state_grad_inputs = {
      NewValueNode(std::make_shared<Primitive>(kBasicLSTMCellCStateGradV2OpName))};
    if (i == num_split_x - 1) {
      std::vector<AnfNodePtr> reshape_inputs = {NewValueNode(std::make_shared<Primitive>(kReshapeOpName)),
                                                dynamic_rnn_grad_cnode->input(kIndex6)};
      auto reshape = NewCNode(reshape_inputs, func_graph);
      auto reshape_out_shape = {IntToSize(1),
                                AnfAlgo::GetOutputInferShape(dynamic_rnn_grad_cnode->input(kIndex6), 0)[0],
                                AnfAlgo::GetOutputInferShape(dynamic_rnn_grad_cnode->input(kIndex6), 0)[1]};
      AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {reshape_out_shape}, reshape.get());
      (void)basic_lstm_cell_c_state_grad_inputs.emplace_back(reshape);
    } else {
      (void)basic_lstm_cell_c_state_grad_inputs.emplace_back(lstm_split_c_outputs[idx - 1]);
    }
    (void)basic_lstm_cell_c_state_grad_inputs.emplace_back(lstm_split_dy_outputs[idx]);
    if (i == 0) {
      (void)basic_lstm_cell_c_state_grad_inputs.emplace_back(dynamic_rnn_grad_cnode->input(kIndex10));
      (void)basic_lstm_cell_c_state_grad_inputs.emplace_back(dynamic_rnn_grad_cnode->input(kIndex11));
    } else {
      (void)basic_lstm_cell_c_state_grad_inputs.emplace_back(pre_split_outputs[1]);
      (void)basic_lstm_cell_c_state_grad_inputs.emplace_back(pre_basic_lstm_cell_c_state_grad_outputs[1]);
    }
    (void)basic_lstm_cell_c_state_grad_inputs.emplace_back(lstm_split_i_outputs[idx]);
    (void)basic_lstm_cell_c_state_grad_inputs.emplace_back(lstm_split_j_outputs[idx]);
    (void)basic_lstm_cell_c_state_grad_inputs.emplace_back(lstm_split_f_outputs[idx]);
    (void)basic_lstm_cell_c_state_grad_inputs.emplace_back(lstm_split_o_outputs[idx]);
    (void)basic_lstm_cell_c_state_grad_inputs.emplace_back(lstm_split_tanh_outputs[idx]);
    auto basic_lstm_cell_c_state_grad = NewCNode(basic_lstm_cell_c_state_grad_inputs, func_graph);
    basic_lstm_cell_c_state_grad->set_abstract(basic_lstm_cell_c_state_grad_nodes[i]->abstract());
    AnfAlgo::CopyNodeAttrs(basic_lstm_cell_c_state_grad_nodes[i], basic_lstm_cell_c_state_grad);
    // Create outputs for current basic_lstm_cell_c_state_grad node
    std::vector<AnfNodePtr> basic_lstm_cell_c_state_grad_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, basic_lstm_cell_c_state_grad, kBasicCellOutputNum,
                                   &basic_lstm_cell_c_state_grad_outputs);
    pre_basic_lstm_cell_c_state_grad_outputs = basic_lstm_cell_c_state_grad_outputs;

    // Create MatMul
    auto matmul_type = specs.shape_need_align ? prim::kPrimBatchMatMulV2->name() : prim::kPrimMatMul->name();
    std::vector<AnfNodePtr> matmul_inputs = {NewValueNode(std::make_shared<Primitive>(matmul_type)),
                                             basic_lstm_cell_c_state_grad_outputs[0],
                                             dynamic_rnn_grad_cnode->input(kIndex2)};
    auto matmul = NewCNode(matmul_inputs, func_graph);
    matmul->set_abstract(matmul_nodes[i]->abstract());
    AnfAlgo::CopyNodeAttrs(matmul_nodes[i], matmul);

    // Create splitv
    std::vector<AnfNodePtr> splitv_input = {NewValueNode(std::make_shared<Primitive>(kSplitVOpName)), matmul};
    auto split_v = NewCNode(splitv_input, func_graph);
    split_v->set_abstract(split_nodes[i]->abstract());
    AnfAlgo::CopyNodeAttrs(split_nodes[i], split_v);

    // Create outputs for current split node
    std::vector<AnfNodePtr> split_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, split_v, kSplitVOutputNum, &split_outputs);
    pre_split_outputs = split_outputs;

    lstm_x_concat_input[idx + 1] = split_outputs[0];

    if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
      lstm_gage_concat_input[idx + 1] = basic_lstm_cell_c_state_grad_outputs[0];
    } else {
      auto basic_lstm_cell_output_0_shape = AnfAlgo::GetOutputInferShape(basic_lstm_cell_c_state_grad_outputs[0], 0);
      std::vector<size_t> temp_shape;
      if (basic_lstm_cell_output_0_shape.size() == kBasicLstmCStateGradOutput0DimNum) {
        temp_shape = basic_lstm_cell_output_0_shape;
      } else {
        temp_shape = {1, basic_lstm_cell_output_0_shape[0], basic_lstm_cell_output_0_shape[1]};
      }
      std::vector<AnfNodePtr> reshape_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReshape->name())),
                                               basic_lstm_cell_c_state_grad_outputs[0]};
      auto reshape = NewCNode(reshape_input, func_graph);
      AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(basic_lstm_cell_c_state_grad_outputs[0], 0)},
                                          {temp_shape}, reshape.get());
      lstm_gage_concat_input[idx + 1] = reshape;
    }
  }
  loop_node_outputs->push_back(pre_basic_lstm_cell_c_state_grad_outputs);
  loop_node_outputs->push_back(pre_split_outputs);
  loop_node_outputs->push_back(lstm_x_concat_input);
  loop_node_outputs->push_back(lstm_gage_concat_input);
}

AnfNodePtr DynamicRnnGradFissionV2::AddLSTMInputGradNode(const FuncGraphPtr &func_graph,
                                                         const CNodePtr &dynamic_rnn_grad_cnode,
                                                         const RNNShapeSpecs &specs,
                                                         std::vector<AnfNodePtr> *outputs) const {
  std::vector<std::vector<AnfNodePtr>> result_nodes;
  CreateTLoopNode(func_graph, dynamic_rnn_grad_cnode, specs, &result_nodes);

  auto origin_input5_shape = AnfAlgo::GetOutputInferShape(dynamic_rnn_grad_cnode->input(kIndex6), 0);
  std::vector<size_t> split_c_dims{IntToSize(1), origin_input5_shape[0], origin_input5_shape[1]};

  auto origin_input7 = dynamic_rnn_grad_cnode->input(kIndex8);
  size_t num_split_x = AnfAlgo::GetOutputInferShape(origin_input7, 0)[0];
  std::vector<std::vector<size_t>> split_shapes;
  std::vector<TypeId> split_types;
  std::vector<int64_t> size_split;
  for (size_t i = 0; i < num_split_x; ++i) {
    split_shapes.emplace_back(split_c_dims);
    split_types.emplace_back(kNumberTypeFloat32);
    size_split.emplace_back(1);
  }
  // Create lstm_split_c
  auto lstm_split_c = CreateLSTMSPlitV(func_graph, origin_input7, split_shapes, split_types, size_split, num_split_x);
  std::vector<AnfNodePtr> lstm_split_c_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, lstm_split_c, num_split_x, &lstm_split_c_outputs);
  result_nodes.push_back(lstm_split_c_outputs);

  // Create lstm_split_dy
  auto lstm_split_dy = CreateLSTMSPlitV(func_graph, dynamic_rnn_grad_cnode->input(kIndex9), split_shapes, split_types,
                                        size_split, num_split_x);
  std::vector<AnfNodePtr> lstm_split_dy_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, lstm_split_dy, num_split_x, &lstm_split_dy_outputs);
  result_nodes.push_back(lstm_split_dy_outputs);

  if (specs.t_size != 1) {
    // Create lstm_split_i
    auto lstm_split_i = CreateLSTMSPlitV(func_graph, dynamic_rnn_grad_cnode->input(kIndex12), split_shapes, split_types,
                                         size_split, num_split_x);
    std::vector<AnfNodePtr> lstm_split_i_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, lstm_split_i, num_split_x, &lstm_split_i_outputs);
    result_nodes.push_back(lstm_split_i_outputs);

    // Create lstm_split_j
    auto lstm_split_j = CreateLSTMSPlitV(func_graph, dynamic_rnn_grad_cnode->input(kIndex13), split_shapes, split_types,
                                         size_split, num_split_x);
    std::vector<AnfNodePtr> lstm_split_j_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, lstm_split_j, num_split_x, &lstm_split_j_outputs);
    result_nodes.push_back(lstm_split_j_outputs);

    // Create lstm_split_f
    auto lstm_split_f = CreateLSTMSPlitV(func_graph, dynamic_rnn_grad_cnode->input(kIndex14), split_shapes, split_types,
                                         size_split, num_split_x);
    std::vector<AnfNodePtr> lstm_split_f_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, lstm_split_f, num_split_x, &lstm_split_f_outputs);
    result_nodes.push_back(lstm_split_f_outputs);

    // Create lstm_split_o
    auto lstm_split_o = CreateLSTMSPlitV(func_graph, dynamic_rnn_grad_cnode->input(kIndex15), split_shapes, split_types,
                                         size_split, num_split_x);
    std::vector<AnfNodePtr> lstm_split_o_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, lstm_split_o, num_split_x, &lstm_split_o_outputs);
    result_nodes.push_back(lstm_split_o_outputs);

    // Create lstm_split_tanh
    auto lstm_split_tanh = CreateLSTMSPlitV(func_graph, dynamic_rnn_grad_cnode->input(kIndex16), split_shapes,
                                            split_types, size_split, num_split_x);
    std::vector<AnfNodePtr> lstm_split_tanh_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, lstm_split_tanh, num_split_x, &lstm_split_tanh_outputs);
    result_nodes.push_back(lstm_split_tanh_outputs);
  } else {
    result_nodes.push_back(std::vector<AnfNodePtr>{dynamic_rnn_grad_cnode->input(kIndex12)});
    result_nodes.push_back(std::vector<AnfNodePtr>{dynamic_rnn_grad_cnode->input(kIndex13)});
    result_nodes.push_back(std::vector<AnfNodePtr>{dynamic_rnn_grad_cnode->input(kIndex14)});
    result_nodes.push_back(std::vector<AnfNodePtr>{dynamic_rnn_grad_cnode->input(kIndex15)});
    result_nodes.push_back(std::vector<AnfNodePtr>{dynamic_rnn_grad_cnode->input(kIndex16)});
  }

  // Add edges
  std::vector<std::vector<AnfNodePtr>> loop_node_outputs;
  CreateTLoopNodeWithEdge(func_graph, dynamic_rnn_grad_cnode, result_nodes, num_split_x, specs, &loop_node_outputs);
  auto &pre_basic_lstm_cell_c_state_grad_outputs = loop_node_outputs[kIndex0];
  auto &pre_split_outputs = loop_node_outputs[kIndex1];
  auto &lstm_x_concat_input = loop_node_outputs[kIndex2];
  auto &lstm_gage_concat_input = loop_node_outputs[kIndex3];

  if (specs.t_size != 1) {
    // Create lstm_x_concat
    auto lstm_x_concat = NewCNode(lstm_x_concat_input, func_graph);
    AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {AnfAlgo::GetOutputInferShape(dynamic_rnn_grad_cnode, 2)},
                                        lstm_x_concat.get());
    AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(num_split_x)), lstm_x_concat);
    AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(std::vector<int64_t>{SizeToLong(num_split_x)}), lstm_x_concat);
    AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(static_cast<int64_t>(0)), lstm_x_concat);

    // Create lstm_gage_concat
    auto lstm_gage_concat = NewCNode(lstm_gage_concat_input, func_graph);
    std::vector<size_t> gage_concat_shape;
    if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
      gage_concat_shape = {specs.t_size * specs.batch_size, kDimMultiNum * specs.hidden_nz_size * kCubeSize};
    } else {
      gage_concat_shape = {specs.t_size, specs.batch_size, kDimMultiNum * specs.hidden_nz_size * kCubeSize};
    }
    AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat16}, {gage_concat_shape}, lstm_gage_concat.get());
    AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(num_split_x)), lstm_gage_concat);
    AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(std::vector<int64_t>{SizeToLong(num_split_x)}),
                         lstm_gage_concat);
    AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(SizeToLong(0)), lstm_gage_concat);

    outputs->emplace_back(lstm_x_concat);
    outputs->emplace_back(pre_split_outputs[1]);
    outputs->emplace_back(pre_basic_lstm_cell_c_state_grad_outputs[1]);
    return lstm_gage_concat;
  } else {
    outputs->emplace_back(lstm_x_concat_input[1]);
    outputs->emplace_back(pre_split_outputs[1]);
    outputs->emplace_back(pre_basic_lstm_cell_c_state_grad_outputs[1]);
    return lstm_gage_concat_input[1];
  }
}

AnfNodePtr DynamicRnnGradFissionV2::CreateSplitV(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_rnn_grad_cnode,
                                                 const RNNShapeSpecs &specs) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dynamic_rnn_grad_cnode);
  // Create node
  auto origin_input6 = dynamic_rnn_grad_cnode->input(kIndex7);
  auto origin_input6_dtype = AnfAlgo::GetOutputInferDataType(origin_input6, 0);
  auto origin_input6_shape = AnfAlgo::GetOutputInferShape(origin_input6, 0);
  std::vector<AnfNodePtr> splitv_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSplitV->name()))};
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    std::vector<AnfNodePtr> reshape_input = {NewValueNode(std::make_shared<Primitive>(kReshapeOpName)), origin_input6};
    auto reshape = NewCNode(reshape_input, func_graph);
    std::vector<size_t> shape = {origin_input6_shape[kDim0] * origin_input6_shape[kDim1], origin_input6_shape[kDim2]};
    AnfAlgo::SetOutputInferTypeAndShape({origin_input6_dtype}, {shape}, reshape.get());
    splitv_input.push_back(reshape);
  } else {
    splitv_input.push_back(origin_input6);
  }
  auto split_v = NewCNode(splitv_input, func_graph);
  // Set infer data type and shape
  std::vector<size_t> shape1, shape2;
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    shape1 = {(origin_input6_shape[kDim0] - 1) * origin_input6_shape[kDim1], origin_input6_shape[kDim2]};
    shape2 = {origin_input6_shape[kDim1], origin_input6_shape[kDim2]};
  } else {
    shape1 = {origin_input6_shape[kDim0] - 1, origin_input6_shape[kDim1], origin_input6_shape[kDim2]};
    shape2 = {1, origin_input6_shape[kDim1], origin_input6_shape[kDim2]};
  }
  auto dtypes = {origin_input6_dtype, origin_input6_dtype};
  std::vector<std::vector<size_t>> shapes = {shape1, shape2};
  AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split_v.get());
  // Set attr
  AnfAlgo::SetNodeAttr(kAttrSplitDim, MakeValue(SizeToLong(0)), split_v);
  AnfAlgo::SetNodeAttr(kAttrNumSplit, MakeValue(SizeToLong(kAttrNumSplitValue)), split_v);
  std::vector<int64_t> size_splits;
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    size_splits = {SizeToLong((origin_input6_shape[kDim0] - 1) * origin_input6_shape[kDim1]),
                   SizeToLong(origin_input6_shape[kDim1])};
  } else {
    size_splits = {SizeToLong(origin_input6_shape[kDim0] - 1), 1};
  }
  AnfAlgo::SetNodeAttr(kAttrSizeSplits, MakeValue(size_splits), split_v);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), split_v);
  return split_v;
}

AnfNodePtr DynamicRnnGradFissionV2::CreateHConcat(const FuncGraphPtr &func_graph,
                                                  const CNodePtr &dynamic_rnn_grad_cnode, const AnfNodePtr &splitv,
                                                  const RNNShapeSpecs &specs) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dynamic_rnn_grad_cnode);
  MS_EXCEPTION_IF_NULL(splitv);
  // Create node
  std::vector<AnfNodePtr> splitv_outputs;
  CreateMultipleOutputsOfAnfNode(func_graph, splitv, kSplitVOutputNum, &splitv_outputs);
  if (splitv_outputs.size() != kSplitVOutputNum) {
    MS_LOG(EXCEPTION) << "Create outputs of node " << splitv->DebugString() << " failed"
                      << trace::DumpSourceLines(dynamic_rnn_grad_cnode);
  }
  auto origin_input4 = dynamic_rnn_grad_cnode->input(kIndex5);
  auto origin_input4_shape = AnfAlgo::GetOutputInferShape(origin_input4, 0);
  // Create reshape to change shape
  std::vector<size_t> shape_tmp;
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    shape_tmp = {origin_input4_shape[0], origin_input4_shape[1]};
  } else {
    if (origin_input4_shape.size() == kShape3dDims) {
      shape_tmp = origin_input4_shape;
    } else {
      // 1, batch_size, hidden_size
      shape_tmp = {1, origin_input4_shape[0], origin_input4_shape[1]};
    }
  }
  std::vector<AnfNodePtr> reshape_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReshape->name())),
                                           origin_input4};
  auto reshape = NewCNode(reshape_input, func_graph);
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(origin_input4, 0)}, {shape_tmp}, reshape.get());
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name())),
                                           reshape, splitv_outputs[0]};
  auto concat = NewCNode(concat_inputs, func_graph);
  // Set infer data type and shape
  auto splitv_output0_shape = AnfAlgo::GetOutputInferShape(splitv, 0);
  std::vector<size_t> shape;
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    shape = {splitv_output0_shape[0] + origin_input4_shape[0], origin_input4_shape[1]};
  } else {
    // t_size - 1 + 1, batch_size, hidden_size
    shape = {splitv_output0_shape[0] + 1, origin_input4_shape[0], origin_input4_shape[1]};
  }
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(origin_input4, 0)}, {shape}, concat.get());
  // Set attr
  AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(kAttrNValue)), concat);
  AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(std::vector<int64_t>{kAttrDynInputSizesValue}), concat);
  AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(SizeToLong(0)), concat);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), concat);
  return concat;
}

AnfNodePtr DynamicRnnGradFissionV2::CreateConcat(const FuncGraphPtr &func_graph, const CNodePtr &dynamic_rnn_grad_cnode,
                                                 const AnfNodePtr &h_concat, const RNNShapeSpecs &specs) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dynamic_rnn_grad_cnode);
  // Create node
  auto origin_input0 = dynamic_rnn_grad_cnode->input(1);
  auto origin_input0_dtype = AnfAlgo::GetOutputInferDataType(origin_input0, 0);
  auto origin_input0_shape = AnfAlgo::GetOutputInferShape(origin_input0, 0);
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name()))};
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    std::vector<AnfNodePtr> reshape_input = {NewValueNode(std::make_shared<Primitive>(kReshapeOpName)), origin_input0};
    auto reshape = NewCNode(reshape_input, func_graph);
    std::vector<size_t> shape = {origin_input0_shape[kDim0] * origin_input0_shape[kDim1], origin_input0_shape[kDim2]};
    AnfAlgo::SetOutputInferTypeAndShape({origin_input0_dtype}, {shape}, reshape.get());
    // t_size * batch_size, input_size
    concat_inputs.push_back(reshape);
  } else {
    // t_size, batch_size, input_size
    concat_inputs.push_back(origin_input0);
  }
  concat_inputs.push_back(h_concat);
  auto concat = NewCNode(concat_inputs, func_graph);
  // Set infer data type and shape
  auto h_concat_output_shape = AnfAlgo::GetOutputInferShape(h_concat, 0);
  std::vector<size_t> shape;
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    shape = {origin_input0_shape[kDim0] * origin_input0_shape[kDim1],
             origin_input0_shape[kDim2] + h_concat_output_shape[kDim1]};
  } else {
    // t_size, batch_size, input_size + hidden_size
    shape = {origin_input0_shape[kDim0], origin_input0_shape[kDim1],
             origin_input0_shape[kDim2] + h_concat_output_shape[kDim2]};
  }
  AnfAlgo::SetOutputInferTypeAndShape({origin_input0_dtype}, {shape}, concat.get());
  // Set attr
  AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(kAttrNValue)), concat);
  AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(std::vector<int64_t>{kAttrDynInputSizesValue}), concat);
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(1), concat);
  } else {
    AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(SizeToLong(kAttrAxis2Value)), concat);
  }
  if (specs.shape_need_align) {
    AnfAlgo::SetNodeAttr(kAttrFixedInputFormat, MakeValue(std::vector<string>{kOpFormat_FRAC_NZ, kOpFormat_FRAC_NZ}),
                         concat);
    AnfAlgo::SetNodeAttr(kAttrFixedOutputFormat, MakeValue(std::vector<string>{kOpFormat_FRAC_NZ}), concat);
    std::vector<size_t> out_shape = {specs.t_size, specs.input_nz_size + specs.hidden_nz_size, specs.batch_nz_size,
                                     kCubeSize, kCubeSize};
    AnfAlgo::SetNodeAttr(kAttrFixedOutputDeviceShape,
                         MakeValue(std::vector<std::vector<int64_t>>{Convert2Long(out_shape)}), concat);
  }
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), concat);
  return concat;
}

AnfNodePtr DynamicRnnGradFissionV2::CreateConcatNodeT1(const FuncGraphPtr &func_graph,
                                                       const CNodePtr &dynamic_rnn_grad_cnode,
                                                       const RNNShapeSpecs &specs) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dynamic_rnn_grad_cnode);
  // Create node
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name()))};
  auto origin_input0 = dynamic_rnn_grad_cnode->input(kIndex1);
  auto origin_input0_dtype = AnfAlgo::GetOutputInferDataType(origin_input0, 0);
  auto origin_input0_shape = AnfAlgo::GetOutputInferShape(origin_input0, 0);
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    std::vector<AnfNodePtr> reshape_inputs = {NewValueNode(std::make_shared<Primitive>(kReshapeOpName)), origin_input0};
    auto reshape_in0 = NewCNode(reshape_inputs, func_graph);
    std::vector<size_t> shape = {origin_input0_shape[kDim0] * origin_input0_shape[kDim1], origin_input0_shape[kDim2]};
    AnfAlgo::SetOutputInferTypeAndShape({origin_input0_dtype}, {shape}, reshape_in0.get());
    // t_size * batch_size, input_size (t_size = 1)
    concat_inputs.push_back(reshape_in0);
  } else {
    // t_size, batch_size, input_size (t_size = 1)
    concat_inputs.push_back(origin_input0);
  }

  auto origin_input4 = dynamic_rnn_grad_cnode->input(kIndex5);
  auto origin_input4_shape = AnfAlgo::GetOutputInferShape(origin_input4, 0);
  std::vector<size_t> shape_tmp;
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    shape_tmp = {origin_input4_shape[0], origin_input4_shape[1]};
  } else {
    if (origin_input4_shape.size() == kShape3dDims) {
      shape_tmp = origin_input4_shape;
    } else {
      // 1, batch_size, hidden_size (t_size = 1)
      shape_tmp = {1, origin_input4_shape[0], origin_input4_shape[1]};
    }
  }
  std::vector<AnfNodePtr> reshape_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReshape->name())),
                                           origin_input4};
  auto reshape_in4 = NewCNode(reshape_input, func_graph);
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(origin_input4, 0)}, {shape_tmp},
                                      reshape_in4.get());
  concat_inputs.push_back(reshape_in4);
  auto concat = NewCNode(concat_inputs, func_graph);
  // Set infer data type and shape
  std::vector<size_t> shape;
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    shape = {origin_input0_shape[kDim0] * origin_input0_shape[kDim1], origin_input0_shape[kDim2] + shape_tmp[kDim1]};
  } else {
    // t_size, batch_size, input_size + hidden_size (t_size = 1)
    shape = {origin_input0_shape[kDim0], origin_input0_shape[kDim1], origin_input0_shape[kDim2] + shape_tmp[kDim2]};
  }
  AnfAlgo::SetOutputInferTypeAndShape({origin_input0_dtype}, {shape}, concat.get());
  // Set attr
  AnfAlgo::SetNodeAttr(kAttrN, MakeValue(SizeToLong(kAttrNValue)), concat);
  AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(std::vector<int64_t>{kAttrDynInputSizesValue}), concat);
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(1), concat);
  } else {
    AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(SizeToLong(kAttrAxis2Value)), concat);
  }
  if (specs.shape_need_align) {
    AnfAlgo::SetNodeAttr(kAttrFixedInputFormat, MakeValue(std::vector<string>{kOpFormat_FRAC_NZ, kOpFormat_FRAC_NZ}),
                         concat);
    AnfAlgo::SetNodeAttr(kAttrFixedOutputFormat, MakeValue(std::vector<string>{kOpFormat_FRAC_NZ}), concat);
    std::vector<size_t> out_shape = {specs.t_size, specs.input_nz_size + specs.hidden_nz_size, specs.batch_nz_size,
                                     kCubeSize, kCubeSize};
    AnfAlgo::SetNodeAttr(kAttrFixedOutputDeviceShape,
                         MakeValue(std::vector<std::vector<int64_t>>{Convert2Long(out_shape)}), concat);
  }
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), concat);
  return concat;
}

AnfNodePtr DynamicRnnGradFissionV2::CreateMatMulNode(const FuncGraphPtr &func_graph, const AnfNodePtr &lstm_input_grad,
                                                     const AnfNodePtr &concat, const RNNShapeSpecs &specs) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  // Create node
  auto matmul_type = (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) ? prim::kPrimMatMulV2->name()
                                                                                    : prim::kPrimBatchMatMul->name();
  std::vector<AnfNodePtr> matmul_inputs = {NewValueNode(std::make_shared<Primitive>(matmul_type)), concat,
                                           lstm_input_grad};
  auto matmul = NewCNode(matmul_inputs, func_graph);
  // Set infer data type and shape
  auto concat_shape = AnfAlgo::GetOutputInferShape(concat, 0);
  auto lstm_input_grad_shape = AnfAlgo::GetOutputInferShape(lstm_input_grad, 0);
  std::vector<size_t> shape;
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    // t_size * (input_size + hidden_size), 4 * hidden_size
    shape = {concat_shape[kDim1], lstm_input_grad_shape[kDim1]};
  } else {
    // t_size, input_size + hidden_size, 4 * hidden_nz_size * 16
    shape = {concat_shape[kDim0], concat_shape[kDim2], lstm_input_grad_shape[kDim2]};
  }
  AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat16}, {shape}, matmul.get());
  // Set attr
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), matmul);
  AnfAlgo::SetNodeAttr("transpose_x1", MakeValue(true), matmul);
  AnfAlgo::SetNodeAttr("transpose_x2", MakeValue(false), matmul);
  if (specs.shape_need_align) {
    AnfAlgo::SetNodeAttr(kAttrFixedInputFormat, MakeValue(std::vector<string>{kOpFormat_FRAC_NZ, kOpFormat_FRAC_NZ}),
                         matmul);
    AnfAlgo::SetNodeAttr(kAttrFixedOutputFormat, MakeValue(std::vector<string>{kOpFormat_FRAC_NZ}), matmul);
    std::vector<size_t> in0_shape = {specs.t_size, specs.input_nz_size + specs.hidden_nz_size, specs.batch_nz_size,
                                     kCubeSize, kCubeSize};
    std::vector<size_t> in1_shape = {specs.t_size, kDimMultiNum * specs.hidden_nz_size, specs.batch_nz_size, kCubeSize,
                                     kCubeSize};
    std::vector<size_t> out_shape = {specs.t_size, kDimMultiNum * specs.hidden_nz_size,
                                     specs.input_nz_size + specs.hidden_nz_size, kCubeSize, kCubeSize};
    AnfAlgo::SetNodeAttr(kAttrFixedInputDeviceShape,
                         MakeValue(std::vector<std::vector<int64_t>>{Convert2Long(in0_shape), Convert2Long(in1_shape)}),
                         matmul);
    AnfAlgo::SetNodeAttr(kAttrFixedOutputDeviceShape,
                         MakeValue(std::vector<std::vector<int64_t>>{Convert2Long(out_shape)}), matmul);
  }
  return matmul;
}

AnfNodePtr DynamicRnnGradFissionV2::CreateMatMulNode2(const FuncGraphPtr &func_graph, const AnfNodePtr &lstm_input_grad,
                                                      const AnfNodePtr &node, const RNNShapeSpecs &specs) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  // Create node
  auto matmul_type = (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) ? prim::kPrimMatMulV2->name()
                                                                                    : prim::kPrimBatchMatMul->name();
  std::vector<AnfNodePtr> matmul_inputs = {NewValueNode(std::make_shared<Primitive>(matmul_type)), node,
                                           lstm_input_grad};
  auto matmul = NewCNode(matmul_inputs, func_graph);
  // Set infer data type and shape
  auto lstm_input_grad_shape = AnfAlgo::GetOutputInferShape(lstm_input_grad, 0);
  std::vector<size_t> out_shape;
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    out_shape = {IntToSize(1), lstm_input_grad_shape[kDim1]};
  } else {
    out_shape = {lstm_input_grad_shape[kDim0], IntToSize(1), lstm_input_grad_shape[kDim2]};
  }
  AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat16}, {out_shape}, matmul.get());
  // Set attr
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), matmul);
  AnfAlgo::SetNodeAttr("transpose_x1", MakeValue(false), matmul);
  AnfAlgo::SetNodeAttr("transpose_x2", MakeValue(false), matmul);
  return matmul;
}

CNodePtr DynamicRnnGradFissionV2::CreateTranspose(const FuncGraphPtr &func_graph, const AnfNodePtr &dw_reduce_sum,
                                                  const RNNShapeSpecs &specs) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> transpose_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimTranspose->name())),
                                              dw_reduce_sum};
  auto transpose = NewCNode(transpose_inputs, func_graph);
  std::vector<size_t> out_shape = {specs.input_size + specs.hidden_size, kDimMultiNum * specs.hidden_size};
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(dw_reduce_sum, 0)}, {out_shape},
                                      transpose.get());
  AnfAlgo::SetNodeAttr(kAttrPerm, MakeValue(std::vector<int64_t>{1, 0, 2, 3}), transpose);
  AnfAlgo::SetNodeAttr(kAttrInputSize, MakeValue(SizeToLong(specs.input_size)), transpose);
  AnfAlgo::SetNodeAttr(kAttrHiddenSize, MakeValue(SizeToLong(specs.hidden_size)), transpose);
  AnfAlgo::SetNodeAttr(kAttrFixedInputFormat, MakeValue(std::vector<string>{kOpFormat_FRAC_NZ}), transpose);
  AnfAlgo::SetNodeAttr(kAttrFixedOutputFormat, MakeValue(std::vector<string>{kOpFormat_FRACTAL_ZN_RNN}), transpose);
  std::vector<size_t> in_shape = {kDimMultiNum * specs.hidden_nz_size, specs.input_nz_size + specs.hidden_nz_size,
                                  kCubeSize, kCubeSize};
  AnfAlgo::SetNodeAttr(kAttrFixedInputDeviceShape, MakeValue(std::vector<std::vector<int64_t>>{Convert2Long(in_shape)}),
                       transpose);
  return transpose;
}

AnfNodePtr DynamicRnnGradFissionV2::CreateDwReduceSum(const FuncGraphPtr &func_graph,
                                                      const CNodePtr &dynamic_rnn_grad_cnode, const AnfNodePtr &matmul,
                                                      const RNNShapeSpecs &specs) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  // Create node
  std::vector<AnfNodePtr> reduce_sum_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceSum->name())),
                                               matmul};
  auto reduce_sum = NewCNode(reduce_sum_inputs, func_graph);
  // Set infer data type and shape
  std::vector<size_t> reduce_sum_shape = {specs.input_size + specs.hidden_size,
                                          kDimMultiNum * specs.hidden_nz_size * kCubeSize};
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(dynamic_rnn_grad_cnode, 0)}, {reduce_sum_shape},
                                      reduce_sum.get());
  // Set attr
  AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(std::vector<int64_t>{0}), reduce_sum);
  AnfAlgo::SetNodeAttr(kAttrKeepDims, MakeValue(false), reduce_sum);
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), reduce_sum);
  if (specs.shape_need_align) {
    std::vector<size_t> in_shape = {specs.t_size, kDimMultiNum * specs.hidden_nz_size,
                                    specs.input_nz_size + specs.hidden_nz_size, kCubeSize, kCubeSize};
    std::vector<size_t> out_shape = {kDimMultiNum * specs.hidden_nz_size, specs.input_nz_size + specs.hidden_nz_size,
                                     kCubeSize, kCubeSize};
    AnfAlgo::SetNodeAttr(kAttrFixedInputDeviceShape,
                         MakeValue(std::vector<std::vector<int64_t>>{Convert2Long(in_shape)}), reduce_sum);
    AnfAlgo::SetNodeAttr(kAttrFixedOutputDeviceShape,
                         MakeValue(std::vector<std::vector<int64_t>>{Convert2Long(out_shape)}), reduce_sum);
  }

  auto ret_node = reduce_sum;
  if (specs.shape_need_align) {
    ret_node = CreateTranspose(func_graph, reduce_sum, specs);
  }
  return ret_node;
}

AnfNodePtr DynamicRnnGradFissionV2::CreateDwReshape(const FuncGraphPtr &func_graph,
                                                    const CNodePtr &dynamic_rnn_grad_cnode, const AnfNodePtr &matmul,
                                                    const RNNShapeSpecs &specs) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  // Create node
  std::vector<AnfNodePtr> reshape_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReshape->name())),
                                            matmul};
  auto reshape = NewCNode(reshape_inputs, func_graph);
  // Set infer data type and shape
  std::vector<size_t> out_shape = {specs.input_size + specs.hidden_size,
                                   kDimMultiNum * specs.hidden_nz_size * kCubeSize};
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(dynamic_rnn_grad_cnode, 0)}, {out_shape},
                                      reshape.get());
  AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), reshape);

  auto ret_node = reshape;
  if (specs.shape_need_align) {
    ret_node = CreateTranspose(func_graph, reshape, specs);
  }
  return ret_node;
}

AnfNodePtr DynamicRnnGradFissionV2::CreateValueNode(const FuncGraphPtr &func_graph,
                                                    const CNodePtr &dynamic_rnn_grad_cnode,
                                                    const RNNShapeSpecs &specs) const {
  std::vector<size_t> shape;
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    shape = {IntToSize(1), specs.t_size * specs.batch_size};
  } else {
    shape = {specs.t_size, IntToSize(1), specs.batch_size};
  }
  std::vector<int64_t> output_shape = Convert2Long(shape);
  std::vector<int64_t> output_tensor = {SizeToLong(specs.t_size) * SizeToLong(specs.batch_size)};
  auto tensor = TensorConstructUtils::CreateOnesTensor(kFloat32, output_tensor);
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, output_shape);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  auto value_node = kernel_graph->NewValueNode(x_abstract, tensor);
  kernel_graph->AddValueNodeToGraph(value_node);
  AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {shape}, value_node.get());
  return value_node;
}

AnfNodePtr DynamicRnnGradFissionV2::CreateDbReduceSum(const FuncGraphPtr &func_graph, const CNodePtr &,
                                                      const AnfNodePtr &lstm_input_grad, const AnfNodePtr &value_node,
                                                      const RNNShapeSpecs &specs) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  // Create node
  auto matmul = CreateMatMulNode2(func_graph, lstm_input_grad, value_node, specs);
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    std::vector<AnfNodePtr> reshape_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReshape->name())),
                                              matmul};
    auto reshape = NewCNode(reshape_inputs, func_graph);
    std::vector<size_t> out_shape = {kDimMultiNum * specs.hidden_size};
    AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat16}, {out_shape}, reshape.get());
    AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), reshape);
    return reshape;
  } else {
    std::vector<AnfNodePtr> reduce_sum_inputs = {
      NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceSum->name())), matmul};
    auto reduce_sum = NewCNode(reduce_sum_inputs, func_graph);
    // Set infer data type and shape
    std::vector<size_t> out_shape = {kDimMultiNum * specs.hidden_size};
    AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat16}, {out_shape}, reduce_sum.get());
    // Set attr
    AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(std::vector<int64_t>{0}), reduce_sum);
    AnfAlgo::SetNodeAttr(kAttrKeepDims, MakeValue(false), reduce_sum);
    AnfAlgo::SetNodeAttr("is_backend_insert", MakeValue(true), reduce_sum);
    if (specs.shape_need_align) {
      AnfAlgo::SetNodeAttr(kAttrInputSize, MakeValue(SizeToLong(specs.input_size)), reduce_sum);
      AnfAlgo::SetNodeAttr(kAttrHiddenSize, MakeValue(SizeToLong(specs.hidden_size)), reduce_sum);
      AnfAlgo::SetNodeAttr(kAttrFixedInputFormat, MakeValue(std::vector<string>{kOpFormat_DEFAULT}), reduce_sum);
      AnfAlgo::SetNodeAttr(kAttrFixedOutputFormat, MakeValue(std::vector<string>{kOpFormat_ND_RNN_BIAS}), reduce_sum);
    }
    return reduce_sum;
  }
}

const BaseRef DynamicRnnGradFissionV2::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimDynamicRNNGrad, Xs});
}

const AnfNodePtr DynamicRnnGradFissionV2::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto dynamic_rnn_grad_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dynamic_rnn_grad_cnode);
  if (dynamic_rnn_grad_cnode->inputs().size() < kDynamicRNNGradInputNum + 1) {
    MS_LOG(INFO) << "The node " << dynamic_rnn_grad_cnode->DebugString() << " has less than "
                 << (kDynamicRNNGradInputNum + 1) << " inputs";
    return nullptr;
  }
  if (AnfAlgo::IsDynamicShape(node)) {
    MS_LOG(INFO) << "DynamicRNNGrad is dynamic shape, can not do fission.";
    return nullptr;
  }
  auto input0_shape = AnfAlgo::GetOutputInferShape(dynamic_rnn_grad_cnode->input(kIndex1), 0);
  RNNShapeSpecs specs;
  specs.t_size = input0_shape[0];
  specs.batch_size = input0_shape[1];
  specs.input_size = input0_shape[kDim2];
  specs.hidden_size = AnfAlgo::GetOutputInferShape(dynamic_rnn_grad_cnode->input(kIndex7), 0)[kDim2];
  if (specs.input_size % kCubeSize != 0 || specs.hidden_size % kCubeSize != 0) {
    specs.shape_need_align = true;
    SetAttrInputAndHiddenSize(func_graph, dynamic_rnn_grad_cnode, SizeToLong(specs.input_size),
                              SizeToLong(specs.hidden_size));
  }
  specs.batch_nz_size = (specs.batch_size + kCubeSize - 1) / kCubeSize;
  specs.input_nz_size = (specs.input_size + kCubeSize - 1) / kCubeSize;
  specs.hidden_nz_size = (specs.hidden_size + kCubeSize - 1) / kCubeSize;

  std::vector<AnfNodePtr> new_outputs;
  auto lstm_input_grad = AddLSTMInputGradNode(func_graph, dynamic_rnn_grad_cnode, specs, &new_outputs);
  AnfNodePtr concat = nullptr;
  if (specs.t_size != 1) {
    auto splitv = CreateSplitV(func_graph, dynamic_rnn_grad_cnode, specs);
    auto h_concat = CreateHConcat(func_graph, dynamic_rnn_grad_cnode, splitv, specs);
    concat = CreateConcat(func_graph, dynamic_rnn_grad_cnode, h_concat, specs);
  } else {
    concat = CreateConcatNodeT1(func_graph, dynamic_rnn_grad_cnode, specs);
  }

  auto matmul = CreateMatMulNode(func_graph, lstm_input_grad, concat, specs);
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  if (specs.batch_size % kCubeSize == 0 && !specs.shape_need_align) {
    make_tuple_inputs.push_back(matmul);
  } else if (specs.t_size != 1) {
    auto dw_reduce_sum = CreateDwReduceSum(func_graph, dynamic_rnn_grad_cnode, matmul, specs);
    make_tuple_inputs.push_back(dw_reduce_sum);
  } else {
    auto dw_reshape = CreateDwReshape(func_graph, dynamic_rnn_grad_cnode, matmul, specs);
    make_tuple_inputs.push_back(dw_reshape);
  }

  auto value_node = CreateValueNode(func_graph, dynamic_rnn_grad_cnode, specs);
  // create reduce_sum_2
  auto db_reduce_sum = CreateDbReduceSum(func_graph, dynamic_rnn_grad_cnode, lstm_input_grad, value_node, specs);
  make_tuple_inputs.emplace_back(db_reduce_sum);
  make_tuple_inputs.insert(make_tuple_inputs.end(), new_outputs.begin(), new_outputs.end());
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}
}  // namespace opt
}  // namespace mindspore
