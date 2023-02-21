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

#include "plugin/device/ascend/optimizer/ge/sparse_softmax_cross_entropy_with_logits_split.h"

#include <string>
#include <memory>
#include <vector>

#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kMulOpInputNum = 2;
constexpr size_t kDependOpInputNum = 2;

constexpr size_t kSparseSoftmaxCrossEntropyWithLogitsInputTensorsNum = 2;
constexpr size_t kSparseSoftmaxCrossEntropyWithLogitsInputNum = 3;
constexpr size_t kRealDivOpInputNum = 2;
constexpr size_t kSoftmaxCrossEntropyWithLogitsOpOutputNum = 2;
constexpr size_t kSoftmaxOutputShapeSize = 2;
constexpr char kIsFeatureMapInputList[] = "IsFeatureMapInputList";
}  // namespace

ValueNodePtr CreateValueNode(const ValuePtr &value_ptr) {
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto new_node = std::make_shared<ValueNode>(value_ptr);
  MS_EXCEPTION_IF_NULL(new_node);
  auto value_abstract = value_ptr->ToAbstract();
  new_node->set_abstract(value_abstract);
  return new_node;
}

CNodePtr CreateOneHot(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);

  auto logits_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 0);
  int64_t depth = 0;
  if (!logits_shape.empty()) {
    size_t index = logits_shape.size() - 1;
    depth = logits_shape[index];
  } else {
    MS_LOG(EXCEPTION) << "Logits's shape of node [" << sparse_softmax_node->DebugString() << "] is empty"
                      << trace::DumpSourceLines(sparse_softmax_node);
  }

  auto value_on = std::make_shared<tensor::Tensor>(1.0, kFloat32);
  auto value_on_node = CreateValueNode(value_on);
  MS_EXCEPTION_IF_NULL(value_on_node);
  auto value_off = std::make_shared<tensor::Tensor>(0.0, kFloat32);
  auto value_off_node = CreateValueNode(value_off);
  MS_EXCEPTION_IF_NULL(value_off_node);

  auto one_hot_primitive = std::make_shared<Primitive>(kOneHotOpName);
  std::vector<std::string> input_names = {"indices", "depth", "on_value", "off_value"};
  std::vector<std::string> output_names = {"output"};
  one_hot_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  one_hot_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));

  std::vector<AnfNodePtr> one_hot_inputs;
  auto depth_node = NewValueNode(depth);
  MS_EXCEPTION_IF_NULL(depth_node);
  auto depth_abstract = std::make_shared<abstract::AbstractScalar>();
  MS_EXCEPTION_IF_NULL(depth_abstract);
  depth_abstract->set_type(kInt64);
  depth_node->set_abstract(depth_abstract);
  one_hot_inputs = {NewValueNode(one_hot_primitive), sparse_softmax_node->input(kIndex2), depth_node, value_on_node,
                    value_off_node};
  auto one_hot_node = graph->NewCNode(one_hot_inputs);
  MS_EXCEPTION_IF_NULL(one_hot_node);
  one_hot_node->set_scope(sparse_softmax_node->scope());
  auto labels_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 1);
  (void)labels_shape.emplace_back(depth);
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {labels_shape}, one_hot_node.get());
  return one_hot_node;
}

CNodePtr CreateSoftmaxCrossEntropyWithLogits(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node,
                                             const CNodePtr &one_hot_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(one_hot_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorsNum);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(kSoftmaxCrossEntropyWithLogitsOpName)),
                                    sparse_softmax_node->input(kIndex1), one_hot_node};
  auto softmax_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(softmax_node);
  softmax_node->set_scope(sparse_softmax_node->scope());

  auto labels_shape = common::AnfAlgo::GetOutputInferShape(one_hot_node, 0);
  ShapeVector loss_shape;
  if (!labels_shape.empty()) {
    (void)loss_shape.emplace_back(labels_shape[0]);
  } else {
    MS_LOG(EXCEPTION) << "One_hot output's shape is empty." << trace::DumpSourceLines(one_hot_node);
  }

  auto data_types = common::AnfAlgo::GetOutputInferDataType(one_hot_node, 0);
  auto types = {data_types, data_types};
  // Loss shape:(N,) labels shape:(N,C)
  auto shapes = {loss_shape, labels_shape};
  common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, softmax_node.get());
  return softmax_node;
}

void CreateMultiOutputsOfAnfNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t output_num,
                                 std::vector<AnfNodePtr> *outputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto type_ptr = node->Type();
  auto shape_ptr = node->Shape();
  for (size_t i = 0; i < output_num; i++) {
    int64_t temp = SizeToLong(i);
    auto idx = NewValueNode(temp);
    MS_EXCEPTION_IF_NULL(idx);
    auto imm = std::make_shared<Int64Imm>(temp);
    auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
    idx->set_abstract(abstract_scalar);
    auto tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
    tuple_getitem->set_abstract(idx->abstract());
    MS_EXCEPTION_IF_NULL(tuple_getitem);
    common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(type_ptr, i)},
                                                {common::AnfAlgo::GetOutputInferShape(node, shape_ptr, i)},
                                                tuple_getitem.get());
    outputs->push_back(tuple_getitem);
  }
}

std::vector<int64_t> GetAxis(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto output_shape = common::AnfAlgo::GetOutputInferShape(node, 0);
  if (output_shape.empty()) {
    MS_LOG(EXCEPTION) << node->fullname_with_scope() << "'s output shape is empty" << trace::DumpSourceLines(node);
  }
  std::vector<int64_t> range;
  for (size_t i = 0; i < output_shape.size(); i++) {
    (void)range.emplace_back(i);
  }
  return range;
}

ValueNodePtr GetAxisNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto range = GetAxis(node);
  auto axis_node = CreateValueNode(MakeValue(range));
  MS_EXCEPTION_IF_NULL(axis_node);
  return axis_node;
}

CNodePtr CreateReduceMean(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node,
                          const AnfNodePtr &softmax_output_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(softmax_output_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorsNum);

  auto axis_node = GetAxisNode(softmax_output_node);
  MS_EXCEPTION_IF_NULL(axis_node);

  std::vector<std::string> input_names = {"x", "axis"};
  std::vector<std::string> output_names = {"y"};
  auto reduce_node =
    graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kReduceMeanOpName)), softmax_output_node, axis_node});
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), reduce_node);
  common::AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), reduce_node);
  MS_EXCEPTION_IF_NULL(reduce_node);
  reduce_node->set_scope(sparse_softmax_node->scope());
  auto reduce_abstract = softmax_output_node->abstract();
  reduce_abstract->set_shape(std::make_shared<abstract::Shape>());
  reduce_node->set_abstract(reduce_abstract);
  return reduce_node;
}

CNodePtr CreateMul(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node,
                   const AnfNodePtr &softmax_output_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(softmax_output_node);
  auto softmax_output_shape = common::AnfAlgo::GetOutputInferShape(softmax_output_node, 0);
  if (softmax_output_shape.size() != kSoftmaxOutputShapeSize) {
    MS_LOG(EXCEPTION) << "SoftmaxCrossEntropyWithLogits the second output shape size should be "
                      << kSoftmaxOutputShapeSize << ", but got " << softmax_output_shape.size()
                      << trace::DumpSourceLines(softmax_output_node);
  }
  ShapeVector tensor_shape;
  (void)tensor_shape.emplace_back(softmax_output_shape[0]);
  (void)tensor_shape.emplace_back(1);
  if (softmax_output_shape[0] == 0) {
    MS_LOG(EXCEPTION) << "Output_shape[0] of softmax should not be 0" << trace::DumpSourceLines(softmax_output_node);
  }
  std::vector<float> tensor_value(softmax_output_shape[0], 1.0 / softmax_output_shape[0]);
  auto buf_size = sizeof(float) * tensor_value.size();
  auto tensor_y = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, tensor_shape, tensor_value.data(), buf_size);
  auto y_node = CreateValueNode(tensor_y);
  MS_EXCEPTION_IF_NULL(y_node);
  std::vector<std::string> input_names = {"x", "y"};
  std::vector<std::string> output_names = {"output"};
  auto mul_node = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kMulOpName)), softmax_output_node, y_node});
  MS_EXCEPTION_IF_NULL(mul_node);
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), mul_node);
  common::AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), mul_node);
  mul_node->set_scope(sparse_softmax_node->scope());
  mul_node->set_abstract(softmax_output_node->abstract());
  return mul_node;
}

CNodePtr CreateTile(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node, const CNodePtr &mul_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(mul_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorsNum);
  CheckCNodeInputSize(mul_node, kMulOpInputNum);

  auto multiple_value = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 1);
  if (std::all_of(multiple_value.begin(), multiple_value.end(), [](int64_t value) { return value == 1; })) {
    return nullptr;
  }
  auto multiples = MakeValue(multiple_value);
  auto multiples_node = CreateValueNode(multiples);
  MS_EXCEPTION_IF_NULL(multiples_node);

  std::vector<std::string> input_names = {"x", "multiples"};
  std::vector<std::string> output_names = {"output"};

  auto tile_node =
    graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kTileOpName)), mul_node->input(2), multiples_node});
  MS_EXCEPTION_IF_NULL(tile_node);
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), tile_node);
  common::AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), tile_node);

  tile_node->set_scope(mul_node->scope());
  common::AnfAlgo::SetOutputTypeAndDetailShape({common::AnfAlgo::GetPrevNodeOutputInferDataType(mul_node, 1)},
                                               {AnfAlgo::GetPrevNodeOutputDetailShape(sparse_softmax_node, 1)},
                                               tile_node.get());
  // Feature map set
  std::vector<size_t> feature_map_input_indexs;
  (void)feature_map_input_indexs.emplace_back(0);
  common::AnfAlgo::SetNodeAttr(kIsFeatureMapInputList, MakeValue(feature_map_input_indexs), tile_node);
  return tile_node;
}

CNodePtr CreateRealDiv(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node, const AnfNodePtr &tile_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(tile_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorsNum);
  auto labels_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 1);
  if (labels_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "Label's shape should be 1-D, but got " << labels_shape.size()
                      << trace::DumpSourceLines(sparse_softmax_node);
  }
  auto y_value = static_cast<float>(labels_shape[0]);
  auto y = std::make_shared<tensor::Tensor>(y_value, kFloat32);
  auto y_node = CreateValueNode(y);
  MS_EXCEPTION_IF_NULL(y_node);

  std::vector<std::string> input_names = {"x", "y"};
  std::vector<std::string> output_names = {"output"};
  auto real_div_node = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kDivOpName)), tile_node, y_node});
  MS_EXCEPTION_IF_NULL(real_div_node);
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), real_div_node);
  common::AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), real_div_node);
  real_div_node->set_scope(sparse_softmax_node->scope());
  real_div_node->set_abstract(tile_node->abstract());
  return real_div_node;
}

CNodePtr CreateExpandDims(const FuncGraphPtr &graph, const CNodePtr &real_div_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(real_div_node);
  CheckCNodeInputSize(real_div_node, kRealDivOpInputNum);

  int64_t axis = -1;
  auto axis_node = NewValueNode(axis);
  MS_EXCEPTION_IF_NULL(axis_node);
  auto axis_abstract = std::make_shared<abstract::AbstractScalar>();
  MS_EXCEPTION_IF_NULL(axis_abstract);
  axis_abstract->set_type(kInt64);
  axis_node->set_abstract(axis_abstract);
  std::vector<std::string> input_names = {"x", "axis"};
  std::vector<std::string> output_names = {"output"};
  auto expand_dims_node =
    graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kExpandDimsOpName)), real_div_node, axis_node});
  MS_EXCEPTION_IF_NULL(expand_dims_node);
  common::AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names), expand_dims_node);
  common::AnfAlgo::SetNodeAttr(kAttrOutputNames, MakeValue(output_names), expand_dims_node);
  expand_dims_node->set_scope(real_div_node->scope());
  auto y_shape = common::AnfAlgo::GetOutputInferShape(real_div_node, 0);
  (void)y_shape.emplace_back(1);
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(real_div_node, 0)}, {y_shape},
                                              expand_dims_node.get());
  return expand_dims_node;
}

bool IsSparseSoftmaxCrossEntropyWithLogitsGrad(const CNodePtr &sparse) {
  MS_EXCEPTION_IF_NULL(sparse);
  if (common::AnfAlgo::GetCNodeName(sparse) != kSparseSoftmaxCrossEntropyWithLogitsOpName) {
    MS_LOG(EXCEPTION) << "Input node should be " << kSparseSoftmaxCrossEntropyWithLogitsOpName << ", but got "
                      << common::AnfAlgo::GetCNodeName(sparse) << trace::DumpSourceLines(sparse);
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrIsGrad, sparse)) {
    return common::AnfAlgo::GetNodeAttr<bool>(sparse, kAttrIsGrad);
  } else {
    MS_LOG(EXCEPTION) << "Node of " << sparse->fullname_with_scope() << " does not have the attr " << kAttrIsGrad
                      << trace::DumpSourceLines(sparse);
  }
}

CNodePtr GetDependNode(const CNodePtr &mul_node) {
  MS_EXCEPTION_IF_NULL(mul_node);
  CheckCNodeInputSize(mul_node, kMulOpInputNum);
  auto depend_node = mul_node->input(kIndex1);
  MS_EXCEPTION_IF_NULL(depend_node);
  return depend_node->cast<CNodePtr>();
}

CNodePtr GetSparseNode(const CNodePtr &depend_node, size_t index) {
  MS_EXCEPTION_IF_NULL(depend_node);
  CheckCNodeInputSize(depend_node, kDependOpInputNum);
  auto sparse_node = depend_node->input(index);
  MS_EXCEPTION_IF_NULL(sparse_node);
  return sparse_node->cast<CNodePtr>();
}

CNodePtr HandleTrain(const FuncGraphPtr &fg, const AnfNodePtr &node) {
  MS_LOG(INFO) << "HandleTrain for SparseSoftmaxCrossEntropyWithLogi split start.";
  auto depend_node = node->cast<CNodePtr>();
  auto inputs = depend_node->inputs();
  if (inputs.size() != kDependInputSize) {
    MS_LOG(EXCEPTION) << "Check depend input size failed!";
  }
  auto sparse_softmax_node_grad_anf = inputs[1];
  MS_EXCEPTION_IF_NULL(sparse_softmax_node_grad_anf);
  auto sparse_softmax_node_grad = sparse_softmax_node_grad_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sparse_softmax_node_grad);
  auto sparse_softmax_node_anf = inputs[2];
  MS_EXCEPTION_IF_NULL(sparse_softmax_node_anf);
  auto sparse_softmax_node = sparse_softmax_node_anf->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);

  auto one_hot_node = CreateOneHot(fg, sparse_softmax_node);
  auto softmax_node = CreateSoftmaxCrossEntropyWithLogits(fg, sparse_softmax_node, one_hot_node);

  std::vector<AnfNodePtr> softmax_node_outputs;
  CreateMultiOutputsOfAnfNode(fg, softmax_node, kSoftmaxCrossEntropyWithLogitsOpOutputNum, &softmax_node_outputs);
  auto reduce_node = CreateReduceMean(fg, sparse_softmax_node_grad, softmax_node_outputs[0]);
  auto mul_node = CreateMul(fg, sparse_softmax_node_grad, softmax_node_outputs[1]);

  auto manager = fg->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(sparse_softmax_node, reduce_node);
  MS_LOG(INFO) << "HandleTrain for SparseSoftmaxCrossEntropyWithLogits split success.";
  return mul_node;
}

CNodePtr HandleTrainWithLossScale(const FuncGraphPtr &fg, const AnfNodePtr &node) {
  MS_LOG(INFO) << "HandleTrainWithLossScale for SparseSoftmaxCrossEntropyWithLogi split start.";
  auto mul_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul_node);

  auto depend_node = GetDependNode(mul_node);
  auto sparse_softmax_node = GetSparseNode(depend_node, kIndex2);
  auto sparse_softmax_node_grad = GetSparseNode(depend_node, 1);
  CheckCNodeInputSize(sparse_softmax_node_grad, kSparseSoftmaxCrossEntropyWithLogitsInputTensorsNum);

  auto one_hot_node = CreateOneHot(fg, sparse_softmax_node);
  auto softmax_node = CreateSoftmaxCrossEntropyWithLogits(fg, sparse_softmax_node, one_hot_node);

  std::vector<AnfNodePtr> softmax_node_outputs;
  CreateMultiOutputsOfAnfNode(fg, softmax_node, kSoftmaxCrossEntropyWithLogitsOpOutputNum, &softmax_node_outputs);
  auto reduce_node = CreateReduceMean(fg, sparse_softmax_node_grad, softmax_node_outputs[0]);
  auto tile_node = CreateTile(fg, sparse_softmax_node_grad, mul_node);
  CNodePtr real_div_node;
  if (tile_node == nullptr) {
    real_div_node = CreateRealDiv(fg, sparse_softmax_node_grad, mul_node->input(kIndex2));
  } else {
    real_div_node = CreateRealDiv(fg, sparse_softmax_node_grad, tile_node);
  }
  auto expand_dims_node = CreateExpandDims(fg, real_div_node);
  auto new_mul_node =
    fg->NewCNode({NewValueNode(std::make_shared<Primitive>(kMulOpName)), softmax_node_outputs[1], expand_dims_node});
  MS_EXCEPTION_IF_NULL(new_mul_node);

  new_mul_node->set_scope(mul_node->scope());
  new_mul_node->set_abstract(mul_node->abstract());

  auto manager = fg->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(sparse_softmax_node, reduce_node);
  (void)manager->Replace(mul_node, new_mul_node);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                    NewValueNode(MakeValue<bool>(true)), NewValueNode(MakeValue<bool>(true))};
  auto new_depend = fg->NewCNode(inputs);
  (void)manager->Replace(sparse_softmax_node_grad, new_depend);
  MS_LOG(INFO) << "HandleTrainWithLossScale for SparseSoftmaxCrossEntropyWithLogits split success.";
  return new_mul_node;
}

CNodePtr HandleInfer(const FuncGraphPtr &fg, const AnfNodePtr &node) {
  MS_LOG(INFO) << "HandleInfer for SparseSoftmaxCrossEntropyWithLogi split start.";
  auto sparse_softmax_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorsNum);

  if (IsSparseSoftmaxCrossEntropyWithLogitsGrad(sparse_softmax_node)) {
    return nullptr;
  }

  auto one_hot_node = CreateOneHot(fg, sparse_softmax_node);
  auto softmax_node = CreateSoftmaxCrossEntropyWithLogits(fg, sparse_softmax_node, one_hot_node);

  std::vector<AnfNodePtr> softmax_node_outputs;
  CreateMultiOutputsOfAnfNode(fg, softmax_node, kSoftmaxCrossEntropyWithLogitsOpOutputNum, &softmax_node_outputs);
  auto reduce_node = CreateReduceMean(fg, sparse_softmax_node, softmax_node_outputs[0]);
  MS_LOG(INFO) << "HandleInfer for SparseSoftmaxCrossEntropyWithLogits split success.";
  return reduce_node;
}

bool IsTargetMulNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!IsPrimitiveCNode(cnode, prim::kPrimMul)) {
    return false;
  }
  CheckCNodeInputSize(cnode, kMulOpInputNum);
  auto inputs = cnode->inputs();
  // Mul(depend, load)
  return IsPrimitiveCNode(inputs[kIndex1], prim::kPrimDepend) && IsPrimitiveCNode(inputs[kIndex2], prim::kPrimLoad);
}

const AnfNodePtr SparseSoftmaxCrossEntropyWithLogitsSplit::Process(const FuncGraphPtr &, const AnfNodePtr &node,
                                                                   const EquivPtr &) const {
  MS_LOG(INFO) << "SparseSoftmaxCrossEntropyWithLogits split start.";
  FuncGraphPtr fg = node->func_graph();
  if (fg != nullptr) {
    if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
      return HandleTrain(fg, node);
    } else if (IsPrimitiveCNode(node, prim::kPrimSparseSoftmaxCrossEntropyWithLogits)) {
      return HandleInfer(fg, node);
    }
  }
  return nullptr;
}

const BaseRef SparseSoftmaxCrossEntropyWithLogitsSplitCond1::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  VectorRef npu_alloc_float_status({prim::kPrimNPUAllocFloatStatus});
  VectorRef sparse_softmax_cross_entropy_with_logits({prim::kPrimSparseSoftmaxCrossEntropyWithLogits, x1, x2});
  return VectorRef({prim::kPrimDepend, npu_alloc_float_status, sparse_softmax_cross_entropy_with_logits});
}

const AnfNodePtr SparseSoftmaxCrossEntropyWithLogitsSplitCond1::Process(const FuncGraphPtr &, const AnfNodePtr &node,
                                                                        const EquivPtr &) const {
  MS_LOG(INFO) << "SparseSoftmaxCrossEntropyWithLogits split start.";
  FuncGraphPtr fg = node->func_graph();
  if (fg != nullptr) {
    if (IsTargetMulNode(node)) {
      return HandleTrainWithLossScale(fg, node);
    } else {
      return nullptr;
    }
  }
  return nullptr;
}

const BaseRef SparseSoftmaxCrossEntropyWithLogitsSplitCond2::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  VectorRef sparse_softmax_cross_entropy_with_logits_grad({prim::kPrimSparseSoftmaxCrossEntropyWithLogits, x1, x2});
  VectorRef sparse_softmax_cross_entropy_with_logits({prim::kPrimSparseSoftmaxCrossEntropyWithLogits, x1, x2});
  return VectorRef(
    {prim::kPrimDepend, sparse_softmax_cross_entropy_with_logits_grad, sparse_softmax_cross_entropy_with_logits});
}

const BaseRef SparseSoftmaxCrossEntropyWithLogitsSplitInfer::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  return VectorRef({prim::kPrimSparseSoftmaxCrossEntropyWithLogits, x1, x2});
}
}  // namespace opt
}  // namespace mindspore
