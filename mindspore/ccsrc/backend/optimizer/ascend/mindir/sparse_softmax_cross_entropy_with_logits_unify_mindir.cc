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

#include "backend/optimizer/ascend/mindir/sparse_softmax_cross_entropy_with_logits_unify_mindir.h"
#include <vector>
#include <string>
#include <algorithm>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/helper.h"
#include "utils/utils.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/dtype/type_id.h"
#include "ir/dtype/type.h"

constexpr auto softmax_output_shape_size = 2;
constexpr auto kAttrDepth = "depth";
constexpr auto kAttrMultiples = "multiples";
constexpr auto kIsFeatureMapInputList = "IsFeatureMapInputList";
namespace mindspore {
namespace opt {
namespace {
ValueNodePtr CreateValueNode(const ValuePtr &value_ptr, TypeId output_type) {
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto new_node = std::make_shared<ValueNode>(value_ptr);
  MS_EXCEPTION_IF_NULL(new_node);
  auto value_abstract = value_ptr->ToAbstract();
  new_node->set_abstract(value_abstract);

  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  new_node->set_kernel_info(kernel_info);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetOutputsFormat({kOpFormat_DEFAULT});
  builder1.SetOutputsDeviceType({output_type});
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), new_node.get());
  return new_node;
}

CNodePtr CreateOneHot(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node,
                      bool is_convert_const_to_attr = false) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);

  std::vector<size_t> logits_shape = AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 0);
  int64_t depth = 0;
  if (logits_shape.size() >= 1) {
    size_t index = logits_shape.size() - 1;
    depth = logits_shape[index];
  } else {
    MS_LOG(EXCEPTION) << "logits's shape of sparse_softmax_cross_entropy_with_logits is empty.";
  }

  auto value_on = std::make_shared<tensor::Tensor>(1.0, kFloat32);
  auto value_on_node = CreateValueNode(value_on, kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(value_on_node);
  auto value_off = std::make_shared<tensor::Tensor>(0.0, kFloat32);
  auto value_off_node = CreateValueNode(value_off, kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(value_off_node);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  kernel_graph->AddValueNodeToGraph(value_on_node);
  kernel_graph->AddValueNodeToGraph(value_off_node);

  auto one_hot_primitive = std::make_shared<Primitive>(kOneHotOpName);
  std::vector<std::string> input_names = {"indices", "depth", "on_value", "off_value"};
  std::vector<std::string> output_names = {"output"};
  one_hot_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  one_hot_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));

  std::vector<AnfNodePtr> one_hot_inputs;
  if (is_convert_const_to_attr) {
    one_hot_inputs = {NewValueNode(one_hot_primitive), sparse_softmax_node->input(2), value_on_node, value_off_node};
  } else {
    auto depth_node = NewValueNode(depth);
    MS_EXCEPTION_IF_NULL(depth_node);
    auto depth_abstract = std::make_shared<abstract::AbstractScalar>();
    depth_abstract->set_type(kInt64);
    depth_node->set_abstract(depth_abstract);
    one_hot_inputs = {NewValueNode(one_hot_primitive), sparse_softmax_node->input(2), depth_node, value_on_node,
                      value_off_node};
  }
  auto one_hot_node = graph->NewCNode(one_hot_inputs);
  MS_EXCEPTION_IF_NULL(one_hot_node);
  one_hot_node->set_scope(sparse_softmax_node->scope());
  std::vector<size_t> labels_shape = AnfAlgo ::GetPrevNodeOutputInferShape(sparse_softmax_node, 1);
  labels_shape.emplace_back(depth);
  AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {labels_shape}, one_hot_node.get());
  if (is_convert_const_to_attr) {
    AnfAlgo::SetNodeAttr(kAttrDepth, MakeValue(depth), one_hot_node);
  }
  return one_hot_node;
}

CNodePtr CreateSoftmaxCrossEntropyWithLogits(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node,
                                             const CNodePtr &one_hot_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(one_hot_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(kSoftmaxCrossEntropyWithLogitsOpName)),
                                    sparse_softmax_node->input(1), one_hot_node};
  auto softmax_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(softmax_node);
  softmax_node->set_scope(sparse_softmax_node->scope());

  std::vector<size_t> labels_shape = AnfAlgo::GetOutputInferShape(one_hot_node, 0);
  std::vector<size_t> loss_shape;
  if (!labels_shape.empty()) {
    loss_shape.emplace_back(labels_shape[0]);
  } else {
    MS_LOG(EXCEPTION) << "one_hot output's shape is empty.";
  }

  auto shapes = {loss_shape, AnfAlgo::GetOutputInferShape(one_hot_node, 0)};
  auto data_types = AnfAlgo::GetOutputInferDataType(one_hot_node, 0);
  auto types = {data_types, data_types};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, softmax_node.get());
  return softmax_node;
}

std::vector<int64_t> GetAxis(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<size_t> output_shape = AnfAlgo::GetOutputInferShape(node, 0);
  if (output_shape.empty()) {
    MS_LOG(EXCEPTION) << node->fullname_with_scope() << "'s output shape is empty";
  }
  std::vector<int64_t> range;
  for (size_t i = 0; i < output_shape.size(); i++) {
    range.emplace_back(i);
  }
  return range;
}

ValueNodePtr GetAxisNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto range = GetAxis(node);
  auto axis_node = CreateValueNode(MakeValue(range), kNumberTypeInt64);
  MS_EXCEPTION_IF_NULL(axis_node);
  return axis_node;
}

CNodePtr CreateReduceMean(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node,
                          const AnfNodePtr &softmax_output_node, bool is_pynative = false) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(softmax_output_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum);

  auto axis_value = GetAxis(softmax_output_node);
  auto axis_node = GetAxisNode(softmax_output_node);
  MS_EXCEPTION_IF_NULL(axis_node);

  auto reduce_primitive = std::make_shared<Primitive>(kReduceMeanOpName);
  std::vector<std::string> input_names = {"x", "axis"};
  std::vector<std::string> output_names = {"y"};
  reduce_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  reduce_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));

  auto kernel_graph = graph->cast<KernelGraphPtr>();
  std::vector<AnfNodePtr> inputs;
  if (is_pynative) {
    inputs = {NewValueNode(reduce_primitive), softmax_output_node};
  } else {
    kernel_graph->AddValueNodeToGraph(axis_node);
    inputs = {NewValueNode(reduce_primitive), softmax_output_node, axis_node};
  }
  auto reduce_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(reduce_node);
  reduce_node->set_scope(sparse_softmax_node->scope());
  auto reduce_abstract = softmax_output_node->abstract();
  reduce_abstract->set_shape(std::make_shared<abstract::Shape>());
  reduce_node->set_abstract(reduce_abstract);
  if (is_pynative) {
    AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(axis_value), reduce_node);
  }
  return reduce_node;
}

CNodePtr CreateExpandDims(const FuncGraphPtr &graph, const CNodePtr &real_div_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(real_div_node);
  CheckCNodeInputSize(real_div_node, kRealDivInputTensorNum);

  int64_t axis = -1;
  auto axis_node = NewValueNode(axis);
  MS_EXCEPTION_IF_NULL(axis_node);
  auto axis_abstract = std::make_shared<abstract::AbstractScalar>();
  axis_abstract->set_type(kInt64);
  axis_node->set_abstract(axis_abstract);

  auto expand_dims_primitive = std::make_shared<Primitive>(kExpandDimsOpName);
  std::vector<std::string> input_names = {"x", "axis"};
  std::vector<std::string> output_names = {"output"};
  expand_dims_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  expand_dims_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));
  std::vector<AnfNodePtr> expand_dims_inputs = {NewValueNode(expand_dims_primitive), real_div_node, axis_node};
  auto expand_dims_node = graph->NewCNode(expand_dims_inputs);
  MS_EXCEPTION_IF_NULL(expand_dims_node);

  expand_dims_node->set_scope(real_div_node->scope());
  std::vector<size_t> y_shape = AnfAlgo::GetOutputInferShape(real_div_node, 0);
  y_shape.emplace_back(1);
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(real_div_node, 0)}, {y_shape},
                                      expand_dims_node.get());
  return expand_dims_node;
}
CNodePtr CreateExpandDimsPynative(const FuncGraphPtr &graph, const CNodePtr &real_div_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(real_div_node);
  CheckCNodeInputSize(real_div_node, kRealDivInputTensorNum);

  int64_t axis = -1;
  auto expand_dims_primitive = std::make_shared<Primitive>(kExpandDimsOpName);
  std::vector<std::string> input_names = {"x"};
  std::vector<std::string> output_names = {"output"};
  expand_dims_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  expand_dims_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));
  std::vector<AnfNodePtr> expand_dims_inputs = {NewValueNode(expand_dims_primitive), real_div_node};
  auto expand_dims_node = graph->NewCNode(expand_dims_inputs);
  MS_EXCEPTION_IF_NULL(expand_dims_node);

  expand_dims_node->set_scope(real_div_node->scope());
  std::vector<size_t> y_shape = AnfAlgo::GetOutputInferShape(real_div_node, 0);
  y_shape.emplace_back(1);
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(real_div_node, 0)}, {y_shape},
                                      expand_dims_node.get());
  AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(axis), expand_dims_node);
  return expand_dims_node;
}

CNodePtr CreateTile(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node, const CNodePtr &mul_node,
                    bool is_convert_const_to_attr = false) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(mul_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum);
  CheckCNodeInputSize(mul_node, kMulInputTensorNum);

  auto labels_shape = AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 1);
  std::vector<int64_t> multiple_value;
  std::transform(labels_shape.begin(), labels_shape.end(), std::back_inserter(multiple_value),
                 [](size_t label) { return static_cast<int64_t>(label); });
  if (std::all_of(multiple_value.begin(), multiple_value.end(), [](int64_t value) { return value == 1; })) {
    return nullptr;
  }
  auto multiples = MakeValue(multiple_value);
  auto multiples_node = CreateValueNode(multiples, kNumberTypeInt64);
  MS_EXCEPTION_IF_NULL(multiples_node);

  auto tile_primitive = std::make_shared<Primitive>(kTileOpName);
  std::vector<std::string> input_names = {"x", "multiples"};
  std::vector<std::string> output_names = {"output"};
  tile_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  tile_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));

  std::vector<AnfNodePtr> tile_inputs;
  if (is_convert_const_to_attr) {
    tile_inputs = {NewValueNode(tile_primitive), mul_node->input(2)};
  } else {
    auto kernel_graph = graph->cast<KernelGraphPtr>();
    kernel_graph->AddValueNodeToGraph(multiples_node);
    tile_inputs = {NewValueNode(tile_primitive), mul_node->input(2), multiples_node};
  }

  auto tile_node = graph->NewCNode(tile_inputs);
  MS_EXCEPTION_IF_NULL(tile_node);
  tile_node->set_scope(mul_node->scope());
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetPrevNodeOutputInferDataType(mul_node, 1)}, {labels_shape},
                                      tile_node.get());
  if (is_convert_const_to_attr) {
    AnfAlgo::SetNodeAttr(kAttrMultiples, MakeValue(multiples), tile_node);
  }
  // feature map set
  std::vector<size_t> feature_map_input_indexs;
  feature_map_input_indexs.push_back(0);
  AnfAlgo::SetNodeAttr(kIsFeatureMapInputList, MakeValue(feature_map_input_indexs), tile_node);
  return tile_node;
}

CNodePtr CreateRealDiv(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node, const AnfNodePtr &tile_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(tile_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum);
  std::vector<size_t> labels_shape = AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 1);
  if (labels_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "label's shape should be 1-D.";
  }
  auto y_value = static_cast<float>(labels_shape[0]);
  auto y = std::make_shared<tensor::Tensor>(y_value, kFloat32);
  auto y_node = CreateValueNode(y, kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(y_node);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  kernel_graph->AddValueNodeToGraph(y_node);

  auto real_div_primitive = std::make_shared<Primitive>(kRealDivOpName);
  std::vector<std::string> input_names = {"x", "y"};
  std::vector<std::string> output_names = {"output"};
  real_div_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  real_div_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));
  std::vector<AnfNodePtr> real_div_inputs = {NewValueNode(real_div_primitive), tile_node, y_node};
  auto real_div_node = graph->NewCNode(real_div_inputs);
  MS_EXCEPTION_IF_NULL(real_div_node);

  real_div_node->set_scope(sparse_softmax_node->scope());
  real_div_node->set_abstract(tile_node->abstract());
  return real_div_node;
}

CNodePtr GetSparseNode(const CNodePtr &depend_node, size_t index) {
  MS_EXCEPTION_IF_NULL(depend_node);
  CheckCNodeInputSize(depend_node, kDependInputTensorNum);
  auto sparse_node = depend_node->input(index);
  MS_EXCEPTION_IF_NULL(sparse_node);
  return sparse_node->cast<CNodePtr>();
}

CNodePtr GetDependNode(const CNodePtr &mul_node) {
  MS_EXCEPTION_IF_NULL(mul_node);
  CheckCNodeInputSize(mul_node, kMulInputTensorNum);
  auto depend_node = mul_node->input(1);
  MS_EXCEPTION_IF_NULL(depend_node);
  return depend_node->cast<CNodePtr>();
}

CNodePtr CreateMul(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node,
                   const AnfNodePtr &softmax_output_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(softmax_output_node);
  auto softmax_output_shape = AnfAlgo::GetOutputInferShape(softmax_output_node, 0);
  if (softmax_output_shape.size() != softmax_output_shape_size) {
    MS_LOG(EXCEPTION) << "SoftmaxCrossEntropyWithLogits the second output shape size should be "
                      << softmax_output_shape_size << ", but got " << softmax_output_shape.size();
  }
  ShapeVector tensor_shape;
  tensor_shape.emplace_back(softmax_output_shape[0]);
  tensor_shape.emplace_back(1);
  std::vector<float> tensor_value(softmax_output_shape[0], 1.0 / softmax_output_shape[0]);
  auto buf_size = sizeof(float) * tensor_value.size();
  auto tensor_y = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, tensor_shape, tensor_value.data(), buf_size);
  auto y_node = CreateValueNode(tensor_y, kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(y_node);

  auto kernel_graph = graph->cast<KernelGraphPtr>();
  kernel_graph->AddValueNodeToGraph(y_node);

  auto mul_primitive = std::make_shared<Primitive>(kMulOpName);
  std::vector<std::string> input_names = {"x", "y"};
  std::vector<std::string> output_names = {"output"};
  mul_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  mul_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));

  std::vector<AnfNodePtr> mul_input = {NewValueNode(mul_primitive), softmax_output_node, y_node};
  auto mul_node = graph->NewCNode(mul_input);
  MS_EXCEPTION_IF_NULL(mul_node);

  mul_node->set_scope(sparse_softmax_node->scope());
  mul_node->set_abstract(softmax_output_node->abstract());
  return mul_node;
}
}  // namespace

const BaseRef SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  return VectorRef({prim::kPrimSparseSoftmaxCrossEntropyWithLogits, x1, x2});
}

const AnfNodePtr SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR::Process(const FuncGraphPtr &graph,
                                                                         const AnfNodePtr &node,
                                                                         const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto sparse_softmax_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum);
  if (AnfAlgo::HasNodeAttr(kAttrIsGrad, sparse_softmax_node) &&
      AnfAlgo::GetNodeAttr<bool>(sparse_softmax_node, kAttrIsGrad)) {
    return nullptr;
  }

  CNodePtr softmax_node;
  auto one_hot_node = CreateOneHot(graph, sparse_softmax_node);
  softmax_node = CreateSoftmaxCrossEntropyWithLogits(graph, sparse_softmax_node, one_hot_node);

  std::vector<AnfNodePtr> softmax_node_outputs;
  CreateMultipleOutputsOfAnfNode(graph, softmax_node, kSoftmaxCrossEntropyWithLogitsOutputNum, &softmax_node_outputs);
  auto reduce_node = CreateReduceMean(graph, sparse_softmax_node, softmax_node_outputs[0]);
  return reduce_node;
}

const BaseRef GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  VarPtr x3 = std::make_shared<Var>();
  VectorRef sparse_softmax_cross_entropy_with_logits_grad({prim::kPrimSparseSoftmaxCrossEntropyWithLogits, x1, x2});
  VectorRef sparse_softmax_cross_entropy_with_logits({prim::kPrimSparseSoftmaxCrossEntropyWithLogits, x1, x2});
  VectorRef depend(
    {prim::kPrimDepend, sparse_softmax_cross_entropy_with_logits_grad, sparse_softmax_cross_entropy_with_logits});
  return VectorRef({prim::kPrimMul, depend, x3});
}

const AnfNodePtr GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR::Process(const FuncGraphPtr &graph,
                                                                             const AnfNodePtr &node,
                                                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto mul_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul_node);
  CheckCNodeInputSize(mul_node, kMulInputTensorNum);

  auto depend_node = GetDependNode(mul_node);
  auto sparse_softmax_node = GetSparseNode(depend_node, 2);
  auto sparse_softmax_node_grad = GetSparseNode(depend_node, 1);
  CheckCNodeInputSize(sparse_softmax_node_grad, kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum);

  CNodePtr softmax_node;
  auto one_hot_node = CreateOneHot(graph, sparse_softmax_node_grad);
  softmax_node = CreateSoftmaxCrossEntropyWithLogits(graph, sparse_softmax_node_grad, one_hot_node);

  std::vector<AnfNodePtr> softmax_node_outputs;
  CreateMultipleOutputsOfAnfNode(graph, softmax_node, kSoftmaxCrossEntropyWithLogitsOutputNum, &softmax_node_outputs);
  auto reduce_node = CreateReduceMean(graph, sparse_softmax_node_grad, softmax_node_outputs[0]);
  auto tile_node = CreateTile(graph, sparse_softmax_node_grad, mul_node);
  CNodePtr real_div_node;
  if (tile_node == nullptr) {
    real_div_node = CreateRealDiv(graph, sparse_softmax_node_grad, mul_node->input(2));
  } else {
    real_div_node = CreateRealDiv(graph, sparse_softmax_node_grad, tile_node);
  }
  auto expand_dims_node = CreateExpandDims(graph, real_div_node);
  std::vector<AnfNodePtr> new_mul_inputs = {NewValueNode(std::make_shared<Primitive>(kMulOpName)),
                                            softmax_node_outputs[1], expand_dims_node};
  auto new_mul_node = graph->NewCNode(new_mul_inputs);
  MS_EXCEPTION_IF_NULL(new_mul_node);
  new_mul_node->set_scope(mul_node->scope());
  new_mul_node->set_abstract(mul_node->abstract());

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->Replace(sparse_softmax_node, reduce_node);
  manager->Replace(mul_node, new_mul_node);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                    NewValueNode(MakeValue<bool>(true)), NewValueNode(MakeValue<bool>(true))};
  auto new_depend = graph->NewCNode(inputs);
  manager->Replace(sparse_softmax_node_grad, new_depend);
  return new_mul_node;
}

const BaseRef GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  VectorRef sparse_softmax_cross_entropy_with_logits_grad({prim::kPrimSparseSoftmaxCrossEntropyWithLogits, x1, x2});
  VectorRef sparse_softmax_cross_entropy_with_logits({prim::kPrimSparseSoftmaxCrossEntropyWithLogits, x1, x2});
  return VectorRef(
    {prim::kPrimDepend, sparse_softmax_cross_entropy_with_logits_grad, sparse_softmax_cross_entropy_with_logits});
}

const AnfNodePtr GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2::Process(const FuncGraphPtr &graph,
                                                                               const AnfNodePtr &node,
                                                                               const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto depend_node = node->cast<CNodePtr>();
  auto sparse_softmax_node_grad = GetSparseNode(depend_node, 1);
  auto sparse_softmax_node = GetSparseNode(depend_node, 2);

  CNodePtr softmax_node;
  auto one_hot_node = CreateOneHot(graph, sparse_softmax_node_grad);
  softmax_node = CreateSoftmaxCrossEntropyWithLogits(graph, sparse_softmax_node_grad, one_hot_node);

  std::vector<AnfNodePtr> softmax_node_outputs;
  CreateMultipleOutputsOfAnfNode(graph, softmax_node, kSoftmaxCrossEntropyWithLogitsOutputNum, &softmax_node_outputs);
  auto reduce_node = CreateReduceMean(graph, sparse_softmax_node_grad, softmax_node_outputs[0]);
  auto mul_node = CreateMul(graph, sparse_softmax_node_grad, softmax_node_outputs[1]);

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->Replace(sparse_softmax_node, reduce_node);
  return mul_node;
}

const AnfNodePtr PynativeSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR::Process(const FuncGraphPtr &graph,
                                                                                 const AnfNodePtr &node,
                                                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto sparse_softmax_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum);

  if (AnfAlgo::HasNodeAttr(kAttrIsGrad, sparse_softmax_node) &&
      AnfAlgo::GetNodeAttr<bool>(sparse_softmax_node, kAttrIsGrad)) {
    return nullptr;
  }

  CNodePtr softmax_node;
  auto one_hot_node = CreateOneHot(graph, sparse_softmax_node, true);
  softmax_node = CreateSoftmaxCrossEntropyWithLogits(graph, sparse_softmax_node, one_hot_node);

  std::vector<AnfNodePtr> softmax_node_outputs;
  CreateMultipleOutputsOfAnfNode(graph, softmax_node, kSoftmaxCrossEntropyWithLogitsOutputNum, &softmax_node_outputs);
  auto reduce_node = CreateReduceMean(graph, sparse_softmax_node, softmax_node_outputs[0], true);
  return reduce_node;
}

const BaseRef PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  VarPtr x3 = std::make_shared<Var>();
  VectorRef sparse_softmax_cross_entropy_with_logits({prim::kPrimSparseSoftmaxCrossEntropyWithLogits, x1, x2});
  return VectorRef({prim::kPrimMul, sparse_softmax_cross_entropy_with_logits, x3});
}

const AnfNodePtr PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR::Process(const FuncGraphPtr &graph,
                                                                                     const AnfNodePtr &node,
                                                                                     const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto mul_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul_node);
  CheckCNodeInputSize(mul_node, kMulInputTensorNum);

  auto sparse_softmax_node = mul_node->input(1);
  auto sparse_softmax_node_grad = sparse_softmax_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sparse_softmax_node_grad);
  CheckCNodeInputSize(sparse_softmax_node_grad, kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum);

  CNodePtr softmax_node;
  auto one_hot_node = CreateOneHot(graph, sparse_softmax_node_grad);
  softmax_node = CreateSoftmaxCrossEntropyWithLogits(graph, sparse_softmax_node_grad, one_hot_node);

  std::vector<AnfNodePtr> softmax_node_outputs;
  CreateMultipleOutputsOfAnfNode(graph, softmax_node, kSoftmaxCrossEntropyWithLogitsOutputNum, &softmax_node_outputs);
  auto tile_node = CreateTile(graph, sparse_softmax_node_grad, mul_node);
  CNodePtr real_div_node;
  if (tile_node == nullptr) {
    real_div_node = CreateRealDiv(graph, sparse_softmax_node_grad, mul_node->input(2));
  } else {
    real_div_node = CreateRealDiv(graph, sparse_softmax_node_grad, tile_node);
  }
  auto expand_dims_node = CreateExpandDimsPynative(graph, real_div_node);
  std::vector<AnfNodePtr> new_mul_inputs = {NewValueNode(std::make_shared<Primitive>(kMulOpName)),
                                            softmax_node_outputs[1], expand_dims_node};
  auto new_mul_node = graph->NewCNode(new_mul_inputs);
  MS_EXCEPTION_IF_NULL(new_mul_node);
  new_mul_node->set_scope(mul_node->scope());
  new_mul_node->set_abstract(mul_node->abstract());
  return new_mul_node;
}
}  // namespace opt
}  // namespace mindspore
