/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/mindir/sparse_softmax_cross_entropy_with_logits_unify_mindir.h"
#include <vector>
#include <string>
#include <algorithm>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/helper.h"
#include "include/common/utils/utils.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/dtype/type_id.h"
#include "ir/dtype/type.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kCastInputNum = 2;
constexpr size_t kDependInputNum = 2;
constexpr auto softmax_output_shape_size = 2;
constexpr auto kAttrDepth = "depth";
constexpr auto kAttrMultiples = "multiples";
constexpr auto kIsFeatureMapInputList = "IsFeatureMapInputList";

bool CheckMulInputShapeEqual(const CNodePtr &mul_node) {
  MS_EXCEPTION_IF_NULL(mul_node);
  if (!IsPrimitiveCNode(mul_node, prim::kPrimMul)) {
    MS_LOG(EXCEPTION) << "Node is not mul, but is " << mul_node->fullname_with_scope();
  }
  auto input1_shape = common::AnfAlgo::GetOutputInferShape(mul_node->input(kIndex1), 0);
  auto input2_shape = common::AnfAlgo::GetOutputInferShape(mul_node->input(kIndex2), 0);
  return input1_shape == input2_shape;
}

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
  builder1.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), new_node.get());
  return new_node;
}

CNodePtr CreateOneHot(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node, const PatternProcessPass &pass,
                      bool is_convert_const_to_attr = false) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);

  auto logits_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 0UL);
  int64_t depth = 0;
  if (!logits_shape.empty()) {
    size_t index = logits_shape.size() - 1;
    depth = logits_shape[index];
  } else {
    MS_LOG(EXCEPTION) << "Logits's shape of node [" << sparse_softmax_node->DebugString() << "] is empty"
                      << trace::DumpSourceLines(sparse_softmax_node);
  }

  auto value_on = std::make_shared<tensor::Tensor>(1.0, kFloat32);
  auto value_on_node = CreateValueNode(value_on, kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(value_on_node);
  auto value_off = std::make_shared<tensor::Tensor>(0.0, kFloat32);
  auto value_off_node = CreateValueNode(value_off, kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(value_off_node);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->AddValueNodeToGraph(value_on_node);
  kernel_graph->AddValueNodeToGraph(value_off_node);

  auto one_hot_primitive = std::make_shared<Primitive>(kOneHotOpName);
  std::vector<std::string> input_names = {"indices", "depth", "on_value", "off_value"};
  std::vector<std::string> output_names = {"output"};
  one_hot_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  one_hot_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));
  one_hot_primitive->set_attr(kAttrAxis, MakeValue(static_cast<int64_t>(-1)));

  std::vector<AnfNodePtr> one_hot_inputs;
  if (is_convert_const_to_attr) {
    one_hot_inputs = {NewValueNode(one_hot_primitive), sparse_softmax_node->input(kIndex2), value_on_node,
                      value_off_node};
  } else {
    auto depth_node = NewValueNode(depth);
    MS_EXCEPTION_IF_NULL(depth_node);
    auto depth_abstract = std::make_shared<abstract::AbstractScalar>();
    MS_EXCEPTION_IF_NULL(depth_abstract);
    depth_abstract->set_type(kInt64);
    depth_node->set_abstract(depth_abstract);
    one_hot_inputs = {NewValueNode(one_hot_primitive), sparse_softmax_node->input(kIndex2), depth_node, value_on_node,
                      value_off_node};
  }
  auto one_hot_node = pass.NewCNode(one_hot_inputs, graph);
  MS_EXCEPTION_IF_NULL(one_hot_node);
  one_hot_node->set_scope(sparse_softmax_node->scope());
  auto labels_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 1UL);
  labels_shape.emplace_back(depth);
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {labels_shape}, one_hot_node.get());
  if (is_convert_const_to_attr) {
    common::AnfAlgo::SetNodeAttr(kAttrDepth, MakeValue(depth), one_hot_node);
  }
  return one_hot_node;
}

CNodePtr CreateSoftmaxCrossEntropyWithLogits(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node,
                                             const CNodePtr &one_hot_node, const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(one_hot_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(kSoftmaxCrossEntropyWithLogitsOpName)),
                                    sparse_softmax_node->input(kIndex1), one_hot_node};
  auto softmax_node = pass.NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(softmax_node);
  softmax_node->set_scope(sparse_softmax_node->scope());

  auto labels_shape = common::AnfAlgo::GetOutputInferShape(one_hot_node, 0);
  ShapeVector loss_shape;
  if (!labels_shape.empty()) {
    loss_shape.emplace_back(labels_shape[0]);
  } else {
    MS_LOG(EXCEPTION) << "One_hot output's shape is empty." << trace::DumpSourceLines(one_hot_node);
  }

  auto data_types = common::AnfAlgo::GetOutputInferDataType(one_hot_node, 0UL);
  auto types = {data_types, data_types};
  auto shapes = {loss_shape, labels_shape};
  common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, softmax_node.get());
  return softmax_node;
}

std::vector<int64_t> GetAxis(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto output_shape = common::AnfAlgo::GetOutputInferShape(node, 0);
  if (output_shape.empty()) {
    MS_LOG(EXCEPTION) << node->fullname_with_scope() << "'s output shape is empty" << trace::DumpSourceLines(node);
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
                          const AnfNodePtr &softmax_output_node, const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(softmax_output_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum);
  auto axis_node = GetAxisNode(softmax_output_node);
  MS_EXCEPTION_IF_NULL(axis_node);

  auto reduce_primitive = std::make_shared<Primitive>(kReduceMeanOpName);
  std::vector<std::string> input_names = {"x", "axis"};
  std::vector<std::string> output_names = {"y"};
  reduce_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  reduce_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));
  reduce_primitive->set_attr(kAttrKeepDims, MakeValue(false));

  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<AnfNodePtr> inputs;
  kernel_graph->AddValueNodeToGraph(axis_node);
  inputs = {NewValueNode(reduce_primitive), softmax_output_node, axis_node};
  auto reduce_node = pass.NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(reduce_node);
  reduce_node->set_scope(sparse_softmax_node->scope());
  auto reduce_abstract = softmax_output_node->abstract();
  reduce_abstract->set_shape(std::make_shared<abstract::Shape>());
  reduce_node->set_abstract(reduce_abstract);
  return reduce_node;
}

void UpdateAbstract(const CNodePtr &real_div_node, const CNodePtr &expand_dims_node) {
  MS_EXCEPTION_IF_NULL(real_div_node);
  MS_EXCEPTION_IF_NULL(expand_dims_node);
  auto y_shape = common::AnfAlgo::GetOutputInferShape(real_div_node, 0UL);
  (void)y_shape.emplace_back(1);
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(real_div_node, 0UL)}, {y_shape},
                                              expand_dims_node.get());
}

CNodePtr CreateExpandDims(const FuncGraphPtr &graph, const CNodePtr &real_div_node, const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(real_div_node);
  CheckCNodeInputSize(real_div_node, kRealDivInputTensorNum);

  int64_t axis = -1;
  auto axis_node = NewValueNode(axis);
  MS_EXCEPTION_IF_NULL(axis_node);
  auto axis_abstract = std::make_shared<abstract::AbstractScalar>();
  MS_EXCEPTION_IF_NULL(axis_abstract);
  axis_abstract->set_type(kInt64);
  axis_node->set_abstract(axis_abstract);

  auto expand_dims_primitive = std::make_shared<Primitive>(kExpandDimsOpName);
  std::vector<std::string> input_names = {"x", "axis"};
  std::vector<std::string> output_names = {"output"};
  expand_dims_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  expand_dims_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));
  std::vector<AnfNodePtr> expand_dims_inputs = {NewValueNode(expand_dims_primitive), real_div_node, axis_node};
  auto expand_dims_node = pass.NewCNode(expand_dims_inputs, graph);
  MS_EXCEPTION_IF_NULL(expand_dims_node);

  expand_dims_node->set_scope(real_div_node->scope());
  UpdateAbstract(real_div_node, expand_dims_node);
  return expand_dims_node;
}

CNodePtr CreateExpandDimsPynative(const FuncGraphPtr &graph, const CNodePtr &real_div_node,
                                  const PatternProcessPass &pass) {
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
  auto expand_dims_node = pass.NewCNode(expand_dims_inputs, graph);
  MS_EXCEPTION_IF_NULL(expand_dims_node);

  expand_dims_node->set_scope(real_div_node->scope());
  UpdateAbstract(real_div_node, expand_dims_node);
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(axis), expand_dims_node);
  return expand_dims_node;
}

CNodePtr CreateTile(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node, const CNodePtr &mul_node,
                    const PatternProcessPass &pass, bool is_convert_const_to_attr = false) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(mul_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum);
  CheckCNodeInputSize(mul_node, kMulInputTensorNum);

  auto multiple_value = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 1);
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

  auto tile_node = pass.NewCNode(tile_inputs, graph);
  MS_EXCEPTION_IF_NULL(tile_node);
  tile_node->set_scope(mul_node->scope());
  common::AnfAlgo::SetOutputTypeAndDetailShape({common::AnfAlgo::GetPrevNodeOutputInferDataType(mul_node, 1UL)},
                                               {AnfAlgo::GetPrevNodeOutputDetailShape(sparse_softmax_node, 1UL)},
                                               tile_node.get());
  if (is_convert_const_to_attr) {
    common::AnfAlgo::SetNodeAttr(kAttrMultiples, MakeValue(multiples), tile_node);
  }
  // feature map set
  std::vector<size_t> feature_map_input_indexs;
  (void)feature_map_input_indexs.emplace_back(0);
  common::AnfAlgo::SetNodeAttr(kIsFeatureMapInputList, MakeValue(feature_map_input_indexs), tile_node);
  return tile_node;
}

CNodePtr CreateRealDiv(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node, const AnfNodePtr &tile_node,
                       const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(tile_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum);
  auto labels_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 1UL);
  if (labels_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "Label's shape should be 1-D, but got " << labels_shape.size()
                      << trace::DumpSourceLines(sparse_softmax_node);
  }
  auto y_value = static_cast<float>(labels_shape[0]);
  auto y = std::make_shared<tensor::Tensor>(y_value, kFloat32);
  auto y_node = CreateValueNode(y, kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(y_node);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->AddValueNodeToGraph(y_node);

  auto real_div_primitive = std::make_shared<Primitive>(kRealDivOpName);
  std::vector<std::string> input_names = {"x", "y"};
  std::vector<std::string> output_names = {"output"};
  real_div_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  real_div_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));
  std::vector<AnfNodePtr> real_div_inputs = {NewValueNode(real_div_primitive), tile_node, y_node};
  auto real_div_node = pass.NewCNode(real_div_inputs, graph);
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
  auto depend_node = mul_node->input(kIndex1);
  MS_EXCEPTION_IF_NULL(depend_node);
  return depend_node->cast<CNodePtr>();
}

CNodePtr CreateMul(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node,
                   const AnfNodePtr &softmax_output_node, const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(softmax_output_node);
  auto softmax_output_shape = common::AnfAlgo::GetOutputInferShape(softmax_output_node, 0UL);
  if (softmax_output_shape.size() != softmax_output_shape_size) {
    MS_LOG(EXCEPTION) << "SoftmaxCrossEntropyWithLogits the second output shape size should be "
                      << softmax_output_shape_size << ", but got " << softmax_output_shape.size()
                      << trace::DumpSourceLines(softmax_output_node);
  }
  ShapeVector tensor_shape;
  tensor_shape.emplace_back(softmax_output_shape[0]);
  tensor_shape.emplace_back(1);
  if (softmax_output_shape[0] == 0) {
    MS_LOG(EXCEPTION) << "output_shape[0] of softmax should not be 0" << trace::DumpSourceLines(softmax_output_node);
  }
  std::vector<float> tensor_value(softmax_output_shape[0], 1.0 / softmax_output_shape[0]);
  auto buf_size = sizeof(float) * tensor_value.size();
  auto tensor_y = std::make_shared<tensor::Tensor>(kNumberTypeFloat32, tensor_shape, tensor_value.data(), buf_size);
  auto y_node = CreateValueNode(tensor_y, kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(y_node);

  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->AddValueNodeToGraph(y_node);

  auto mul_primitive = std::make_shared<Primitive>(kMulOpName);
  std::vector<std::string> input_names = {"x", "y"};
  std::vector<std::string> output_names = {"output"};
  mul_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  mul_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));

  std::vector<AnfNodePtr> mul_input = {NewValueNode(mul_primitive), softmax_output_node, y_node};
  auto mul_node = pass.NewCNode(mul_input, graph);
  MS_EXCEPTION_IF_NULL(mul_node);

  mul_node->set_scope(sparse_softmax_node->scope());
  mul_node->set_abstract(softmax_output_node->abstract());
  return mul_node;
}

CNodePtr CreateCast(const FuncGraphPtr &graph, const CNodePtr &cast, const AnfNodePtr &cast_input,
                    const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(cast);
  MS_EXCEPTION_IF_NULL(cast_input);

  std::vector<AnfNodePtr> new_cast_inputs = {cast->input(kAnfPrimitiveIndex), cast_input, cast->input(kIndex2)};
  auto new_cast_node = pass.NewCNode(new_cast_inputs, graph);
  MS_EXCEPTION_IF_NULL(new_cast_node);
  new_cast_node->set_scope(cast->scope());
  new_cast_node->set_abstract(cast->abstract());
  return new_cast_node;
}

CNodePtr CreateDepend(const FuncGraphPtr &graph, const CNodePtr &depend, const AnfNodePtr &depend_input,
                      const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(depend);
  MS_EXCEPTION_IF_NULL(depend_input);

  std::vector<AnfNodePtr> new_depend_inputs = {depend->input(kAnfPrimitiveIndex), depend_input, depend->input(kIndex2)};
  auto new_depend_node = pass.NewCNode(new_depend_inputs, graph);
  MS_EXCEPTION_IF_NULL(new_depend_node);
  new_depend_node->set_scope(depend->scope());
  new_depend_node->set_abstract(depend->abstract());
  return new_depend_node;
}

bool IsSparseSoftmaxCrossEntropyWithLogitsGrad(const CNodePtr &sparse, const string &pass_name) {
  MS_EXCEPTION_IF_NULL(sparse);
  if (common::AnfAlgo::GetCNodeName(sparse) != kSparseSoftmaxCrossEntropyWithLogitsOpName) {
    MS_LOG(EXCEPTION) << "The pass of " << pass_name << "'s input node should be "
                      << kSparseSoftmaxCrossEntropyWithLogitsOpName << ", but got "
                      << common::AnfAlgo::GetCNodeName(sparse) << trace::DumpSourceLines(sparse);
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrIsGrad, sparse)) {
    return common::AnfAlgo::GetNodeAttr<bool>(sparse, kAttrIsGrad);
  } else {
    MS_LOG(EXCEPTION) << "Node of " << sparse->fullname_with_scope() << " does not have the attr " << kAttrIsGrad
                      << ", related pass: " << pass_name << trace::DumpSourceLines(sparse);
  }
}

CNodePtr CreateMulInput(const FuncGraphPtr &graph, const CNodePtr &mul_node, const AnfNodePtr &sparse_softmax_node,
                        const std::string &pass_name, const PatternProcessPass &pass,
                        std::vector<AnfNodePtr> *softmax_node_outputs, bool *is_sp_grad_flag) {
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(softmax_node_outputs);
  MS_EXCEPTION_IF_NULL(is_sp_grad_flag);

  auto sparse_softmax_node_grad = sparse_softmax_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sparse_softmax_node_grad);
  CheckCNodeInputSize(sparse_softmax_node_grad, kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum);

  if (!IsSparseSoftmaxCrossEntropyWithLogitsGrad(sparse_softmax_node_grad, pass_name)) {
    *is_sp_grad_flag = false;
    return nullptr;
  }

  CNodePtr softmax_node;
  auto one_hot_node = CreateOneHot(graph, sparse_softmax_node_grad, pass);
  softmax_node = CreateSoftmaxCrossEntropyWithLogits(graph, sparse_softmax_node_grad, one_hot_node, pass);

  CreateMultipleOutputsOfAnfNode(graph, softmax_node, kSoftmaxCrossEntropyWithLogitsOutputNum, softmax_node_outputs);
  auto tile_node = CreateTile(graph, sparse_softmax_node_grad, mul_node, pass);
  CNodePtr real_div_node;
  if (tile_node == nullptr) {
    real_div_node = CreateRealDiv(graph, sparse_softmax_node_grad, mul_node->input(kIndex2), pass);
  } else {
    real_div_node = CreateRealDiv(graph, sparse_softmax_node_grad, tile_node, pass);
  }
  auto expand_dims_node = CreateExpandDimsPynative(graph, real_div_node, pass);
  return expand_dims_node;
}
}  // namespace

std::vector<std::string> SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimSparseSoftmaxCrossEntropyWithLogits->name());
  return ret;
}

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

  if (IsSparseSoftmaxCrossEntropyWithLogitsGrad(sparse_softmax_node, name())) {
    return nullptr;
  }

  CNodePtr softmax_node;
  auto one_hot_node = CreateOneHot(graph, sparse_softmax_node, *this);
  softmax_node = CreateSoftmaxCrossEntropyWithLogits(graph, sparse_softmax_node, one_hot_node, *this);

  std::vector<AnfNodePtr> softmax_node_outputs;
  CreateMultipleOutputsOfAnfNode(graph, softmax_node, kSoftmaxCrossEntropyWithLogitsOutputNum, &softmax_node_outputs);
  auto reduce_node = CreateReduceMean(graph, sparse_softmax_node, softmax_node_outputs[0], *this);
  return reduce_node;
}

const BaseRef GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  VarPtr x3 = std::make_shared<Var>();
  VarPtr x4 = std::make_shared<Var>();
  VectorRef sparse_softmax_cross_entropy_with_logits_grad({prim::kPrimSparseSoftmaxCrossEntropyWithLogits, x1, x2});
  VectorRef sparse_softmax_cross_entropy_with_logits({prim::kPrimSparseSoftmaxCrossEntropyWithLogits, x1, x4});
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

  if (CheckMulInputShapeEqual(mul_node)) {
    return nullptr;
  }

  auto depend_node = GetDependNode(mul_node);
  auto sparse_softmax_node = GetSparseNode(depend_node, kIndex2);
  auto sparse_softmax_node_grad = GetSparseNode(depend_node, 1);
  CheckCNodeInputSize(sparse_softmax_node_grad, kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum);

  CNodePtr softmax_node;
  auto one_hot_node = CreateOneHot(graph, sparse_softmax_node_grad, *this);
  softmax_node = CreateSoftmaxCrossEntropyWithLogits(graph, sparse_softmax_node_grad, one_hot_node, *this);

  std::vector<AnfNodePtr> softmax_node_outputs;
  CreateMultipleOutputsOfAnfNode(graph, softmax_node, kSoftmaxCrossEntropyWithLogitsOutputNum, &softmax_node_outputs);
  auto reduce_node = CreateReduceMean(graph, sparse_softmax_node_grad, softmax_node_outputs[0], *this);
  auto tile_node = CreateTile(graph, sparse_softmax_node_grad, mul_node, *this);
  CNodePtr real_div_node;
  if (tile_node == nullptr) {
    real_div_node = CreateRealDiv(graph, sparse_softmax_node_grad, mul_node->input(kIndex2), *this);
  } else {
    real_div_node = CreateRealDiv(graph, sparse_softmax_node_grad, tile_node, *this);
  }
  auto expand_dims_node = CreateExpandDims(graph, real_div_node, *this);
  auto mul_primitive = common::AnfAlgo::GetCNodePrimitive(mul_node);
  MS_EXCEPTION_IF_NULL(mul_primitive);
  std::vector<AnfNodePtr> new_mul_inputs = {NewValueNode(mul_primitive), softmax_node_outputs[1], expand_dims_node};
  auto new_mul_node = NewCNode(new_mul_inputs, graph);
  MS_EXCEPTION_IF_NULL(new_mul_node);
  new_mul_node->set_scope(mul_node->scope());
  new_mul_node->set_abstract(mul_node->abstract());

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(sparse_softmax_node, reduce_node);
  (void)manager->Replace(mul_node, new_mul_node);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                    NewValueNode(MakeValue<bool>(true)), NewValueNode(MakeValue<bool>(true))};
  auto new_depend = graph->NewCNode(inputs);
  (void)manager->Replace(sparse_softmax_node_grad, new_depend);
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
  auto sparse_softmax_node_grad = GetSparseNode(depend_node, 1UL);
  auto sparse_softmax_node = GetSparseNode(depend_node, kIndex2);

  CNodePtr softmax_node;
  auto one_hot_node = CreateOneHot(graph, sparse_softmax_node_grad, *this);
  softmax_node = CreateSoftmaxCrossEntropyWithLogits(graph, sparse_softmax_node_grad, one_hot_node, *this);

  std::vector<AnfNodePtr> softmax_node_outputs;
  CreateMultipleOutputsOfAnfNode(graph, softmax_node, kSoftmaxCrossEntropyWithLogitsOutputNum, &softmax_node_outputs);
  auto reduce_node = CreateReduceMean(graph, sparse_softmax_node_grad, softmax_node_outputs[0], *this);
  auto mul_node = CreateMul(graph, sparse_softmax_node_grad, softmax_node_outputs[1], *this);

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(sparse_softmax_node, reduce_node);
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

  if (IsSparseSoftmaxCrossEntropyWithLogitsGrad(sparse_softmax_node, name())) {
    return nullptr;
  }

  CNodePtr softmax_node;
  auto one_hot_node = CreateOneHot(graph, sparse_softmax_node, *this);
  softmax_node = CreateSoftmaxCrossEntropyWithLogits(graph, sparse_softmax_node, one_hot_node, *this);

  std::vector<AnfNodePtr> softmax_node_outputs;
  CreateMultipleOutputsOfAnfNode(graph, softmax_node, kSoftmaxCrossEntropyWithLogitsOutputNum, &softmax_node_outputs);
  // Both of the forward loss function and the backward loss function from cangjie will match this pattern,
  // the true branch is for the backward loss function, and the false branch is for the other one.
  if (common::AnfAlgo::HasNodeAttr(kAttrIsGrad, sparse_softmax_node) &&
      common::AnfAlgo::GetNodeAttr<bool>(sparse_softmax_node, kAttrIsGrad)) {
    return softmax_node_outputs[1];
  } else {
    auto reduce_node = CreateReduceMean(graph, sparse_softmax_node, softmax_node_outputs[0], *this);
    return reduce_node;
  }
}

std::vector<std::string> PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimSparseSoftmaxCrossEntropyWithLogits->name());
  ret.emplace_back(prim::kPrimMul->name());
  return ret;
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

  auto sparse_softmax_node = mul_node->input(kIndex1);
  bool is_sp_grad_flag = true;
  std::vector<AnfNodePtr> softmax_node_outputs;
  auto expand_dims_node =
    CreateMulInput(graph, mul_node, sparse_softmax_node, name(), *this, &softmax_node_outputs, &is_sp_grad_flag);
  if (!is_sp_grad_flag) {
    return nullptr;
  }
  std::vector<AnfNodePtr> new_mul_inputs = {NewValueNode(std::make_shared<Primitive>(kMulOpName)),
                                            softmax_node_outputs[1], expand_dims_node};
  auto new_mul_node = NewCNode(new_mul_inputs, graph);
  MS_EXCEPTION_IF_NULL(new_mul_node);
  new_mul_node->set_scope(mul_node->scope());
  new_mul_node->set_abstract(mul_node->abstract());
  return new_mul_node;
}

std::vector<std::string> PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimSparseSoftmaxCrossEntropyWithLogits->name());
  ret.emplace_back(prim::kPrimCast->name());
  ret.emplace_back(prim::kPrimMul->name());
  return ret;
}

const BaseRef PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  VarPtr x3 = std::make_shared<Var>();
  VarPtr x4 = std::make_shared<Var>();
  VectorRef sparse_softmax_cross_entropy_with_logits({prim::kPrimSparseSoftmaxCrossEntropyWithLogits, x1, x2});
  VectorRef cast({prim::kPrimCast, sparse_softmax_cross_entropy_with_logits, x4});
  return VectorRef({prim::kPrimMul, cast, x3});
}

const AnfNodePtr PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2::Process(const FuncGraphPtr &graph,
                                                                                       const AnfNodePtr &node,
                                                                                       const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto mul_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul_node);
  CheckCNodeInputSize(mul_node, kMulInputTensorNum);

  auto cast_node = mul_node->input(kIndex1);
  auto cast_cnode = cast_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cast_cnode);
  CheckCNodeInputSize(cast_cnode, kCastInputNum);

  auto sparse_softmax_node = cast_cnode->input(kIndex1);
  bool is_sp_grad_flag = true;
  std::vector<AnfNodePtr> softmax_node_outputs;
  auto expand_dims_node =
    CreateMulInput(graph, mul_node, sparse_softmax_node, name(), *this, &softmax_node_outputs, &is_sp_grad_flag);
  if (!is_sp_grad_flag) {
    return nullptr;
  }
  auto new_cast = CreateCast(graph, cast_cnode, softmax_node_outputs[1], *this);
  std::vector<AnfNodePtr> new_mul_inputs = {NewValueNode(std::make_shared<Primitive>(kMulOpName)), new_cast,
                                            expand_dims_node};
  auto new_mul_node = NewCNode(new_mul_inputs, graph);
  MS_EXCEPTION_IF_NULL(new_mul_node);
  new_mul_node->set_scope(mul_node->scope());
  new_mul_node->set_abstract(mul_node->abstract());
  return new_mul_node;
}

const BaseRef PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV3::DefinePattern() const {
  VarPtr x1 = std::make_shared<Var>();
  VarPtr x2 = std::make_shared<Var>();
  VarPtr x3 = std::make_shared<Var>();
  VarPtr x4 = std::make_shared<Var>();
  VectorRef sparse_softmax_cross_entropy_with_logits({prim::kPrimSparseSoftmaxCrossEntropyWithLogits, x1, x2});
  VectorRef depend({prim::kPrimDepend, sparse_softmax_cross_entropy_with_logits, x3});
  return VectorRef({prim::kPrimMul, depend, x4});
}

const AnfNodePtr PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV3::Process(const FuncGraphPtr &graph,
                                                                                       const AnfNodePtr &node,
                                                                                       const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto mul_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul_node);
  CheckCNodeInputSize(mul_node, kMulInputTensorNum);

  auto depend_node = mul_node->input(kIndex1);
  auto depend_cnode = depend_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(depend_cnode);
  CheckCNodeInputSize(depend_cnode, kDependInputNum);

  auto sparse_softmax_node = depend_cnode->input(kIndex1);
  bool is_sp_grad_flag = true;
  std::vector<AnfNodePtr> softmax_node_outputs;
  auto expand_dims_node =
    CreateMulInput(graph, mul_node, sparse_softmax_node, name(), *this, &softmax_node_outputs, &is_sp_grad_flag);
  if (!is_sp_grad_flag) {
    return nullptr;
  }
  auto new_depend = CreateDepend(graph, depend_cnode, softmax_node_outputs[1], *this);
  std::vector<AnfNodePtr> new_mul_inputs = {NewValueNode(std::make_shared<Primitive>(kMulOpName)), new_depend,
                                            expand_dims_node};
  auto new_mul_node = NewCNode(new_mul_inputs, graph);
  MS_EXCEPTION_IF_NULL(new_mul_node);
  new_mul_node->set_scope(mul_node->scope());
  new_mul_node->set_abstract(mul_node->abstract());
  return new_mul_node;
}
}  // namespace opt
}  // namespace mindspore
