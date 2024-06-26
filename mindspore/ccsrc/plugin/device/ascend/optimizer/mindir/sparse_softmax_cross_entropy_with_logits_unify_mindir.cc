/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include <functional>
#include "ops/nn_ops.h"
#include "ops/math_ops.h"
#include "ops/array_ops.h"
#include "ops/framework_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/utils.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/dtype/type_id.h"
#include "ir/dtype/type.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kCastInputNum = 2;
constexpr auto softmax_output_shape_size = 2;
constexpr auto kAttrDepth = "depth";
constexpr auto kAttrMultiples = "multiples";
constexpr auto kIsFeatureMapInputList = "IsFeatureMapInputList";
constexpr auto kShapeFromTensor = "shape_from_tensor";
constexpr size_t kSparseSoftmaxCrossEntropyWithLogitsInputNum = 3;
constexpr size_t kSparseSoftmaxCrossEntropyWithLogitsOutputNum = 2;

bool CheckMulInputShapeEqual(const CNodePtr &mul_node) {
  MS_EXCEPTION_IF_NULL(mul_node);
  if (!IsPrimitiveCNode(mul_node, prim::kPrimMul)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Node is not mul, but is " << mul_node->fullname_with_scope();
  }
  auto input1_shape = common::AnfAlgo::GetOutputInferShape(mul_node->input(kIndex1), 0);
  auto input2_shape = common::AnfAlgo::GetOutputInferShape(mul_node->input(kIndex2), 0);
  return input1_shape == input2_shape;
}

ValueNodePtr CreateValueNode(const ValuePtr &value_ptr, TypeId output_type, bool is_scalar = false) {
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
  if (is_scalar) {
    builder1.SetOutputsKernelObjectType({kernel::KernelObjectType::SCALAR});
  } else {
    builder1.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  }
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), new_node.get());
  return new_node;
}

CNodePtr CreateReshape(const FuncGraphPtr &graph, const AnfNodePtr &input_node, const ShapeVector &shape,
                       const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input_node);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  auto reshape_primitive = std::make_shared<Primitive>(kReshapeOpName);
  std::vector<std::string> input_names = {"x", "shape"};
  std::vector<std::string> output_names = {"output"};
  reshape_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  reshape_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));

  auto shape_node = CreateTensorInput(kernel_graph, NewValueNode(shape));
  std::vector<AnfNodePtr> reshape_inputs = {NewValueNode(reshape_primitive), input_node, shape_node};
  auto reshape_node = pass.NewCNode(reshape_inputs, graph);
  MS_EXCEPTION_IF_NULL(reshape_node);
  auto data_types = common::AnfAlgo::GetOutputInferDataType(input_node, 0UL);
  common::AnfAlgo::SetOutputInferTypeAndShape({data_types}, {shape}, reshape_node.get());
  common::AnfAlgo::SetNodeAttr(kShapeFromTensor, MakeValue(true), reshape_node);
  reshape_node->set_scope(input_node->scope());
  return reshape_node;
}
void GetDepthAndBatchSizeFromSparseSoftmaxNode(const AnfNodePtr &sparse_softmax_node, int64_t *batch_size,
                                               int64_t *depth) {
  auto logits_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 0UL);
  auto labels_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 1UL);
  *depth = 0;
  if (!logits_shape.empty()) {
    size_t index = logits_shape.size() - 1;
    *depth = logits_shape[index];
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Logits's shape of node [" << sparse_softmax_node->DebugString() << "] is empty"
                               << trace::DumpSourceLines(sparse_softmax_node);
  }
  *batch_size = std::accumulate(labels_shape.begin(), labels_shape.end(), 1, std::multiplies<int64_t>());
}
CNodePtr CreateOneHot(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node, const PatternProcessPass &pass,
                      bool is_convert_const_to_attr = false) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);

  int64_t batch_size;
  int64_t depth;
  GetDepthAndBatchSizeFromSparseSoftmaxNode(sparse_softmax_node, &batch_size, &depth);
  auto is_dynamic = common::AnfAlgo::IsDynamicShape(sparse_softmax_node);
  ShapeVector shape = is_dynamic ? ShapeVector{-1} : ShapeVector{batch_size};

  // Reshape multi-dim labels to 1D labels.
  auto reshape_node = CreateReshape(graph, sparse_softmax_node->input(kIndex2), shape, pass);

  auto value_on = std::make_shared<tensor::Tensor>(1.0, kFloat32);
  auto value_on_node = CreateValueNode(value_on, kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(value_on_node);
  auto value_off = std::make_shared<tensor::Tensor>(0.0, kFloat32);
  auto value_off_node = CreateValueNode(value_off, kNumberTypeFloat32);
  MS_EXCEPTION_IF_NULL(value_off_node);
  auto value_axis_node = CreateValueNode(MakeValue<int64_t>(-1), kNumberTypeInt64, true);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->AddValueNodeToGraph(value_on_node);
  kernel_graph->AddValueNodeToGraph(value_off_node);
  kernel_graph->AddValueNodeToGraph(value_axis_node);

  auto one_hot_primitive = std::make_shared<Primitive>(kOneHotOpName);
  std::vector<std::string> input_names = {"indices", "depth", "on_value", "off_value", "axis"};
  std::vector<std::string> output_names = {"output"};
  one_hot_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  one_hot_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));

  std::vector<AnfNodePtr> one_hot_inputs;
  if (is_convert_const_to_attr) {
    one_hot_inputs = {NewValueNode(one_hot_primitive), reshape_node, value_on_node, value_off_node};
  } else {
    auto depth_node = CreateTensorInput(kernel_graph, NewValueNode(depth));
    MS_EXCEPTION_IF_NULL(depth_node);
    one_hot_inputs = {
      NewValueNode(one_hot_primitive), reshape_node, depth_node, value_on_node, value_off_node, value_axis_node};
  }
  auto one_hot_node = pass.NewCNode(one_hot_inputs, graph);
  MS_EXCEPTION_IF_NULL(one_hot_node);
  one_hot_node->set_scope(sparse_softmax_node->scope());
  ShapeVector one_hot_shape = {batch_size, depth};
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {one_hot_shape}, one_hot_node.get());
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

  int64_t batch_size;
  int64_t depth;
  GetDepthAndBatchSizeFromSparseSoftmaxNode(sparse_softmax_node, &batch_size, &depth);
  auto is_dynamic = common::AnfAlgo::IsDynamicShape(sparse_softmax_node);
  ShapeVector shape = is_dynamic ? ShapeVector{-1, depth} : ShapeVector{batch_size, depth};

  // Reshape multi-dim logits to 2D logits.
  auto reshape_node = CreateReshape(graph, sparse_softmax_node->input(kIndex1), shape, pass);

  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(kSoftmaxCrossEntropyWithLogitsOpName)),
                                    reshape_node, one_hot_node};
  auto softmax_node = pass.NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(softmax_node);
  softmax_node->set_scope(sparse_softmax_node->scope());

  ShapeVector loss_shape = {batch_size};
  auto data_types = common::AnfAlgo::GetOutputInferDataType(one_hot_node, 0UL);
  auto types = {data_types, data_types};
  auto shapes = {loss_shape, shape};
  common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, softmax_node.get());
  return softmax_node;
}

std::vector<int64_t> GetAxis(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto output_shape = common::AnfAlgo::GetOutputInferShape(node, 0);
  if (output_shape.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << node->fullname_with_scope() << "'s output shape is empty"
                               << trace::DumpSourceLines(node);
  }
  std::vector<int64_t> range;
  for (size_t i = 0; i < output_shape.size(); i++) {
    range.emplace_back(i);
  }
  return range;
}

AnfNodePtr GetAxisNode(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  auto range = GetAxis(node);
  auto axis_node = CreateTensorInput(kernel_graph, CreateValueNode(MakeValue(range), kNumberTypeInt64));
  MS_EXCEPTION_IF_NULL(axis_node);
  return axis_node;
}

AnfNodePtr GetKeepDimsNode(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  auto keep_dims_node = CreateValueNode(MakeValue(false), kNumberTypeBool, true);
  kernel_graph->AddValueNodeToGraph(keep_dims_node);
  return keep_dims_node;
}

CNodePtr CreateReduceMean(const FuncGraphPtr &graph, const CNodePtr &sparse_softmax_node,
                          const AnfNodePtr &softmax_output_node, const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);
  MS_EXCEPTION_IF_NULL(softmax_output_node);
  CheckCNodeInputSize(sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsInputTensorNum);
  auto axis_node = GetAxisNode(graph, softmax_output_node);
  MS_EXCEPTION_IF_NULL(axis_node);
  auto keep_dims_node = GetKeepDimsNode(graph);
  MS_EXCEPTION_IF_NULL(keep_dims_node);

  auto reduce_primitive = std::make_shared<Primitive>(kReduceMeanOpName);
  std::vector<std::string> input_names = {"x", "axis"};
  std::vector<std::string> output_names = {"y"};
  reduce_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  reduce_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));

  std::vector<AnfNodePtr> inputs;
  inputs = {NewValueNode(reduce_primitive), softmax_output_node, axis_node, keep_dims_node};
  auto reduce_node = pass.NewCNode(inputs, graph);
  MS_EXCEPTION_IF_NULL(reduce_node);
  reduce_node->set_scope(sparse_softmax_node->scope());
  auto reduce_abstract = softmax_output_node->abstract();
  if (!softmax_output_node->Shape()->IsDynamic()) {
    reduce_abstract->set_shape(std::make_shared<abstract::Shape>());
  }
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
  auto axis_abstract = std::make_shared<abstract::AbstractScalar>();
  MS_EXCEPTION_IF_NULL(axis_abstract);
  axis_abstract->set_type(kInt64);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto axis_node = kernel_graph->NewValueNode(axis_abstract, MakeValue(axis));
  MS_EXCEPTION_IF_NULL(axis_node);

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

  auto labels_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 1);
  if (std::all_of(labels_shape.begin(), labels_shape.end(), [](int64_t value) { return value == 1; })) {
    return nullptr;
  }
  int64_t batch_size = std::accumulate(labels_shape.begin(), labels_shape.end(), 1, std::multiplies<int64_t>());

  auto tile_primitive = std::make_shared<Primitive>(kTileOpName);
  std::vector<std::string> input_names = {"x", "multiples"};
  std::vector<std::string> output_names = {"output"};
  tile_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
  tile_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));

  std::vector<AnfNodePtr> tile_inputs;
  if (is_convert_const_to_attr) {
    tile_inputs = {NewValueNode(tile_primitive), mul_node->input(kIndex2)};
  } else {
    if (std::any_of(labels_shape.begin(), labels_shape.end(), [](int64_t value) { return value < 0; })) {
      std::vector<AnfNodePtr> dynamic_shape_inputs = {NewValueNode(std::make_shared<Primitive>("TensorShape")),
                                                      sparse_softmax_node->input(kIndex2)};
      auto shape_node = pass.NewCNode(dynamic_shape_inputs, graph);
      MS_EXCEPTION_IF_NULL(shape_node);
      ShapeVector tensor_shp({static_cast<int64_t>(labels_shape.size())});
      auto dynamic_shape_abstract =
        std::make_shared<abstract::AbstractTensor>(kInt64, std::make_shared<abstract::Shape>(tensor_shp));
      MS_EXCEPTION_IF_NULL(dynamic_shape_abstract);
      shape_node->set_abstract(dynamic_shape_abstract);
      tile_inputs = {NewValueNode(tile_primitive), mul_node->input(kIndex2), shape_node};
    } else {
      auto multiples = MakeValue(batch_size);
      std::vector<int64_t> multiples_v = {batch_size};
      auto kernel_graph = graph->cast<KernelGraphPtr>();
      MS_EXCEPTION_IF_NULL(kernel_graph);
      auto multiples_node = CreateTensorInput(kernel_graph, NewValueNode(multiples_v));
      MS_EXCEPTION_IF_NULL(multiples_node);
      tile_inputs = {NewValueNode(tile_primitive), mul_node->input(kIndex2), multiples_node};
    }
  }

  auto tile_node = pass.NewCNode(tile_inputs, graph);
  MS_EXCEPTION_IF_NULL(tile_node);
  tile_node->set_scope(mul_node->scope());
  ShapeVector tile_shape = {batch_size};
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetPrevNodeOutputInferDataType(mul_node, 1UL)},
                                              {tile_shape}, tile_node.get());
  if (is_convert_const_to_attr) {
    common::AnfAlgo::SetNodeAttr(kAttrMultiples, MakeValue(batch_size), tile_node);
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
  int64_t batch_size = std::accumulate(labels_shape.begin(), labels_shape.end(), 1, std::multiplies<int64_t>());
  auto y_value = static_cast<float>(batch_size);
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
    MS_LOG(INTERNAL_EXCEPTION) << "SoftmaxCrossEntropyWithLogits the second output shape size should be "
                               << softmax_output_shape_size << ", but got " << softmax_output_shape.size()
                               << trace::DumpSourceLines(softmax_output_node);
  }
  ShapeVector tensor_shape;
  tensor_shape.emplace_back(softmax_output_shape[0]);
  tensor_shape.emplace_back(1);
  if (softmax_output_shape[0] == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "output_shape[0] of softmax should not be 0"
                               << trace::DumpSourceLines(softmax_output_node);
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

bool IsSparseSoftmaxCrossEntropyWithLogitsGrad(const CNodePtr &sparse, const string &pass_name) {
  MS_EXCEPTION_IF_NULL(sparse);
  if (common::AnfAlgo::GetCNodeName(sparse) != kSparseSoftmaxCrossEntropyWithLogitsOpName) {
    MS_LOG(INTERNAL_EXCEPTION) << "The pass of " << pass_name << "'s input node should be "
                               << kSparseSoftmaxCrossEntropyWithLogitsOpName << ", but got "
                               << common::AnfAlgo::GetCNodeName(sparse) << trace::DumpSourceLines(sparse);
  }
  if (common::AnfAlgo::HasNodeAttr(kAttrIsGrad, sparse)) {
    return common::AnfAlgo::GetNodeAttr<bool>(sparse, kAttrIsGrad);
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Node of " << sparse->fullname_with_scope() << " does not have the attr "
                               << kAttrIsGrad << ", related pass: " << pass_name << trace::DumpSourceLines(sparse);
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
  static std::vector<std::string> ret{prim::kPrimSparseSoftmaxCrossEntropyWithLogits->name()};
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

  auto is_dynamic = common::AnfAlgo::IsDynamicShape(sparse_softmax_node);
  MS_EXCEPTION_IF_CHECK_FAIL(
    !is_dynamic, "SparseSoftmaxCrossEntropyWithLogits with dynamic inputs is not supported yet in Graph mode!");

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

  auto is_dynamic = common::AnfAlgo::IsDynamicShape(sparse_softmax_node);
  MS_EXCEPTION_IF_CHECK_FAIL(
    !is_dynamic, "Grad SparseSoftmaxCrossEntropyWithLogits with dynamic inputs is not supported yet in Graph mode!");

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
  new_depend->set_scope(depend_node->scope());
  (void)manager->Replace(sparse_softmax_node_grad, new_depend);

  int64_t batch_size;
  int64_t depth;
  GetDepthAndBatchSizeFromSparseSoftmaxNode(sparse_softmax_node, &batch_size, &depth);
  ShapeVector shape = ShapeVector{batch_size, depth};
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {shape}, new_mul_node.get());

  auto logits_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 0UL);
  // Reshape 1D result to multi-dim result.
  auto reshape_node = CreateReshape(graph, new_mul_node, logits_shape, *this);
  return reshape_node;
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

  auto is_dynamic = common::AnfAlgo::IsDynamicShape(sparse_softmax_node);
  MS_EXCEPTION_IF_CHECK_FAIL(
    !is_dynamic, "Grad SparseSoftmaxCrossEntropyWithLogits with dynamic inputs is not supported yet in Graph mode!");

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

  // Reshape 1D result to multi-dim result.
  auto logits_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 0UL);
  auto reshape_node = CreateReshape(graph, mul_node, logits_shape, *this);
  return reshape_node;
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

  auto is_dynamic = common::AnfAlgo::IsDynamicShape(sparse_softmax_node);
  MS_EXCEPTION_IF_CHECK_FAIL(
    !is_dynamic, "SparseSoftmaxCrossEntropyWithLogits with dynamic inputs is not supported yet in PyNative mode!");

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
  static std::vector<std::string> ret{prim::kPrimSparseSoftmaxCrossEntropyWithLogits->name(), prim::kPrimMul->name()};
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
  auto is_dynamic = common::AnfAlgo::IsDynamicShape(sparse_softmax_node);
  MS_EXCEPTION_IF_CHECK_FAIL(
    !is_dynamic, "SparseSoftmaxCrossEntropyWithLogits with dynamic inpputs is not supported yet in PyNative mode.");

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

  int64_t batch_size;
  int64_t depth;
  opt::GetDepthAndBatchSizeFromSparseSoftmaxNode(sparse_softmax_node, &batch_size, &depth);
  ShapeVector shape = ShapeVector{batch_size, depth};
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {shape}, new_mul_node.get());

  auto logits_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 0UL);
  // Reshape 1D result to multi-dim result.
  auto reshape_node = CreateReshape(graph, new_mul_node, logits_shape, *this);
  return reshape_node;
}

std::vector<std::string> PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2::MustExistPrimitiveName() const {
  static std::vector<std::string> ret{prim::kPrimSparseSoftmaxCrossEntropyWithLogits->name(), prim::kPrimCast->name(),
                                      prim::kPrimMul->name()};
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
  auto is_dynamic = common::AnfAlgo::IsDynamicShape(sparse_softmax_node);
  MS_EXCEPTION_IF_CHECK_FAIL(
    !is_dynamic, "SparseSoftmaxCrossEntropyWithLogits with dynamic inputs is not supported yet in PyNative mode.");

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

  int64_t batch_size;
  int64_t depth;
  GetDepthAndBatchSizeFromSparseSoftmaxNode(sparse_softmax_node, &batch_size, &depth);
  ShapeVector shape = ShapeVector{batch_size, depth};
  common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {shape}, new_mul_node.get());

  // Reshape 1D result to multi-dim result.
  auto logits_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, 0UL);
  auto reshape_node = CreateReshape(graph, new_mul_node, logits_shape, *this);
  return reshape_node;
}

const BaseRef GeSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimSparseSoftmaxCrossEntropyWithLogits, Xs});
}

const AnfNodePtr GeSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR::Process(const FuncGraphPtr &graph,
                                                                           const AnfNodePtr &node,
                                                                           const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto sparse_softmax_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(sparse_softmax_node);

  auto is_dynamic = common::AnfAlgo::IsDynamicShape(sparse_softmax_node);
  MS_EXCEPTION_IF_CHECK_FAIL(
    !is_dynamic, "GE SparseSoftmaxCrossEntropyWithLogits with dynamic inputs is not supported yet in Graph mode!");

  if (common::AnfAlgo::HasNodeAttr(kAttrVisited, sparse_softmax_node)) {
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  const auto &input_list = sparse_softmax_node->inputs();
  if (input_list.size() != kSparseSoftmaxCrossEntropyWithLogitsInputNum) {
    MS_LOG(EXCEPTION) << "SparseSoftmaxCrossEntropyWithLogits's input size must be "
                      << kSparseSoftmaxCrossEntropyWithLogitsInputNum << ", but got " << input_list.size();
  }

  bool is_grad = false;
  if (common::AnfAlgo::HasNodeAttr(kAttrIsGrad, sparse_softmax_node)) {
    is_grad = common::AnfAlgo::GetNodeAttr<bool>(sparse_softmax_node, kAttrIsGrad);
  }

  auto features = input_list[kIndex1];
  MS_EXCEPTION_IF_NULL(features);
  auto dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(sparse_softmax_node, kIndex0);
  ShapeVector output_shape = {};
  auto loss_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(dtype), output_shape);
  AbstractBasePtrList new_node_abstract_list{loss_abstract, features->abstract()};
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(new_node_abstract_list);
  sparse_softmax_node->set_abstract(abstract_tuple);
  std::vector<AnfNodePtr> new_cnode_outputs;
  CreateMultipleOutputsOfAnfNode(graph, sparse_softmax_node, kSparseSoftmaxCrossEntropyWithLogitsOutputNum,
                                 &new_cnode_outputs);
  if (new_cnode_outputs.size() != kSparseSoftmaxCrossEntropyWithLogitsOutputNum) {
    MS_LOG(INTERNAL_EXCEPTION) << "The output size of node " << sparse_softmax_node->DebugString() << " should be "
                               << kAdamApplyOneOutputNum << trace::DumpSourceLines(node);
  }

  return is_grad ? new_cnode_outputs[kIndex1] : new_cnode_outputs[kIndex0];
}

std::vector<std::string> GeSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR::MustExistPrimitiveName() const {
  static std::vector<std::string> ret{prim::kPrimSparseSoftmaxCrossEntropyWithLogits->name()};
  return ret;
}
}  // namespace opt
}  // namespace mindspore
