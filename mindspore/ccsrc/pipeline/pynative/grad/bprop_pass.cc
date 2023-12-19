/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "pipeline/pynative/grad/bprop_pass.h"
#include <memory>
#include <vector>
#include <functional>
#include "pipeline/pynative/pynative_utils.h"
#include "ops/sequence_ops.h"
#include "ops/nn_ops.h"
#include "ops/op_utils.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace pynative {
namespace bprop_pass {
namespace {
constexpr auto kTupleToMakeTuple = "tuple_to_make_tuple";

mindspore::HashMap<AnfNodePtr, std::vector<std::pair<size_t, AnfNodePtr>>> node_attr_value_;

void CreateTensorByConstantValue(const ValueNodePtr &v_node) {
  MS_EXCEPTION_IF_NULL(v_node);
  const auto &value = v_node->value();
  MS_EXCEPTION_IF_NULL(value);
  auto type = value->type();
  if (PyNativeAlgo::Common::IsTensor(value, true) || value->isa<Number>() || value->isa<None>() ||
      (type != nullptr && type->isa<String>())) {
    return;
  }
  tensor::TensorPtr tensor_ptr = nullptr;
  if (value->isa<Scalar>()) {
    tensor_ptr = ScalarToTensor(value->cast<ScalarPtr>());
  } else if (value->isa<ValueTuple>()) {
    tensor_ptr = opt::CreateTupleTensor(value->cast<ValueTuplePtr>());
  } else if (value->isa<ValueList>()) {
    tensor_ptr = opt::CreateTupleTensor(std::make_shared<ValueTuple>(value->cast<ValueListPtr>()->value()));
  } else {
    MS_LOG(EXCEPTION) << "The value should be a scalar or value tuple, but get type " << value->type_name()
                      << ", value " << value->ToString();
  }
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  v_node->set_value(tensor_ptr);
  v_node->set_abstract(tensor_ptr->ToAbstract());
}

void ChangeInputToAttr(const PrimitivePtr &prim, const CNodePtr &cnode, const ValuePtr &input_names,
                       const mindspore::HashSet<size_t> &input_to_attr, bool grad_by_value) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(input_names);
  const auto &input_names_vec = GetValue<std::vector<std::string>>(input_names);
  AnfNodePtrList new_inputs{NewValueNode(prim)};
  size_t convert_size = 0;
  for (size_t i = 0; i < cnode->size() - 1; ++i) {
    auto input_node = cnode->input(i + 1);
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_node->isa<ValueNode>() && input_to_attr.find(i) != input_to_attr.end()) {
      const auto &value_node = input_node->cast<ValueNodePtr>();
      MS_LOG(DEBUG) << "start erase input[" << i << "] of cnode[" + cnode->DebugString() + "]";
      if (i >= input_names_vec.size()) {
        MS_LOG(EXCEPTION) << "Index " << i << " is larger than input names size [" << input_names_vec.size() << "]";
      }
      const auto &value = value_node->value();
      if (value->isa<tensor::Tensor>()) {
        auto tensor = value->cast<tensor::TensorPtr>();
        if (tensor->data().const_data() == nullptr && !tensor->has_user_data(kTensorValueIsEmpty)) {
          return;
        }
      }
      ++convert_size;
      if (!grad_by_value) {
        auto &pair = node_attr_value_[cnode];
        (void)pair.emplace_back(i, value_node);
      }
      prim->set_attr(input_names_vec[i], value);
    } else {
      (void)new_inputs.emplace_back(input_node);
    }
  }
  if (convert_size > 0) {
    cnode->AddAttr(kAttrConvertAttrNode, MakeValue(convert_size));
  }
  cnode->set_inputs(new_inputs);
}

void SetReverseParameterReplaceInfo(autograd::AutoGradCellImpl *auto_grad_cell_ptr, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(auto_grad_cell_ptr);
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return;
  }
  const auto &cnode = node->cast<CNodePtr>();
  for (size_t i = 1; i < cnode->size(); ++i) {
    const auto &input = cnode->input(i);
    MS_EXCEPTION_IF_NULL(input);
    if (input->isa<Parameter>()) {
      auto_grad_cell_ptr->AddReverseUser(input, cnode, i);
    } else if (input->isa<CNode>()) {
      SetReverseParameterReplaceInfo(auto_grad_cell_ptr, input);
    }
  }
}

template <typename T>
std::optional<T> GetScalarAnfNodeValue(const AnfNodePtr &anf_node) {
  if (!anf_node->isa<ValueNode>()) {
    return std::nullopt;
  }
  auto value_node = anf_node->cast<ValueNodePtr>();
  auto value_opt = mindspore::ops::GetScalarValue<T>(value_node->value());
  if (!value_opt.has_value()) {
    return std::nullopt;
  }
  return value_opt.value();
}

CNodePtr CreateBNInferGrad(autograd::AutoGradCellImpl *auto_grad_cell_ptr, const CNodePtr &batchnorm_cnode,
                           const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(auto_grad_cell_ptr);
  MS_EXCEPTION_IF_NULL(batchnorm_cnode);
  MS_EXCEPTION_IF_NULL(node);
  constexpr size_t kIdxGrads = 1;
  constexpr size_t kIdxScale = 3;
  constexpr size_t kIdxVariance = 5;
  constexpr size_t kIdxIsTraining = 7;
  constexpr size_t kIdxEpsilon = 8;

  AnfNodePtrList inputs{NewValueNode(prim::kPrimBNInferGrad)};
  (void)inputs.emplace_back(batchnorm_cnode->input(kIdxGrads));
  (void)inputs.emplace_back(batchnorm_cnode->input(kIdxScale));
  (void)inputs.emplace_back(batchnorm_cnode->input(kIdxVariance));
  (void)inputs.emplace_back(batchnorm_cnode->input(kIdxEpsilon));
  auto new_node = auto_grad_cell_ptr->ad_param()->tape_->FuncGraph::NewCNode(inputs);
  new_node->set_abstract(node->abstract());
  new_node->set_scope(batchnorm_cnode->scope());

  if (!auto_grad_cell_ptr->grad_by_value()) {
    SetReverseParameterReplaceInfo(auto_grad_cell_ptr, batchnorm_cnode->input(kIndex2));
    SetReverseParameterReplaceInfo(auto_grad_cell_ptr, batchnorm_cnode->input(kIndex4));
    SetReverseParameterReplaceInfo(auto_grad_cell_ptr, batchnorm_cnode->input(kIndex6));
  }
  auto_grad_cell_ptr->AddUser(batchnorm_cnode->input(kIdxGrads), new_node, kIndex1);
  auto_grad_cell_ptr->AddUser(batchnorm_cnode->input(kIdxScale), new_node, kIndex2);
  auto_grad_cell_ptr->AddUser(batchnorm_cnode->input(kIdxVariance), new_node, kIndex3);

  auto is_training_opt = GetScalarAnfNodeValue<bool>(batchnorm_cnode->input(kIdxIsTraining));
  if (is_training_opt.has_value()) {
    auto is_training = is_training_opt.value();
    common::AnfAlgo::SetNodeAttr(kAttrIsTraining, MakeValue(is_training), new_node);
  } else {
    MS_LOG(ERROR) << "For BNInferGrad pass, failed to get attr is_training.";
  }

  auto epsilon_opt = GetScalarAnfNodeValue<pyfloat>(batchnorm_cnode->input(kIdxEpsilon));
  float epsilon{1e-5};
  if (epsilon_opt.has_value()) {
    epsilon = epsilon_opt.has_value() ? epsilon_opt.value() : 1e-5;
  } else {
    MS_LOG(ERROR) << "For BNInferGrad pass, failed to get attr epsilon, use default epsilon: 1e-5.";
  }
  common::AnfAlgo::SetNodeAttr(kAttrEpsilon, MakeValue(epsilon), new_node);
  return new_node;
}

class SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR {
 public:
  CNodePtr Run(const CNodePtr &mul_node, const AnfNodePtr &sparse_softmax_node) {
    GetDepthAndBatchSizeFromSparseSoftmaxNode(sparse_softmax_node);

    AnfNodePtrList softmax_node_outputs;
    auto expand_dims_node = CreateMulInput(mul_node, sparse_softmax_node, &softmax_node_outputs);

    AnfNodePtrList new_mul_inputs{NewValueNode(prim::kPrimMul), softmax_node_outputs[kIndex1], expand_dims_node};
    auto new_mul_node = auto_grad_cell_ptr_->ad_param()->tape_->FuncGraph::NewCNode(new_mul_inputs);
    new_mul_node->set_abstract(mul_node->abstract());
    new_mul_node->set_scope(mul_node->scope());
    auto is_dynamic = common::AnfAlgo::IsDynamicShape(sparse_softmax_node);
    ShapeVector shape = is_dynamic ? ShapeVector{-1, depth_} : ShapeVector{batch_size_, depth_};
    common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {shape}, new_mul_node.get());

    auto logits_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, kIndex0);
    // Reshape 1D result to multi-dim result.
    auto reshape_node = CreateReshape(new_mul_node, logits_shape);
    return reshape_node;
  }

  autograd::AutoGradCellImpl *auto_grad_cell_ptr_{nullptr};

 private:
  CNodePtr CreateReshape(const AnfNodePtr &input_node, const ShapeVector &shape) {
    MS_EXCEPTION_IF_NULL(input_node);

    auto reshape_primitive = std::make_shared<Primitive>(kReshapeOpName);
    std::vector<std::string> input_names = {"x", "shape"};
    std::vector<std::string> output_names = {"output"};
    reshape_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
    reshape_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));

    auto shape_node = NewValueNode(shape);
    CreateTensorByConstantValue(shape_node);
    AnfNodePtrList reshape_inputs{NewValueNode(reshape_primitive), input_node, shape_node};
    auto reshape_node = auto_grad_cell_ptr_->ad_param()->tape_->FuncGraph::NewCNode(reshape_inputs);
    auto data_types = common::AnfAlgo::GetOutputInferDataType(input_node, kIndex0);
    common::AnfAlgo::SetOutputInferTypeAndShape({data_types}, {shape}, reshape_node.get());
    reshape_node->set_scope(input_node->scope());
    constexpr auto kShapeFromTensor = "shape_from_tensor";
    common::AnfAlgo::SetNodeAttr(kShapeFromTensor, MakeValue(true), reshape_node);
    auto_grad_cell_ptr_->AddUser(input_node, reshape_node, kIndex1);
    return reshape_node;
  }

  void GetDepthAndBatchSizeFromSparseSoftmaxNode(const AnfNodePtr &sparse_softmax_node) {
    MS_EXCEPTION_IF_NULL(sparse_softmax_node);
    auto logits_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, kIndex0);
    auto labels_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, kIndex1);
    if (!logits_shape.empty()) {
      size_t index = logits_shape.size() - 1;
      depth_ = logits_shape[index];
    } else {
      MS_LOG(EXCEPTION) << "Logits's shape of node [" << sparse_softmax_node->DebugString() << "] is empty"
                        << trace::DumpSourceLines(sparse_softmax_node);
    }
    batch_size_ = std::accumulate(labels_shape.begin(), labels_shape.end(), 1, std::multiplies<int64_t>());
  }

  CNodePtr CreateOneHot(const CNodePtr &sparse_softmax_node) {
    MS_EXCEPTION_IF_NULL(sparse_softmax_node);

    auto is_dynamic = common::AnfAlgo::IsDynamicShape(sparse_softmax_node);
    ShapeVector shape = is_dynamic ? ShapeVector{-1} : ShapeVector{batch_size_};

    // Reshape multi-dim labels to 1D labels.
    auto reshape_node = CreateReshape(sparse_softmax_node->input(kIndex2), shape);

    auto value_on = std::make_shared<tensor::Tensor>(1.0, kFloat32);
    auto value_on_node = PyNativeAlgo::Common::CreateValueNodeByValue(value_on);
    auto value_off = std::make_shared<tensor::Tensor>(0.0, kFloat32);
    auto value_off_node = PyNativeAlgo::Common::CreateValueNodeByValue(value_off);
    auto value_axis = MakeValue<int64_t>(-1);
    auto value_axis_node = PyNativeAlgo::Common::CreateValueNodeByValue(value_axis);
    auto one_hot_primitive = std::make_shared<Primitive>(kOneHotOpName);
    std::vector<std::string> input_names = {"indices", "depth", "on_value", "off_value", "axis"};
    std::vector<std::string> output_names = {"output"};
    one_hot_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
    one_hot_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));

    auto depth_node = PyNativeAlgo::Common::CreateValueNodeByValue(MakeValue<int64_t>(depth_));
    CreateTensorByConstantValue(depth_node);
    AnfNodePtrList one_hot_inputs{
      NewValueNode(one_hot_primitive), reshape_node, depth_node, value_on_node, value_off_node, value_axis_node};
    auto one_hot_node = auto_grad_cell_ptr_->ad_param()->tape_->FuncGraph::NewCNode(one_hot_inputs);
    ShapeVector one_hot_shape = {batch_size_, depth_};
    common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {one_hot_shape}, one_hot_node.get());
    one_hot_node->set_scope(sparse_softmax_node->scope());
    auto_grad_cell_ptr_->AddUser(reshape_node, one_hot_node, kIndex1);
    return one_hot_node;
  }

  CNodePtr CreateSoftmaxCrossEntropyWithLogits(const CNodePtr &sparse_softmax_node, const CNodePtr &one_hot_node) {
    MS_EXCEPTION_IF_NULL(sparse_softmax_node);
    MS_EXCEPTION_IF_NULL(one_hot_node);

    auto is_dynamic = common::AnfAlgo::IsDynamicShape(sparse_softmax_node);
    ShapeVector shape = is_dynamic ? ShapeVector{-1, depth_} : ShapeVector{batch_size_, depth_};

    // Reshape multi-dim logits to 2D logits.
    auto reshape_node = CreateReshape(sparse_softmax_node->input(kIndex1), shape);
    AnfNodePtrList inputs{NewValueNode(std::make_shared<Primitive>(kSoftmaxCrossEntropyWithLogitsOpName)), reshape_node,
                          one_hot_node};
    auto softmax_node = auto_grad_cell_ptr_->ad_param()->tape_->FuncGraph::NewCNode(inputs);
    ShapeVector loss_shape = {batch_size_};
    auto data_types = common::AnfAlgo::GetOutputInferDataType(one_hot_node, kIndex0);
    auto types = {data_types, data_types};
    auto shapes = {loss_shape, shape};
    common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, softmax_node.get());
    softmax_node->set_scope(sparse_softmax_node->scope());
    return softmax_node;
  }

  void CreateMultipleOutputsOfAnfNode(const AnfNodePtr &node, size_t output_num, AnfNodePtrList *outputs) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(outputs);
    MS_EXCEPTION_IF_NULL(node->abstract());
    const auto &abs_seq = node->abstract()->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    if (abs_seq->size() != output_num) {
      MS_LOG(EXCEPTION) << "Abstract seq size " << abs_seq->size() << " is not equal to " << output_num;
    }
    for (size_t i = 0; i < output_num; i++) {
      auto idx = PyNativeAlgo::Common::CreateValueNodeByValue(MakeValue<int64_t>(SizeToLong(i)));
      auto tuple_getitem =
        auto_grad_cell_ptr_->ad_param()->tape_->FuncGraph::NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
      tuple_getitem->set_abstract(abs_seq->elements()[i]);
      (void)outputs->emplace_back(tuple_getitem);
    }
  }

  CNodePtr CreateTile(const CNodePtr &sparse_softmax_node, const CNodePtr &mul_node) {
    MS_EXCEPTION_IF_NULL(sparse_softmax_node);
    MS_EXCEPTION_IF_NULL(mul_node);
    if (batch_size_ == 1) {
      return nullptr;
    }
    auto tile_primitive = std::make_shared<Primitive>(kTileOpName);
    std::vector<std::string> input_names = {"x", "multiples"};
    std::vector<std::string> output_names = {"output"};
    tile_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
    tile_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));

    AnfNodePtrList tile_inputs;
    if (batch_size_ < 0) {
      AnfNodePtrList dynamic_shape_inputs{NewValueNode(std::make_shared<Primitive>("DynamicShape")),
                                          sparse_softmax_node->input(kIndex2)};
      auto shape_node = auto_grad_cell_ptr_->ad_param()->tape_->FuncGraph::NewCNode(dynamic_shape_inputs);
      auto labels_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(sparse_softmax_node, kIndex1);
      ShapeVector tensor_shp({static_cast<int64_t>(labels_shape.size())});
      auto dynamic_shape_abstract =
        std::make_shared<abstract::AbstractTensor>(kInt64, std::make_shared<abstract::Shape>(tensor_shp));
      MS_EXCEPTION_IF_NULL(dynamic_shape_abstract);
      shape_node->set_abstract(dynamic_shape_abstract);
      shape_node->set_scope(mul_node->scope());
      auto_grad_cell_ptr_->AddUser(sparse_softmax_node->input(kIndex2), shape_node, kIndex1);
      tile_inputs = {NewValueNode(tile_primitive), mul_node->input(kIndex2), shape_node};
    } else {
      std::vector<int64_t> multiples_v = {batch_size_};
      auto multiples_node = PyNativeAlgo::Common::CreateValueNodeByValue(MakeValue(multiples_v));
      tile_inputs = {NewValueNode(tile_primitive), mul_node->input(kIndex2), multiples_node};
    }

    auto tile_node = auto_grad_cell_ptr_->ad_param()->tape_->FuncGraph::NewCNode(tile_inputs);
    ShapeVector tile_shape = {batch_size_};
    common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetPrevNodeOutputInferDataType(mul_node, 1UL)},
                                                {tile_shape}, tile_node.get());
    tile_node->set_scope(mul_node->scope());
    auto_grad_cell_ptr_->AddUser(mul_node->input(kIndex2), tile_node, kIndex1);
    // feature map set
    std::vector<size_t> feature_map_input_indexs;
    (void)feature_map_input_indexs.emplace_back(0);
    constexpr auto kIsFeatureMapInputList = "IsFeatureMapInputList";
    common::AnfAlgo::SetNodeAttr(kIsFeatureMapInputList, MakeValue(feature_map_input_indexs), tile_node);
    return tile_node;
  }

  CNodePtr CreateRealDiv(const CNodePtr &sparse_softmax_node, const AnfNodePtr &tile_node) {
    MS_EXCEPTION_IF_NULL(sparse_softmax_node);
    MS_EXCEPTION_IF_NULL(tile_node);
    auto y_value = static_cast<float>(batch_size_);
    auto y = std::make_shared<tensor::Tensor>(y_value, kFloat32);
    auto y_node = PyNativeAlgo::Common::CreateValueNodeByValue(MakeValue(y));

    auto real_div_primitive = std::make_shared<Primitive>(kRealDivOpName);
    std::vector<std::string> input_names = {"x", "y"};
    std::vector<std::string> output_names = {"output"};
    real_div_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
    real_div_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));

    AnfNodePtrList real_div_inputs{NewValueNode(real_div_primitive), tile_node, y_node};
    auto real_div_node = auto_grad_cell_ptr_->ad_param()->tape_->FuncGraph::NewCNode(real_div_inputs);
    real_div_node->set_abstract(tile_node->abstract());
    real_div_node->set_scope(sparse_softmax_node->scope());
    return real_div_node;
  }

  CNodePtr CreateExpandDims(const CNodePtr &real_div_node) {
    MS_EXCEPTION_IF_NULL(real_div_node);

    constexpr int64_t axis = -1;
    auto axis_abstract = std::make_shared<abstract::AbstractScalar>();
    MS_EXCEPTION_IF_NULL(axis_abstract);
    axis_abstract->set_type(kInt64);
    auto axis_node = PyNativeAlgo::Common::CreateValueNodeByValue(MakeValue(axis), axis_abstract);
    MS_EXCEPTION_IF_NULL(axis_node);

    auto expand_dims_primitive = std::make_shared<Primitive>(kExpandDimsOpName);
    std::vector<std::string> input_names = {"x"};
    std::vector<std::string> output_names = {"output"};
    expand_dims_primitive->set_attr(kAttrInputNames, MakeValue(input_names));
    expand_dims_primitive->set_attr(kAttrOutputNames, MakeValue(output_names));

    AnfNodePtrList expand_dims_inputs = {NewValueNode(expand_dims_primitive), real_div_node, axis_node};
    auto expand_dims_node = auto_grad_cell_ptr_->ad_param()->tape_->FuncGraph::NewCNode(expand_dims_inputs);
    auto y_shape = common::AnfAlgo::GetOutputInferShape(real_div_node, 0UL);
    (void)y_shape.emplace_back(1);
    common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(real_div_node, 0UL)},
                                                {y_shape}, expand_dims_node.get());
    common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(axis), expand_dims_node);
    expand_dims_node->set_scope(real_div_node->scope());
    return expand_dims_node;
  }

  CNodePtr CreateMulInput(const CNodePtr &mul_node, const AnfNodePtr &sparse_softmax_node,
                          AnfNodePtrList *softmax_node_outputs) {
    MS_EXCEPTION_IF_NULL(mul_node);
    MS_EXCEPTION_IF_NULL(sparse_softmax_node);
    auto sparse_softmax_cnode = sparse_softmax_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(sparse_softmax_cnode);
    auto one_hot_node = CreateOneHot(sparse_softmax_cnode);
    auto softmax_node = CreateSoftmaxCrossEntropyWithLogits(sparse_softmax_cnode, one_hot_node);
    CreateMultipleOutputsOfAnfNode(softmax_node, opt::kSoftmaxCrossEntropyWithLogitsOutputNum, softmax_node_outputs);
    auto tile_node = CreateTile(sparse_softmax_cnode, mul_node);
    CNodePtr real_div_node;
    if (tile_node == nullptr) {
      real_div_node = CreateRealDiv(sparse_softmax_cnode, mul_node->input(kIndex2));
      auto_grad_cell_ptr_->AddUser(mul_node->input(kIndex2), real_div_node, kIndex1);
    } else {
      real_div_node = CreateRealDiv(sparse_softmax_cnode, tile_node);
    }
    auto expand_dims_node = CreateExpandDims(real_div_node);
    return expand_dims_node;
  }

  int64_t batch_size_{0};
  int64_t depth_{0};
};

void AddCNodeInputs(const CNodePtr &cnode, AnfNodePtrList *cnode_inputs, size_t index, const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  MS_EXCEPTION_IF_NULL(input_node);
  auto new_inputs = cnode->inputs();
  (void)new_inputs.insert(new_inputs.begin() + SizeToLong(index) + kIndex1, input_node);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  (void)cnode_inputs->insert(cnode_inputs->begin() + SizeToLong(index) + kIndex1, input_node);
  cnode->set_inputs(new_inputs);
}

AnfNodePtr GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR(const AnfNodePtr &node, const std::string &op_name,
                                                              autograd::AutoGradCellImpl *auto_grad_cell_ptr) {
  if (op_name != kSparseSoftmaxCrossEntropyWithLogitsOpName) {
    return node;
  }
  MS_EXCEPTION_IF_NULL(node);
  auto mul_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mul_node);
  if (mul_node->HasAttr(kIsKNode) || !IsPrimitiveCNode(mul_node, prim::kPrimMul)) {
    return node;
  }

  auto sparse_softmax_node = mul_node->input(kIndex1);
  if (!common::AnfAlgo::GetNodeAttr<bool>(sparse_softmax_node, kAttrIsGrad)) {
    return node;
  }
  // Use static class for create only once
  static auto sparse_softmax_cross_entropy_with_logits =
    std::make_shared<SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>();
  sparse_softmax_cross_entropy_with_logits->auto_grad_cell_ptr_ = auto_grad_cell_ptr;
  return sparse_softmax_cross_entropy_with_logits->Run(mul_node, sparse_softmax_node);
}
}  // namespace

void PyNativePassForward::ConvertMakeTupleInputToDynamicInput(const AnfNodePtr &node, SeenNum seen) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  bool need_traverse = !grad_by_value_ && cnode->HasAttr(kIsKNode);
  if (need_traverse || cnode->seen_ == seen || IsPrimitiveCNode(cnode, prim::kPrimBpropCut) ||
      !IsPrimitiveCNode(cnode)) {
    return;
  }
  cnode->seen_ = seen;
  if (IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
    ConvertMakeTupleInputToDynamicInput(cnode->input(kIndex1), seen);
    return;
  }
  for (size_t i = 1; i < cnode->size(); ++i) {
    ConvertMakeTupleInputToDynamicInput(cnode->input(i), seen);
  }

  if (!IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) &&
      std::any_of(cnode->inputs().begin() + 1, cnode->inputs().end(),
                  [](const AnfNodePtr &node) { return node->abstract()->isa<abstract::AbstractSequence>(); })) {
    AnfNodePtrList plant_inputs;
    std::vector<int64_t> dyn_input_sizes;
    (void)plant_inputs.emplace_back(common::AnfAlgo::GetCNodePrimitiveNode(cnode));
    for (size_t i = 1; i < cnode->size(); ++i) {
      const auto &input_node = cnode->input(i);
      if (common::AnfAlgo::CheckPrimitiveType(input_node, prim::kPrimMakeTuple)) {
        auto dyn_input_size = opt::SplitTupleInputs(auto_grad_cell_ptr_->ad_param()->tape_, input_node, &plant_inputs);
        (void)dyn_input_sizes.emplace_back(dyn_input_size);
      } else {
        (void)plant_inputs.emplace_back(input_node);
        (void)dyn_input_sizes.emplace_back(-1);
      }
    }
    // If there is dynamic input, set the dyn_input_sizes as an attribute and update the inputs.
    if (std::any_of(dyn_input_sizes.begin(), dyn_input_sizes.end(), [](int64_t s) { return s >= 0; })) {
      cnode->AddAttr(kTupleToMakeTuple, MakeValue(true));
      common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_sizes), cnode);
      MS_LOG(DEBUG) << "Change node to dynamic len " << cnode->DebugString();
      cnode->set_inputs(plant_inputs);
      for (size_t i = 1; i < plant_inputs.size(); ++i) {
        auto_grad_cell_ptr_->AddUser(plant_inputs[i], cnode, i);
      }
    }
  }
}

CNodePtr PyNativePassForward::ConvertConstInputToAttr(const CNodePtr &cnode, bool is_dynamic_shape) {
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(DEBUG) << "Get cnode not primitive " << cnode->DebugString();
    return cnode;
  }
  auto TraverseCNode = [this, is_dynamic_shape](const CNodePtr &cnode) {
    for (size_t i = 1; i < cnode->size(); ++i) {
      // Avoiding infinite loops
      if (!cnode->HasAttr(kIsKNode) && cnode->input(i)->isa<CNode>()) {
        cnode->set_input(i, ConvertConstInputToAttr(cnode->input(i)->cast<CNodePtr>(), is_dynamic_shape));
      }
    }
  };

  mindspore::HashSet<size_t> input_to_attr = {};
  PyNativeAlgo::Common::GetConstInputToAttr(prim, prim->name(), device_target_, is_dynamic_shape, &input_to_attr);
  if (input_to_attr.empty()) {
    TraverseCNode(cnode);
    return cnode;
  }
  const auto &input_names = prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return cnode;
  }

  ChangeInputToAttr(prim, cnode, input_names, input_to_attr, grad_by_value_);

  // If cast input has a cast
  TraverseCNode(cnode);
  return cnode;
}

AnfNodePtr PyNativePassForward::BatchNormGradToBNInferGrad(const AnfNodePtr &node, const std::string &op_name) {
  if (op_name != kBatchNormOpName) {
    return node;
  }
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->HasAttr(kIsKNode) || !IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
    return cnode;
  }
  auto batchnorm_grad_node = cnode->input(kRealInputNodeIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(batchnorm_grad_node);
  if (!IsPrimitiveCNode(batchnorm_grad_node, prim::kPrimBatchNormGrad)) {
    return cnode;
  }
  AnfNodePtr index_node = cnode->input(kInputNodeOutputIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(index_node);
  auto value_node = index_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto index = GetValue<int64_t>(value_node->value());
  if (index != 0) {
    MS_LOG(DEBUG) << "TupleGetitem must be 0th output of BatchNormGrad";
    return cnode;
  }
  auto batchnorm_grad_cnode = batchnorm_grad_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(batchnorm_grad_cnode);
  constexpr size_t kIdxIsTraining = 7;
  auto is_training_opt = GetScalarAnfNodeValue<bool>(batchnorm_grad_cnode->input(kIdxIsTraining));
  if (!is_training_opt.has_value()) {
    return cnode;
  }
  auto is_training = is_training_opt.value();
  if (is_training) {
    MS_LOG(DEBUG) << "Attr 'is_training' is true, no need do fusion";
    return cnode;
  }

  need_reverse_graph_ = true;
  auto new_cnode = CreateBNInferGrad(auto_grad_cell_ptr_, batchnorm_grad_cnode, node);
  auto &pair = node_attr_value_[new_cnode];
  (void)pair.emplace_back(UINT32_MAX, node);
  return new_cnode;
}

void PyNativePassForward::ReverseConstantToAttrNode(const CNodePtr &cnode, ValuePtrList *inputs_value,
                                                    AnfNodePtrList *cnode_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!cnode->HasAttr(kAttrConvertAttrNode)) {
    return;
  }
  ReverseCNodeInputs(cnode, cnode_inputs, inputs_value);
}

void PyNativePassForward::ReverseMakeTupleNode(const CNodePtr &cnode, ValuePtrList *inputs_value,
                                               AnfNodePtrList *cnode_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(inputs_value);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  if (!cnode->HasAttr(kTupleToMakeTuple)) {
    return;
  }
  AnfNodePtrList new_inputs{cnode->input(kIndex0)};
  const auto &dyn_input_sizes = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrDynInputSizes);
  for (size_t i = 0; i < dyn_input_sizes.size(); ++i) {
    if (dyn_input_sizes[i] >= 0) {
      // Compress input
      AnfNodePtrList cnode_tuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
      AnfNodePtrList knode_inputs{NewValueNode(prim::kPrimMakeTuple)};
      ValuePtrList value_tuple;
      abstract::AbstractBasePtrList abs_list;
      for (int64_t j = 0; j < dyn_input_sizes[i]; ++j) {
        auto input = cnode->input(i + j + kIndex1);
        (void)cnode_tuple_inputs.emplace_back(input);
        (void)knode_inputs.emplace_back(cnode_inputs->at(i + j + kIndex1));
        (void)value_tuple.emplace_back(inputs_value->at(i + j));
        (void)abs_list.emplace_back(input->abstract());
      }
      // Update knode inputs to make tuple inputs
      auto cnode_graph = cnode->func_graph();
      MS_EXCEPTION_IF_NULL(cnode_graph);
      auto cnode_tuple = cnode_graph->NewCNode(cnode_tuple_inputs);
      auto abs = std::make_shared<abstract::AbstractTuple>(abs_list);
      cnode_tuple->set_abstract(abs);
      (void)new_inputs.emplace_back(cnode_tuple);

      // Update knode inputs
      auto knode_input = auto_grad_cell_ptr_->ad_param()->tape_->FuncGraph::NewCNode(knode_inputs);
      knode_input->set_abstract(abs);
      size_t begin_index = i + kIndex1;
      (void)cnode_inputs->erase(cnode_inputs->begin() + SizeToLong(begin_index),
                                cnode_inputs->begin() + SizeToLong(begin_index) + dyn_input_sizes[i]);
      (void)cnode_inputs->emplace_back(knode_input);

      // Update input value
      (void)inputs_value->erase(inputs_value->begin() + SizeToLong(begin_index),
                                inputs_value->begin() + SizeToLong(begin_index) + dyn_input_sizes[i]);
      (void)inputs_value->emplace_back(std::make_shared<ValueTuple>(value_tuple));
    } else {
      (void)new_inputs.emplace_back(cnode->input(i + kIndex1));
    }
  }
  cnode->set_inputs(new_inputs);
  cnode->EraseAttr(kTupleToMakeTuple);
}

void PyNativePassForward::ReverseBNInfer(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!IsPrimitiveCNode(cnode, prim::kPrimBNInferGrad)) {
    return;
  }
  const auto item = node_attr_value_.find(cnode);
  if (item == node_attr_value_.end()) {
    return;
  }
  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, false);
  }
  if (item->second.size() != kIndex1) {
    MS_LOG(EXCEPTION) << "Replace item size " << item->second.size() << " is not equal to " << kIndex1;
  }
  if (!manager->Replace(cnode, item->second[kIndex0].second)) {
    MS_LOG(EXCEPTION) << "Replace failed. cnode " << cnode->DebugString() << " to cnode "
                      << item->second[kIndex0].second->DebugString();
  }
  node_attr_value_.erase(item);
}

void PyNativePassForward::ReverseCNodeInputs(const CNodePtr &cnode, AnfNodePtrList *cnode_inputs,
                                             ValuePtrList *inputs_value) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(inputs_value);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  const auto item = node_attr_value_.find(cnode);
  if (item == node_attr_value_.end()) {
    return;
  }
  for (const auto &t : item->second) {
    if (t.second->isa<ValueNode>()) {
      auto vnode = t.second->cast<ValueNodePtr>();
      auto v = vnode->value();
      (void)PyNativeAlgo::Common::SetValueGradInfo(v, nullptr, TensorGradType::kConstant);
      AddCNodeInputs(cnode, cnode_inputs, t.first, PyNativeAlgo::Common::CreateValueNodeByValue(v, nullptr));
      (void)inputs_value->insert(inputs_value->begin() + SizeToLong(t.first), v);
    } else if (t.second->isa<Parameter>()) {
      const auto it = auto_grad_cell_ptr_->ad_param()->anfnode_to_variable_adjoint_.find(t.second);
      if (it == auto_grad_cell_ptr_->ad_param()->anfnode_to_variable_adjoint_.end()) {
        MS_LOG(EXCEPTION) << "Can not find " << t.second << " in anfnode_to_variable_adjoint";
      }
      AddCNodeInputs(cnode, cnode_inputs, t.first, it->second->k_node());
      (void)inputs_value->insert(inputs_value->begin() + SizeToLong(t.first), it->second->out_value());
    } else {
      MS_LOG(EXCEPTION) << "No scenario for " << t.second->DebugString();
    }
  }
  node_attr_value_.erase(item);
}

void PyNativePassForward::ReversePassFuncGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &order = TopoSort(func_graph->output());
  for (const auto &node : order) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    // Bn Ascend only
    if (device_target_ == kAscendDevice) {
      ReverseBNInfer(cnode);
    }
  }
  need_reverse_graph_ = false;
  PyNativeAlgo::Common::DumpGraphIR("reverse_cnode_graph.ir", func_graph);
}

void PyNativePassForward::ReversePassCNode(const CNodePtr &cnode, ValuePtrList *inputs_value,
                                           AnfNodePtrList *cnode_inputs) {
  // Notice, The reverser step is opposite to the positive pass
  auto tape_graph = auto_grad_cell_ptr_->ad_param()->tape_;
  MS_EXCEPTION_IF_NULL(tape_graph);

  ReverseMakeTupleNode(cnode, inputs_value, cnode_inputs);
  ReverseConstantToAttrNode(cnode, inputs_value, cnode_inputs);
}

CNodePtr PyNativePassForward::PassForDin(const CNodePtr &cnode, const std::string &op_name, bool is_dynamic_shape) {
  // If you want add a pass here, please take care of high grad
  MS_EXCEPTION_IF_NULL(auto_grad_cell_ptr_);
  AnfNodePtr new_din = ConvertConstInputToAttr(cnode, is_dynamic_shape);

  // Ascend only
  if (device_target_ == kAscendDevice) {
    new_din = BatchNormGradToBNInferGrad(new_din, op_name);
    new_din = GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR(new_din, op_name, auto_grad_cell_ptr_);
  }
  return new_din->cast<CNodePtr>();
}

bool PyNativePassForward::need_reverse_graph_ = false;

void ClearCache() { node_attr_value_.clear(); }
}  // namespace bprop_pass
}  // namespace pynative
}  // namespace mindspore
