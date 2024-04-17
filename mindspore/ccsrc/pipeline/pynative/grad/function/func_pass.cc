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

#include "pipeline/pynative/grad/function/func_pass.h"
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
NodePtrList ChangeInputToAttr(const PrimitivePtr &prim, const NodePtrList &inputs, const ValuePtr &input_names,
                              const mindspore::HashSet<size_t> &input_to_attr) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(input_names);
  const auto &input_names_vec = GetValue<std::vector<std::string>>(input_names);
  NodePtrList new_inputs{};
  size_t convert_size = 0;
  size_t input_size = inputs.size();
  for (size_t i = 0; i < input_size; ++i) {
    auto value = inputs[i]->Value();
    if (value->isa<Scalar>() && input_to_attr.find(i) != input_to_attr.end()) {
      MS_LOG(DEBUG) << "start erase input[" << i << "] of op[" + prim->name() + "]";
      if (i >= input_names_vec.size()) {
        MS_LOG(EXCEPTION) << "Index " << i << " is larger than input names size [" << input_names_vec.size() << "]";
      }
      if (value->isa<tensor::Tensor>()) {
        auto tensor = value->cast<tensor::TensorPtr>();
        if (tensor->data().const_data() == nullptr && !tensor->has_user_data(kTensorValueIsEmpty)) {
          return inputs;
        }
      }
      ++convert_size;
      prim->set_attr(input_names_vec[i], value);
    } else {
      (void)new_inputs.emplace_back(inputs[i]);
    }
  }
  if (convert_size > 0) {
    (void)prim->AddAttr(kAttrConvertAttrNode, MakeValue(convert_size));
  }
  return new_inputs;
}

class SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR {
 public:
  NodePtr Run(const NodePtrList &inputs, const NodePtr &dout) {
    GetDepthAndBatchSizeFromSparseSoftmaxNode(inputs);

    NodePtrList softmax_node_outputs;
    auto expand_dims_node = CreateMulInput(inputs, dout, &softmax_node_outputs);

    NodePtr new_mul_node =
      func_builder_->EmitOp(func_builder_->NewPrimitive(kMulOpName), {softmax_node_outputs[kIndex1], expand_dims_node});
    // Reshape 1D result to multi-dim result.
    auto reshape_node = CreateReshape(new_mul_node, BaseShapeToShape(inputs[kIndex0]->GetShape()));
    return reshape_node;
  }

  autograd::FuncBuilder *func_builder_{nullptr};

 private:
  NodePtr CreateReshape(const NodePtr &input_node, const ShapeVector &shape) {
    MS_EXCEPTION_IF_NULL(input_node);
    std::vector<std::string> input_names = {"x", "shape"};
    std::vector<std::string> output_names = {"output"};
    auto prim = func_builder_->NewPrimitive(
      kReshapeOpName, {{kAttrInputNames, MakeValue(input_names)}, {kAttrOutputNames, MakeValue(output_names)}});
    constexpr auto kShapeFromTensor = "shape_from_tensor";
    prim->set_attr(kShapeFromTensor, MakeValue(true));
    auto shape_node = func_builder_->NewFuncNode(PyNativeAlgo::Common::CreateTensorByConstantValue(MakeValue(shape)),
                                                 nullptr, InputType::kConstant);
    shape_node->set_abstract(shape_node->Value()->ToAbstract());
    return func_builder_->EmitOp(prim, {input_node, shape_node});
  }

  void GetDepthAndBatchSizeFromSparseSoftmaxNode(const NodePtrList &inputs) {
    auto logits_shape = BaseShapeToShape(inputs[kIndex0]->GetShape());
    auto labels_shape = BaseShapeToShape(inputs[kIndex1]->GetShape());
    if (!logits_shape.empty()) {
      size_t index = logits_shape.size() - 1;
      depth_ = logits_shape[index];
    } else {
      MS_LOG(EXCEPTION) << "Logits's shape of node SparseSoftmaxCrossEntropyWithLogit is empty";
    }
    batch_size_ = std::accumulate(labels_shape.begin(), labels_shape.end(), 1, std::multiplies<int64_t>());
  }

  NodePtr CreateOneHot(const NodePtrList &inputs) {
    ShapeVector shape = ShapeVector{batch_size_};

    // Reshape multi-dim labels to 1D labels.
    auto reshape_node = CreateReshape(inputs[kIndex1], shape);

    auto value_on = std::make_shared<tensor::Tensor>(1.0, kFloat32);
    auto value_off = std::make_shared<tensor::Tensor>(0.0, kFloat32);
    auto value_axis = MakeValue<int64_t>(-1);
    std::vector<std::string> input_names = {"indices", "depth", "on_value", "off_value", "axis"};
    std::vector<std::string> output_names = {"output"};
    auto one_hot_primitive = func_builder_->NewPrimitive(
      kOneHotOpName, {{kAttrInputNames, MakeValue(input_names)}, {kAttrOutputNames, MakeValue(output_names)}});
    auto depth_node = func_builder_->NewFuncNode(MakeValue<int64_t>(depth_), nullptr, InputType::kConstant);
    depth_node->set_abstract(depth_node->Value()->ToAbstract());
    auto value_on_node = func_builder_->NewFuncNode(value_on, nullptr, InputType::kConstant);
    value_on_node->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(value_on_node->Value()->ToAbstract()));
    auto value_off_node = func_builder_->NewFuncNode(value_off, nullptr, InputType::kConstant);
    value_off_node->set_abstract(value_off_node->Value()->ToAbstract());
    auto value_axis_node = func_builder_->NewFuncNode(value_axis, nullptr, InputType::kConstant);
    value_axis_node->set_abstract(
      PyNativeAlgo::Common::SetAbstractValueToAnyValue(value_axis_node->Value()->ToAbstract()));
    NodePtrList one_hot_inputs{reshape_node, depth_node, value_on_node, value_off_node, value_axis_node};
    return func_builder_->EmitOp(one_hot_primitive, one_hot_inputs);
  }

  NodePtr CreateSoftmaxCrossEntropyWithLogits(const NodePtrList &inputs, const NodePtr &one_hot_node) {
    MS_EXCEPTION_IF_NULL(one_hot_node);
    ShapeVector shape = ShapeVector{batch_size_, depth_};
    // Reshape multi-dim logits to 2D logits.
    auto reshape_node = CreateReshape(inputs[kIndex0], shape);
    auto softmax_prim = func_builder_->NewPrimitive(kSoftmaxCrossEntropyWithLogitsOpName);
    return func_builder_->EmitOp(softmax_prim, {reshape_node, one_hot_node});
  }

  void CreateMultipleOutputsOfAnfNode(const NodePtr &node, size_t output_num, NodePtrList *outputs) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(outputs);
    MS_EXCEPTION_IF_NULL(node->abstract());
    const auto &abs_seq = node->abstract()->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    if (abs_seq->size() != output_num) {
      MS_LOG(EXCEPTION) << "Abstract seq size " << abs_seq->size() << " is not equal to " << output_num;
    }
    for (size_t i = 0; i < output_num; i++) {
      (void)outputs->emplace_back(func_builder_->TupleGetItem(node, i));
    }
  }

  NodePtr CreateTile(const NodePtrList &inputs, const NodePtr &dout) {
    if (batch_size_ == 1) {
      return nullptr;
    }
    std::vector<std::string> input_names = {"x", "multiples"};
    std::vector<std::string> output_names = {"output"};
    auto tile_primitive = func_builder_->NewPrimitive(
      kTileOpName, {{kAttrInputNames, MakeValue(input_names)}, {kAttrOutputNames, MakeValue(output_names)}});
    NodePtrList tile_inputs;
    if (batch_size_ < 0) {
      auto shape_node = func_builder_->EmitOp(func_builder_->NewPrimitive("DynamicShape"), {inputs[kIndex1]});
      tile_inputs = {dout, shape_node};
    } else {
      std::vector<int64_t> multiples_v = {batch_size_};
      auto multiples_node = func_builder_->NewFuncNode(MakeValue(multiples_v), nullptr, InputType::kConstant);
      multiples_node->set_abstract(multiples_node->Value()->ToAbstract());
      tile_inputs = {dout, multiples_node};
    }
    auto tile_node = func_builder_->EmitOp(tile_primitive, tile_inputs);
    // feature map set
    std::vector<size_t> feature_map_input_indexs;
    (void)feature_map_input_indexs.emplace_back(0);
    constexpr auto kIsFeatureMapInputList = "IsFeatureMapInputList";
    tile_primitive->set_attr(kIsFeatureMapInputList, MakeValue(feature_map_input_indexs));
    return tile_node;
  }

  NodePtr CreateRealDiv(const NodePtr &tile_node) {
    MS_EXCEPTION_IF_NULL(tile_node);
    auto y_value = static_cast<float>(batch_size_);
    auto y = std::make_shared<tensor::Tensor>(y_value, kFloat32);
    auto y_node = func_builder_->NewFuncNode(y, nullptr, InputType::kConstant);
    y_node->set_abstract(PyNativeAlgo::Common::SetAbstractValueToAnyValue(y_node->Value()->ToAbstract()));
    std::vector<std::string> input_names = {"x", "y"};
    std::vector<std::string> output_names = {"output"};
    auto real_div_primitive = func_builder_->NewPrimitive(
      kRealDivOpName, {{kAttrInputNames, MakeValue(input_names)}, {kAttrOutputNames, MakeValue(output_names)}});
    return func_builder_->EmitOp(real_div_primitive, {tile_node, y_node});
  }

  NodePtr CreateExpandDims(const NodePtr &real_div_node) {
    MS_EXCEPTION_IF_NULL(real_div_node);
    constexpr int64_t axis = -1;
    auto axis_v = MakeValue(axis);
    auto axis_node = func_builder_->NewFuncNode(axis_v, nullptr, InputType::kConstant);
    axis_node->set_abstract(axis_v->ToAbstract());
    std::vector<std::string> input_names = {"x"};
    std::vector<std::string> output_names = {"output"};
    auto expand_dims_primitive = func_builder_->NewPrimitive(
      kExpandDimsOpName, {{kAttrInputNames, MakeValue(input_names)}, {kAttrOutputNames, MakeValue(output_names)}});
    expand_dims_primitive->set_attr(kAttrAxis, axis_v);
    return func_builder_->EmitOp(expand_dims_primitive, {real_div_node, axis_node});
  }

  NodePtr CreateMulInput(const NodePtrList &inputs, const NodePtr &dout, NodePtrList *softmax_node_outputs) {
    MS_EXCEPTION_IF_NULL(softmax_node_outputs);
    auto one_hot_node = CreateOneHot(inputs);
    auto softmax_node = CreateSoftmaxCrossEntropyWithLogits(inputs, one_hot_node);
    CreateMultipleOutputsOfAnfNode(softmax_node, opt::kSoftmaxCrossEntropyWithLogitsOutputNum, softmax_node_outputs);
    auto tile_node = CreateTile(inputs, dout);
    NodePtr real_div_node;
    if (tile_node == nullptr) {
      real_div_node = CreateRealDiv(dout);
    } else {
      real_div_node = CreateRealDiv(tile_node);
    }
    auto expand_dims_node = CreateExpandDims(real_div_node);
    return expand_dims_node;
  }

  int64_t batch_size_{0};
  int64_t depth_{0};
};

size_t SplitTupleInputs(autograd::FuncBuilder *func_builder, const NodePtr &input, NodePtrList *plant_inputs) {
  MS_EXCEPTION_IF_NULL(func_builder);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(plant_inputs);
  MS_EXCEPTION_IF_NULL(input->Value());
  auto input_abs = input->abstract();
  auto value_seq = input->Value()->cast<ValueSequencePtr>()->value();
  auto abs_seq = input_abs->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(abs_seq);
  size_t input_size = value_seq.size();
  for (size_t i = 0; i < input_size; ++i) {
    const auto &value = value_seq[i];
    const auto &abs = abs_seq->elements()[i];
    (void)plant_inputs->emplace_back(func_builder->NewFuncNode(value, abs, input->input_type()));
  }
  return input_size;
}
}  // namespace

NodePtrList FuncPassForward::ConvertMakeTupleInputToDynamicInput(const PrimitivePtr &prim, const NodePtrList &inputs) {
  MS_EXCEPTION_IF_NULL(prim);
  if (!IsPrimitiveEquals(prim, prim::kPrimMakeTuple) &&
      std::any_of(inputs.begin(), inputs.end(),
                  [](const NodePtr &node) { return node->Value()->isa<abstract::AbstractSequence>(); })) {
    NodePtrList plant_inputs;
    std::vector<int64_t> dyn_input_sizes;
    for (const auto &input : inputs) {
      MS_EXCEPTION_IF_NULL(input->Value());
      if (input->Value()->isa<ValueSequence>()) {
        auto dyn_input_size = SplitTupleInputs(func_builder_, input, &plant_inputs);
        (void)dyn_input_sizes.emplace_back(dyn_input_size);
      } else {
        (void)plant_inputs.emplace_back(input);
        (void)dyn_input_sizes.emplace_back(-1);
      }
    }
    // If there is dynamic input, set the dyn_input_sizes as an attribute and update the inputs.
    if (std::any_of(dyn_input_sizes.begin(), dyn_input_sizes.end(), [](int64_t s) { return s >= 0; })) {
      prim->set_attr(kAttrDynInputSizes, MakeValue(dyn_input_sizes));
      MS_LOG(DEBUG) << "Change node to dynamic len " << prim->name();
    }
    return plant_inputs;
  }
  return inputs;
}

NodePtrList FuncPassForward::ConvertConstInputToAttr(const PrimitivePtr &prim, const NodePtrList &inputs) {
  MS_EXCEPTION_IF_NULL(prim);
  mindspore::HashSet<size_t> input_to_attr = {};
  PyNativeAlgo::Common::GetConstInputToAttr(prim, prim->name(), device_target_, false, &input_to_attr);
  if (input_to_attr.empty()) {
    return inputs;
  }
  const auto &input_names = prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return inputs;
  }
  return ChangeInputToAttr(prim, inputs, input_names, input_to_attr);
}

NodePtr FuncPassForward::BatchNormGradToBNInferGrad(const NodePtrList &inputs, bool is_scale_or_bias_grad) {
  if (device_target_ != kAscendDevice || is_scale_or_bias_grad) {
    return func_builder_->Emit(kBatchNormGradOpName, inputs);
  }
  constexpr size_t kIdxIsTraining = 6;
  auto is_training_opt = mindspore::ops::GetScalarValue<bool>(inputs[kIdxIsTraining]->Value());
  if (!is_training_opt.has_value()) {
    MS_LOG(DEBUG) << "Can not find Attr 'is_training' in training input";
    return func_builder_->Emit(kBatchNormGradOpName, inputs);
  }
  if (is_training_opt.value()) {
    MS_LOG(DEBUG) << "Attr 'is_training' is true, no need do fusion";
    return func_builder_->Emit(kBatchNormGradOpName, inputs);
  }

  auto bn_infer_grad_prim = func_builder_->NewPrimitive(kBNInferGradOpName);
  constexpr size_t kIdxGrads = 0;
  constexpr size_t kIdxScale = 2;
  constexpr size_t kIdxVariance = 4;
  constexpr size_t kIdxEpsilon = 7;
  NodePtrList new_inputs{inputs[kIdxGrads], inputs[kIdxScale], inputs[kIdxVariance], inputs[kIdxEpsilon]};

  auto epsilon_opt = mindspore::ops::GetScalarValue<pyfloat>(inputs[kIdxEpsilon]->Value());
  float epsilon{1e-5};
  if (epsilon_opt.has_value()) {
    epsilon = epsilon_opt.has_value() ? epsilon_opt.value() : 1e-5;
  } else {
    MS_LOG(ERROR) << "For BNInferGrad pass, failed to get attr epsilon, use default epsilon: 1e-5.";
  }
  bn_infer_grad_prim->set_attr(kAttrIsTraining, MakeValue(epsilon));
  bn_infer_grad_prim->set_attr(kAttrIsTraining, MakeValue(is_training_opt.value()));
  auto dx = func_builder_->EmitOp(bn_infer_grad_prim, new_inputs);
  return func_builder_->MakeTuple(
    {dx, func_builder_->OutZeros(inputs[kIdxScale]), func_builder_->OutZeros(inputs[kIdxScale])});
}

NodePtr FuncPassForward::GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR(const NodePtrList &inputs,
                                                                            const expander::DAttr &attrs,
                                                                            const NodePtr &out, const NodePtr &dout,
                                                                            bool is_graph_mode) {
  if (device_target_ != kAscendDevice) {
    auto grad = func_builder_->Emit(kSparseSoftmaxCrossEntropyWithLogitsOpName, inputs, attrs);
    if (is_graph_mode) {
      grad = func_builder_->Depend(grad, out);
    }
    grad = func_builder_->Emit(kMulOpName, {grad, dout});
    return grad;
  }

  // Use static class for create only once
  static auto sparse_softmax_cross_entropy_with_logits =
    std::make_shared<SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>();
  sparse_softmax_cross_entropy_with_logits->func_builder_ = func_builder_;
  return sparse_softmax_cross_entropy_with_logits->Run(inputs, dout);
}

NodePtrList FuncPassForward::PassForOpInput(const PrimitivePtr &prim, const NodePtrList &inputs) {
  MS_EXCEPTION_IF_NULL(func_builder_);
  if (prim == nullptr) {
    NodePtrList new_inputs = ConvertConstInputToAttr(prim, inputs);
    return ConvertMakeTupleInputToDynamicInput(prim, new_inputs);
  }
  return inputs;
}
}  // namespace bprop_pass
}  // namespace pynative
}  // namespace mindspore
