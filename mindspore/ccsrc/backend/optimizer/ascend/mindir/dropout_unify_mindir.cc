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

#include "backend/optimizer/ascend/mindir/dropout_unify_mindir.h"
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <functional>
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/log_adapter.h"

/*
    DropoutGenMask：
    attr: seed0 seed1:
    input:  1.shape <>;
            2. keep_prob: type base on inputx type, if x in float/float16, then use this type, else use float16;
    output: shape: (count + 127) % 128 * 16
 */
namespace mindspore::opt {
namespace {
constexpr auto kKeepProb = "keep_prob";
constexpr auto kSeed0 = "Seed0";
constexpr auto kSeed1 = "Seed1";
constexpr auto kUint8BitSize = 8;
constexpr int64_t kMaskAlignNum = 128;
constexpr int64_t kMaskMultiNum = 16;
constexpr size_t kDropoutGradInputTensorNum = 2;
constexpr size_t kFloat16Len = 2;  // size of float16
constexpr size_t kInt64Len = 8;    // size of int64

TypeId GetInputXDataType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto dropout_input_type = AnfAlgo::GetPrevNodeOutputInferDataType(node, 0);
  if (dropout_input_type != kNumberTypeFloat32 && dropout_input_type != kNumberTypeFloat &&
      dropout_input_type != kNumberTypeFloat16) {
    dropout_input_type = kNumberTypeFloat16;
  }
  MS_LOG(INFO) << "Dropout input data type: " << TypeIdLabel(dropout_input_type);
  return dropout_input_type;
}

std::vector<int64_t> GetInputXShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<int64_t> shapes;
  auto shape_size_t = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  std::transform(shape_size_t.begin(), shape_size_t.end(), std::back_inserter(shapes), SizeToLong);
  return shapes;
}

ValueNodePtr CreateKeepPorbValueNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, TypeId type_id) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // Step1: get keep_prob
  if (!AnfAlgo::HasNodeAttr(kKeepProb, cnode)) {
    MS_LOG(EXCEPTION) << "Dropout node does not have attr: keep_prob.";
  }
  if (AnfAlgo::GetCNodeName(cnode) == kDropoutOpName) {
    if (!AnfAlgo::HasNodeAttr(kSeed0, cnode) || !AnfAlgo::HasNodeAttr(kSeed1, cnode)) {
      MS_LOG(EXCEPTION) << "Dropout node does not have attr: seed0 or seed1.";
    }
  }
  auto keep_prob = AnfAlgo::GetNodeAttr<float>(node, kKeepProb);
  MS_LOG(INFO) << "Keep_prob value: " << keep_prob;

  std::vector<int64_t> keep_prob_shape = {};
  auto keep_prob_tensor = std::make_shared<tensor::Tensor>(type_id, keep_prob_shape);
  MS_EXCEPTION_IF_NULL(keep_prob_tensor);
  auto data_ptr = keep_prob_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  // keep_prob's datatype is same with input data
  if (type_id == kNumberTypeFloat16) {
    auto *val16 = reinterpret_cast<float16 *>(data_ptr);
    *val16 = float16(keep_prob);
  } else {
    auto *val = reinterpret_cast<float *>(data_ptr);
    *val = keep_prob;
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), keep_prob_shape);
  auto keep_prob_value = kernel_graph->NewValueNode(abstract, keep_prob_tensor);
  MS_EXCEPTION_IF_NULL(keep_prob_value);
  kernel_graph->AddValueNodeToGraph(keep_prob_value);
  return keep_prob_value;
}

std::vector<int64_t> CalDropoutGenMaskOutput(const std::vector<int64_t> &shape) {
  auto output_size = std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  auto output_count = output_size / kMaskAlignNum;
  if (output_size % kMaskAlignNum != 0) {
    output_count++;
  }
  auto ret = output_count * kMaskMultiNum;
  MS_LOG(INFO) << "Output_size: " << ret;
  return {ret};
}

bool NeedUpdate(const CNodePtr &getitem_cnode) {
  MS_EXCEPTION_IF_NULL(getitem_cnode);
  MS_EXCEPTION_IF_NULL(getitem_cnode->input(kIndex2));
  auto index_vnode = getitem_cnode->input(kIndex2)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(index_vnode);
  auto index_value = index_vnode->value();
  MS_EXCEPTION_IF_NULL(index_value);
  auto index = GetValue<int64_t>(index_value);
  return index == 1;
}

CNodePtr CreateDynamicShapeCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node_input,
                                 const abstract::ShapePtr &input_shape) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_shape);
  std::vector<AnfNodePtr> dynamic_shape_inputs{NewValueNode(std::make_shared<Primitive>("DynamicShape")), node_input};
  CNodePtr dynamic_shape = func_graph->NewCNode(dynamic_shape_inputs);
  MS_EXCEPTION_IF_NULL(dynamic_shape);
  ShapeVector tensor_shp({static_cast<int64_t>(input_shape->shape().size())});
  auto dynamic_shape_abstract =
    std::make_shared<abstract::AbstractTensor>(kInt64, std::make_shared<abstract::Shape>(tensor_shp));
  auto max_value = MakeValue(input_shape->max_shape());
  auto min_value = MakeValue(input_shape->min_shape());
  MS_EXCEPTION_IF_NULL(dynamic_shape_abstract);
  dynamic_shape_abstract->set_value_range(min_value, max_value);
  dynamic_shape->set_abstract(dynamic_shape_abstract);
  return dynamic_shape;
}

CNodePtr CreateDropoutGenMaskCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &dropout,
                                   const ValueNodePtr &keep_prob_value, const AnfNodePtr &dropout_input,
                                   const abstract::ShapePtr &input_shape) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dropout);
  MS_EXCEPTION_IF_NULL(input_shape);
  std::vector<AnfNodePtr> dropout_gen_mask_inputs{NewValueNode(std::make_shared<Primitive>(kDropoutGenMaskOpName))};
  if (input_shape->IsDynamic()) {
    CNodePtr dynamic_shape = CreateDynamicShapeCNode(func_graph, dropout_input, input_shape);
    dynamic_shape->set_scope(dropout->scope());
    dropout_gen_mask_inputs.push_back(dynamic_shape);
    dropout_gen_mask_inputs.push_back(keep_prob_value);
  } else {
    auto shape_value = CreateShapeValueNode(func_graph, input_shape->shape(), true);
    dropout_gen_mask_inputs.push_back(shape_value);
    dropout_gen_mask_inputs.push_back(keep_prob_value);
  }
  CNodePtr dropout_gen_mask = func_graph->NewCNode(dropout_gen_mask_inputs);
  MS_EXCEPTION_IF_NULL(dropout_gen_mask);

  std::shared_ptr<abstract::AbstractTensor> gen_mask_abstract;
  if (input_shape->IsDynamic()) {
    ShapeVector mask_shp = {abstract::Shape::SHP_ANY};
    ShapeVector mask_min_shp = CalDropoutGenMaskOutput(input_shape->min_shape());
    ShapeVector mask_max_shp = CalDropoutGenMaskOutput(input_shape->max_shape());
    auto gen_mask_shape = std::make_shared<abstract::Shape>(mask_shp, mask_min_shp, mask_max_shp);
    MS_EXCEPTION_IF_NULL(gen_mask_shape);
    gen_mask_abstract = std::make_shared<abstract::AbstractTensor>(kUInt8, gen_mask_shape);
  } else {
    auto gen_mask_shape = CalDropoutGenMaskOutput(input_shape->shape());
    gen_mask_abstract = std::make_shared<abstract::AbstractTensor>(kUInt8, gen_mask_shape);
  }
  MS_EXCEPTION_IF_NULL(gen_mask_abstract);
  dropout_gen_mask->set_abstract(gen_mask_abstract);
  dropout_gen_mask->set_scope(dropout->scope());
  AnfAlgo::CopyNodeAttrs(dropout, dropout_gen_mask);
  return dropout_gen_mask;
}

abstract::ShapePtr GetDropoutInputShape(const AnfNodePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  auto input_base_shape = input->Shape();
  MS_EXCEPTION_IF_NULL(input_base_shape);
  auto input_shape = input_base_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(input_shape);
  return input_shape;
}
}  // namespace

const BaseRef DropoutAndDropoutGradUnifyMindIR::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Y = std::make_shared<Var>();
  auto dropout_prim = std::make_shared<Primitive>(kDropoutOpName);
  auto tuple_getitem_prim = prim::kPrimTupleGetItem;
  auto dropout_grad_prim = std::make_shared<Primitive>(kDropoutGradOpName);
  MS_EXCEPTION_IF_NULL(dropout_prim);
  MS_EXCEPTION_IF_NULL(dropout_grad_prim);
  auto ref0 = VectorRef({dropout_prim, X});
  auto ref1 = VectorRef({tuple_getitem_prim, ref0, Y});
  return VectorRef({dropout_grad_prim, grad_input_, ref1});
}

const AnfNodePtr DropoutAndDropoutGradUnifyMindIR::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                           const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto dropout_grad_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dropout_grad_cnode);
  auto getitem1_node = dropout_grad_cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(getitem1_node);
  auto getitem1_cnode = getitem1_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(getitem1_cnode);
  auto dropout_node = getitem1_cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(dropout_node);
  auto dropout_cnode = dropout_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dropout_cnode);

  auto inputx_type_id = GetInputXDataType(dropout_node);
  auto keep_prob_value = CreateKeepPorbValueNode(func_graph, dropout_node, inputx_type_id);

  auto dropout_input = dropout_cnode->input(kIndex1);
  auto input_shape = GetDropoutInputShape(dropout_input);
  // CreateDropoutGenMask
  auto dropout_gen_mask =
    CreateDropoutGenMaskCNode(func_graph, dropout_node, keep_prob_value, dropout_input, input_shape);
  // CreateDropoutDoMask-forward
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto iter = node_users.find(dropout_node);
  CNodePtr dropout_do_mask1 = nullptr;
  if (iter != node_users.end()) {
    for (auto &node_index : iter->second) {
      auto used_node = node_index.first;
      MS_EXCEPTION_IF_NULL(used_node);
      if (AnfAlgo::CheckPrimitiveType(used_node, prim::kPrimTupleGetItem)) {
        // check if Dropout's first output, which is used by forward, is used
        if (AnfAlgo::GetTupleGetItemOutIndex(used_node->cast<CNodePtr>()) == 0) {
          // if Dropout's first output is used, create forward DropoutDoMask
          std::vector<AnfNodePtr> dropout_do_mask1_inputs{
            NewValueNode(std::make_shared<Primitive>(kDropoutDoMaskOpName)), dropout_input, dropout_gen_mask,
            keep_prob_value};
          dropout_do_mask1 = func_graph->NewCNode(dropout_do_mask1_inputs);
          MS_EXCEPTION_IF_NULL(dropout_do_mask1);
          auto do_mask_abstract1 =
            std::make_shared<abstract::AbstractTensor>(TypeIdToType(inputx_type_id), input_shape);
          dropout_do_mask1->set_abstract(do_mask_abstract1);
          dropout_do_mask1->set_scope(dropout_node->scope());
          (void)manager->Replace(used_node, dropout_do_mask1);
          break;
        }
      }
    }
  }

  // CreateDropoutDoMask-backward
  if (equiv->find(grad_input_) == equiv->end()) {
    MS_LOG(EXCEPTION) << "Can not find grad_input in this pattern.";
  }
  auto dropout_grad_input = utils::cast<AnfNodePtr>((*equiv)[grad_input_]);
  std::vector<AnfNodePtr> dropout_do_mask_inputs{NewValueNode(std::make_shared<Primitive>(kDropoutDoMaskOpName)),
                                                 dropout_grad_input, dropout_gen_mask, keep_prob_value};
  auto dropout_do_mask = func_graph->NewCNode(dropout_do_mask_inputs);
  MS_EXCEPTION_IF_NULL(dropout_do_mask);
  auto do_mask_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(inputx_type_id), input_shape);
  dropout_do_mask->set_abstract(do_mask_abstract);
  dropout_do_mask->set_scope(node->scope());

  return dropout_do_mask;
}

const BaseRef DropoutUnifyMindIR0::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Y = std::make_shared<Var>();
  auto prim = std::make_shared<Primitive>(kDropoutOpName);
  auto ref = VectorRef({prim, X});
  return VectorRef({prim::kPrimTupleGetItem, ref, Y});
}

const AnfNodePtr DropoutUnifyMindIR0::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto tuple_cnode = node->cast<CNodePtr>();
  CheckCNodeInputSize(tuple_cnode, kTupleGetItemInputTensorNum);
  if (!NeedUpdate(tuple_cnode)) {
    return nullptr;
  }

  auto dropout_node = tuple_cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(dropout_node);
  auto inputx_type_id = GetInputXDataType(dropout_node);
  auto keep_prob_value = CreateKeepPorbValueNode(func_graph, dropout_node, inputx_type_id);

  auto dropout_cnode = dropout_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dropout_cnode);
  auto dropout_input = dropout_cnode->input(kIndex1);
  auto input_shape = GetDropoutInputShape(dropout_input);

  // CreateDropoutGenMask
  auto dropout_gen_mask =
    CreateDropoutGenMaskCNode(func_graph, dropout_node, keep_prob_value, dropout_input, input_shape);
  // CreateDropoutDoMask
  std::vector<AnfNodePtr> dropout_do_mask_inputs{NewValueNode(std::make_shared<Primitive>(kDropoutDoMaskOpName)),
                                                 dropout_input, dropout_gen_mask, keep_prob_value};
  auto dropout_do_mask = func_graph->NewCNode(dropout_do_mask_inputs);
  MS_EXCEPTION_IF_NULL(dropout_do_mask);
  auto do_mask_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(inputx_type_id), input_shape);
  dropout_do_mask->set_abstract(do_mask_abstract);
  dropout_do_mask->set_scope(node->scope());

  // make tuple to replace dropout
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple), dropout_do_mask, dropout_gen_mask};
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(dropout_node, make_tuple);

  tuple_cnode->set_abstract(dropout_gen_mask->abstract());
  return tuple_cnode;
}

const BaseRef DropoutUnifyMindIR1::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  return VectorRef({prim::kPrimDropout, X});
}

const AnfNodePtr DropoutUnifyMindIR1::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto dropout_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dropout_node);

  auto inputx_type_id = GetInputXDataType(dropout_node);
  auto keep_prob_value = CreateKeepPorbValueNode(func_graph, dropout_node, inputx_type_id);

  CheckCNodeInputSize(dropout_node, kDropoutInputTensorNum);
  auto dropout_input = dropout_node->input(kIndex1);
  auto input_shape = GetDropoutInputShape(dropout_input);
  // CreateDropoutGenMask
  auto dropout_gen_mask =
    CreateDropoutGenMaskCNode(func_graph, dropout_node, keep_prob_value, dropout_input, input_shape);
  // CreateDropoutDoMask
  std::vector<AnfNodePtr> dropout_do_mask_inputs{NewValueNode(std::make_shared<Primitive>(kDropoutDoMaskOpName)),
                                                 dropout_input, dropout_gen_mask, keep_prob_value};
  auto dropout_do_mask = func_graph->NewCNode(dropout_do_mask_inputs);
  MS_EXCEPTION_IF_NULL(dropout_do_mask);
  auto do_mask_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(inputx_type_id), input_shape);
  dropout_do_mask->set_abstract(do_mask_abstract);
  dropout_do_mask->set_scope(node->scope());

  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple), dropout_do_mask, dropout_gen_mask};
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}

const BaseRef DropoutGradUnifyMindIR::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Y = std::make_shared<Var>();
  auto dropout_grad_prim = std::make_shared<Primitive>(kDropoutGradOpName);
  return VectorRef({dropout_grad_prim, X, Y});
}

const AnfNodePtr DropoutGradUnifyMindIR::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto dropout_grad_cnode = node->cast<CNodePtr>();
  CheckCNodeInputSize(dropout_grad_cnode, kDropoutGradInputTensorNum);

  auto grad_input_type_id = GetInputXDataType(dropout_grad_cnode);
  auto grad_input_shape = GetInputXShape(dropout_grad_cnode);
  auto keep_prob_value = CreateKeepPorbValueNode(func_graph, dropout_grad_cnode, grad_input_type_id);

  // DropoutGrad may not in the same graph with Dropout in heterogeneous scene, and mask input which is a parameter
  // in that scene, need to be updated.
  auto mask_input = dropout_grad_cnode->input(kIndex2);
  if (mask_input->isa<Parameter>()) {
    // update abstract
    auto mask_abstract = mask_input->abstract();
    MS_EXCEPTION_IF_NULL(mask_abstract);
    auto mask_shape = CalDropoutGenMaskOutput(grad_input_shape);
    mask_abstract = std::make_shared<abstract::AbstractTensor>(kUInt8, mask_shape);
    mask_input->set_abstract(mask_abstract);
    // update kernel info
    auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
    kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{kNumberTypeUInt8});
    AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), mask_input.get());
  }

  // CreateDropoutDoMask
  auto grad_input = dropout_grad_cnode->input(kIndex1);
  std::vector<AnfNodePtr> dropout_do_mask_inputs{NewValueNode(std::make_shared<Primitive>(kDropoutDoMaskOpName)),
                                                 grad_input, mask_input, keep_prob_value};
  auto dropout_do_mask = func_graph->NewCNode(dropout_do_mask_inputs);
  MS_EXCEPTION_IF_NULL(dropout_do_mask);
  auto do_mask_abstract =
    std::make_shared<abstract::AbstractTensor>(TypeIdToType(grad_input_type_id), grad_input_shape);
  dropout_do_mask->set_abstract(do_mask_abstract);
  dropout_do_mask->set_scope(node->scope());
  return dropout_do_mask;
}
}  // namespace mindspore::opt
