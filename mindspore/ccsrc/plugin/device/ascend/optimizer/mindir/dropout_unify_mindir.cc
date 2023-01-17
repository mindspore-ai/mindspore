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

#include "plugin/device/ascend/optimizer/mindir/dropout_unify_mindir.h"
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <functional>
#include <algorithm>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/log_adapter.h"
#include "runtime/device/ms_device_shape_transfer.h"

/*
    DropoutGenMask：
    attr: seed0 seed1:
    input:  1.shape <>;
            2. keep_prob: type base on inputx type, if x in float/float16, then use this type, else use float16;
    output: shape: (count + 127) % 128 * 16
 */
namespace mindspore::opt {
namespace {
constexpr auto kAttrKeepProb = "keep_prob";
constexpr auto kAttrMicro = "micro";
constexpr auto kAttrSeed0 = "Seed0";
constexpr auto kAttrSeed1 = "Seed1";
constexpr auto kUint8BitSize = 8;
constexpr int64_t kMaskAlignNum = 128;
constexpr int64_t kMaskMultiNum = 16;
constexpr int64_t kV3ShapeLimitSize = 1 << 30;
constexpr size_t kDropoutGradInputTensorNum = 2;
constexpr size_t kFloat16Len = 2;  // size of float16
constexpr size_t kInt64Len = 8;    // size of int64

TypeId GetInputXDataType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto dropout_input_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0);
  if (dropout_input_type != kNumberTypeFloat32 && dropout_input_type != kNumberTypeFloat &&
      dropout_input_type != kNumberTypeFloat16) {
    dropout_input_type = kNumberTypeFloat16;
  }
  MS_LOG(INFO) << "Dropout input data type: " << TypeIdLabel(dropout_input_type);
  return dropout_input_type;
}

ValueNodePtr CreateKeepPorbValueNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, TypeId type_id) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // Step1: get keep_prob
  if (!common::AnfAlgo::HasNodeAttr(kAttrKeepProb, cnode)) {
    MS_LOG(EXCEPTION) << "Dropout node does not have attr: keep_prob." << trace::DumpSourceLines(node);
  }
  if (common::AnfAlgo::GetCNodeName(cnode) == kDropoutOpName) {
    if (!common::AnfAlgo::HasNodeAttr(kAttrSeed0, cnode) || !common::AnfAlgo::HasNodeAttr(kAttrSeed1, cnode)) {
      MS_LOG(EXCEPTION) << "Dropout node does not have attr: seed0 or seed1." << trace::DumpSourceLines(node);
    }
  }
  auto keep_prob = common::AnfAlgo::GetNodeAttr<float>(node, kAttrKeepProb);
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

std::vector<int64_t> CalGenMaskOutputShape(const std::vector<int64_t> &shape) {
  auto output_size = std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  auto output_count = output_size / kMaskAlignNum;
  if (output_size % kMaskAlignNum != 0) {
    output_count++;
  }
  auto ret = output_count * kMaskMultiNum;
  MS_LOG(INFO) << "Output_size: " << ret;
  return {ret};
}

std::vector<int64_t> CalGenMaskV3OutputShape(const std::vector<int64_t> &shape, TypeId type) {
  // [*dim, M, N] -> [*dim, N/16, M/16, 16, 16] if M%16=0 and N%16=0
  constexpr auto cube_h_offset = 2;
  if (shape.size() >= cube_h_offset && shape[shape.size() - 1] % static_cast<int64_t>(kCubeSize) == 0 &&
      shape[shape.size() - cube_h_offset] % static_cast<int64_t>(kCubeSize) == 0) {
    auto fnz_shape = trans::TransShapeToDevice(shape, kOpFormat_FRAC_NZ, type);
    return fnz_shape;
  }
  return shape;
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

bool WhetherUseDropoutV3(const CNodePtr & /* dropout */, const abstract::ShapePtr & /* input_shape */) {
  // v3 will cause memory error
  return false;
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
  MS_EXCEPTION_IF_NULL(dynamic_shape_abstract);
  dynamic_shape->set_abstract(dynamic_shape_abstract);
  return dynamic_shape;
}

CNodePtr GetRecomputeDropoutGenMask(const FuncGraphPtr &func_graph, const CNodePtr &dropout) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dropout);
  auto recompute_id = GetValue<int64_t>(dropout->GetAttr(kAttrRecomputeId));
  const auto &node_list = TopoSort(func_graph->get_return());
  auto find_recompute_genmask = [recompute_id](const AnfNodePtr &node) {
    if (!node->isa<CNode>() || !IsOneOfPrimitiveCNode(node, {prim::kPrimDropoutGenMask, prim::kPrimDropoutGenMaskV3})) {
      return false;
    }
    auto recompute_id_val = node->cast<CNodePtr>()->GetAttr(kAttrRecomputeId);
    return recompute_id_val != nullptr && GetValue<int64_t>(recompute_id_val) == recompute_id;
  };
  auto recompute_genmask = std::find_if(node_list.begin(), node_list.end(), find_recompute_genmask);
  if (recompute_genmask == node_list.end()) {
    MS_LOG(INFO) << "Can not find DropoutGenMask with recompute id " << recompute_id;
    return nullptr;
  }
  return (*recompute_genmask)->cast<CNodePtr>();
}

CNodePtr CreateDropoutGenMaskCNode(const FuncGraphPtr &func_graph, const CNodePtr &dropout,
                                   const ValueNodePtr &keep_prob_value, const abstract::ShapePtr &input_shape,
                                   const bool use_v3) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dropout);
  MS_EXCEPTION_IF_NULL(input_shape);
  std::vector<AnfNodePtr> dropout_gen_mask_inputs =
    use_v3 ? std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kDropoutGenMaskV3OpName))}
           : std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kDropoutGenMaskOpName))};
  if (input_shape->IsDynamic() || common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, dropout)) {
    CNodePtr dynamic_shape = CreateDynamicShapeCNode(func_graph, dropout->input(kIndex1), input_shape);
    dynamic_shape->set_scope(dropout->scope());
    dropout_gen_mask_inputs.push_back(dynamic_shape);
    dropout_gen_mask_inputs.push_back(keep_prob_value);
  } else {
    auto shape_value = CreateShapeValueNode(func_graph, input_shape->shape(), true);
    dropout_gen_mask_inputs.push_back(shape_value);
    dropout_gen_mask_inputs.push_back(keep_prob_value);
  }
  CNodePtr dropout_gen_mask = opt::NewCNode(dropout_gen_mask_inputs, func_graph, {dropout});
  MS_EXCEPTION_IF_NULL(dropout_gen_mask);
  if (dropout->HasPrimalAttr(kAttrFusion)) {
    dropout_gen_mask->AddPrimalAttr(kAttrFusion, dropout->GetPrimalAttr(kAttrFusion));
  }

  std::shared_ptr<abstract::AbstractTensor> gen_mask_abstract;
  if (input_shape->IsDynamic() || common::AnfAlgo::HasNodeAttr(kAttrMutableKernel, dropout)) {
    ShapeVector mask_shp = {abstract::Shape::kShapeDimAny};
    auto gen_mask_shape = std::make_shared<abstract::Shape>(mask_shp);
    MS_EXCEPTION_IF_NULL(gen_mask_shape);
    gen_mask_abstract = std::make_shared<abstract::AbstractTensor>(kUInt8, gen_mask_shape);
  } else {
    auto gen_mask_shape = use_v3 ? CalGenMaskV3OutputShape(input_shape->shape(), kNumberTypeUInt8)
                                 : CalGenMaskOutputShape(input_shape->shape());
    gen_mask_abstract = std::make_shared<abstract::AbstractTensor>(kUInt8, gen_mask_shape);
  }
  MS_EXCEPTION_IF_NULL(gen_mask_abstract);
  dropout_gen_mask->set_abstract(gen_mask_abstract);
  dropout_gen_mask->set_scope(dropout->scope());
  common::AnfAlgo::CopyNodeAttrs(dropout, dropout_gen_mask);
  if (dropout->HasPrimalAttr(kAttrMicro)) {
    dropout_gen_mask->AddPrimalAttr(kAttrMicro, dropout->GetPrimalAttr(kAttrMicro));
  }
  if (dropout->HasAttr(kAttrRecomputeId)) {
    dropout_gen_mask->AddAttr(kAttrRecomputeId, dropout->GetAttr(kAttrRecomputeId));
  }
  return dropout_gen_mask;
}

CNodePtr CreateDropoutDoMaskCNode(const FuncGraphPtr &func_graph, const CNodePtr &dropout,
                                  const std::vector<AnfNodePtr> &inputs, const abstract::AbstractBasePtr &abstract,
                                  const bool use_v3) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(dropout);
  std::vector<AnfNodePtr> dropout_do_mask_inputs =
    use_v3 ? std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kDropoutDoMaskV3OpName)), inputs[kIndex0],
                                     inputs[kIndex1]}
           : std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kDropoutDoMaskOpName)), inputs[kIndex0],
                                     inputs[kIndex1], inputs[kIndex2]};
  auto dropout_do_mask = opt::NewCNode(dropout_do_mask_inputs, func_graph, {dropout});
  MS_EXCEPTION_IF_NULL(dropout_do_mask);
  dropout_do_mask->set_abstract(abstract);
  dropout_do_mask->set_scope(dropout->scope());
  common::AnfAlgo::CopyNodeAttr(kAttrKeepProb, dropout, dropout_do_mask);

  std::vector<std::string> need_primal_attr = {kAttrMicro, kPrimalAttrUniqueId, kPrimalAttrForwardUniqueId};
  for (auto &primal_attr : need_primal_attr) {
    if (dropout->HasPrimalAttr(primal_attr)) {
      dropout_do_mask->AddPrimalAttr(primal_attr, dropout->GetPrimalAttr(primal_attr));
    }
  }
  return dropout_do_mask;
}

abstract::ShapePtr GetDropoutInputShape(const AnfNodePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  auto input_base_shape = input->Shape();
  MS_EXCEPTION_IF_NULL(input_base_shape);
  auto input_shape = input_base_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(input_shape);
  return input_shape;
}

bool NotDuplicatedDropout(const BaseRef &n) {
  if (utils::isa<CNodePtr>(n)) {
    auto in = utils::cast<CNodePtr>(n);
    MS_EXCEPTION_IF_NULL(in);
    if (IsPrimitiveCNode(in, prim::kPrimDropout)) {
      if (!in->HasAttr(kAttrDuplicated) || GetValue<bool>(in->GetAttr(kAttrDuplicated)) != true) {
        return true;
      }
    }
  }
  return false;
}

void UpdateReturnNode(const FuncGraphPtr &graph, const AnfNodePtr &origin_node, const AnfNodePtr &new_node) {
  // this pass maybe update the graph output abstract
  auto output = graph->output();
  MS_EXCEPTION_IF_NULL(output);
  if (!output->isa<CNode>()) {
    return;
  }
  auto cnode = output->cast<CNodePtr>();
  if (!common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimMakeTuple)) {
    return;
  }

  auto inputs_num = common::AnfAlgo::GetInputNum(cnode);
  std::vector<AnfNodePtr> make_tuple_inputs;
  std::vector<AbstractBasePtr> abstract_list;
  make_tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  bool flag = false;
  for (size_t index = 0; index < inputs_num; index++) {
    auto input_node = common::AnfAlgo::GetInputNode(cnode, index);
    if (input_node == origin_node) {
      flag = true;
      make_tuple_inputs.emplace_back(new_node);
      abstract_list.emplace_back(new_node->abstract());
      continue;
    }
    make_tuple_inputs.emplace_back(input_node);
    abstract_list.emplace_back(input_node->abstract());
  }
  if (!flag) {
    return;
  }

  auto g_output = graph->NewCNode(make_tuple_inputs);
  auto abstract = std::make_shared<abstract::AbstractTuple>(abstract_list);
  g_output->set_abstract(abstract);
  graph->set_output(g_output);
}
}  // namespace

const BaseRef DropoutAndDropoutGradUnifyMindIR::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Y = std::make_shared<Var>();
  auto dropout_var = std::make_shared<CondVar>(NotDuplicatedDropout);
  auto tuple_getitem_prim = prim::kPrimTupleGetItem;
  auto dropout_grad_prim = std::make_shared<Primitive>(kDropoutGradOpName);
  MS_EXCEPTION_IF_NULL(dropout_var);
  MS_EXCEPTION_IF_NULL(dropout_grad_prim);
  auto ref0 = VectorRef({dropout_var, X});
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
  auto use_v3 = WhetherUseDropoutV3(dropout_cnode, input_shape);
  // CreateDropoutGenMask
  auto dropout_gen_mask = CreateDropoutGenMaskCNode(func_graph, dropout_cnode, keep_prob_value, input_shape, use_v3);
  // CreateDropoutDoMask-forward
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto iter = node_users.find(dropout_node);
  if (iter != node_users.end()) {
    for (auto &node_index : iter->second) {
      auto used_node = node_index.first;
      MS_EXCEPTION_IF_NULL(used_node);
      if (common::AnfAlgo::CheckPrimitiveType(used_node, prim::kPrimTupleGetItem)) {
        // check if Dropout's first output, which is used by forward, is used
        if (common::AnfAlgo::GetTupleGetItemOutIndex(used_node->cast<CNodePtr>()) == 0) {
          // if Dropout's first output is used, create forward DropoutDoMask
          auto do_mask_abstract1 =
            std::make_shared<abstract::AbstractTensor>(TypeIdToType(inputx_type_id), input_shape);
          CNodePtr dropout_do_mask1 = CreateDropoutDoMaskCNode(
            func_graph, dropout_cnode, {dropout_input, dropout_gen_mask, keep_prob_value}, do_mask_abstract1, use_v3);
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
  auto do_mask_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(inputx_type_id), input_shape);
  auto dropout_do_mask = CreateDropoutDoMaskCNode(
    func_graph, dropout_grad_cnode, {dropout_grad_input, dropout_gen_mask, keep_prob_value}, do_mask_abstract, use_v3);

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
  auto use_v3 = WhetherUseDropoutV3(dropout_cnode, input_shape);

  // CreateDropoutGenMask
  CNodePtr dropout_gen_mask = nullptr;
  if (dropout_cnode->HasAttr(kAttrRecomputeId)) {
    dropout_gen_mask = GetRecomputeDropoutGenMask(func_graph, dropout_cnode);
  }
  if (dropout_gen_mask == nullptr) {
    dropout_gen_mask = CreateDropoutGenMaskCNode(func_graph, dropout_cnode, keep_prob_value, input_shape, use_v3);
  }
  // CreateDropoutDoMask
  auto do_mask_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(inputx_type_id), input_shape);
  auto dropout_do_mask = CreateDropoutDoMaskCNode(
    func_graph, dropout_cnode, {dropout_input, dropout_gen_mask, keep_prob_value}, do_mask_abstract, use_v3);

  // make tuple to replace dropout
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple), dropout_do_mask, dropout_gen_mask};
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  std::vector<AbstractBasePtr> abstract_list{dropout_do_mask->abstract(), dropout_gen_mask->abstract()};
  auto abstract = std::make_shared<abstract::AbstractTuple>(abstract_list);
  make_tuple->set_abstract(abstract);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  (void)manager->Replace(dropout_node, make_tuple);

  tuple_cnode->set_abstract(dropout_gen_mask->abstract());
  UpdateReturnNode(func_graph, node, tuple_cnode);
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
  auto dropout_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dropout_cnode);

  auto inputx_type_id = GetInputXDataType(dropout_cnode);

  CheckCNodeInputSize(dropout_cnode, kDropoutInputTensorNum);
  auto dropout_input = dropout_cnode->input(kIndex1);
  auto input_shape = GetDropoutInputShape(dropout_input);
  auto use_v3 = WhetherUseDropoutV3(dropout_cnode, input_shape);
  // CreateDropoutGenMask
  CNodePtr dropout_gen_mask = nullptr;
  if (dropout_cnode->HasAttr(kAttrRecomputeId)) {
    dropout_gen_mask = GetRecomputeDropoutGenMask(func_graph, dropout_cnode);
  }
  if (dropout_gen_mask == nullptr) {
    dropout_gen_mask = CreateDropoutGenMaskCNode(func_graph, dropout_cnode,
                                                 CreateKeepPorbValueNode(func_graph, dropout_cnode, inputx_type_id),
                                                 input_shape, use_v3);
  }
  // CreateDropoutDoMask
  auto do_mask_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(inputx_type_id), input_shape);
  auto dropout_do_mask = CreateDropoutDoMaskCNode(
    func_graph, dropout_cnode,
    {dropout_input, dropout_gen_mask, CreateKeepPorbValueNode(func_graph, dropout_cnode, inputx_type_id)},
    do_mask_abstract, use_v3);

  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple), dropout_do_mask, dropout_gen_mask};
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  std::vector<AbstractBasePtr> abstract_list{dropout_do_mask->abstract(), dropout_gen_mask->abstract()};
  auto abstract = std::make_shared<abstract::AbstractTuple>(abstract_list);
  make_tuple->set_abstract(abstract);
  UpdateReturnNode(func_graph, node, make_tuple);
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
  auto grad_input_shape = GetDropoutInputShape(dropout_grad_cnode->input(kIndex1));
  auto keep_prob_value = CreateKeepPorbValueNode(func_graph, dropout_grad_cnode, grad_input_type_id);
  auto use_v3 = WhetherUseDropoutV3(dropout_grad_cnode, grad_input_shape);

  // DropoutGrad may not in the same graph with Dropout in heterogeneous scene, and mask input which is a parameter
  // in that scene, need to be updated.
  auto mask_input = dropout_grad_cnode->input(kIndex2);
  if (mask_input->isa<Parameter>()) {
    // update abstract
    auto mask_abstract = mask_input->abstract();
    MS_EXCEPTION_IF_NULL(mask_abstract);
    auto grad_shape_vec = grad_input_shape->shape();
    auto mask_shape =
      use_v3 ? CalGenMaskV3OutputShape(grad_shape_vec, kNumberTypeUInt8) : CalGenMaskOutputShape(grad_shape_vec);
    mask_abstract = std::make_shared<abstract::AbstractTensor>(kUInt8, mask_shape);
    mask_input->set_abstract(mask_abstract);
    // update kernel info
    auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{kOpFormat_DEFAULT});
    kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{kNumberTypeUInt8});
    AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), mask_input.get());
  }

  // CreateDropoutDoMask
  auto do_mask_abstract =
    std::make_shared<abstract::AbstractTensor>(TypeIdToType(grad_input_type_id), grad_input_shape);
  auto dropout_do_mask = CreateDropoutDoMaskCNode(func_graph, dropout_grad_cnode,
                                                  {dropout_grad_cnode->input(kIndex1), mask_input, keep_prob_value},
                                                  do_mask_abstract, use_v3);
  return dropout_do_mask;
}
}  // namespace mindspore::opt
