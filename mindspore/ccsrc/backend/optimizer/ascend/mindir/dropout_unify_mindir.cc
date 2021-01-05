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

#include "backend/optimizer/ascend/mindir/dropout_unify_mindir.h"
#include <ops/all_ops.h>
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <algorithm>
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
  if (AnfAlgo::GetCNodePrimitive(cnode)->ToString() == kDropoutOpName) {
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
    auto half_data = float16(keep_prob);
    auto ret_code = memcpy_s(data_ptr, static_cast<size_t>(keep_prob_tensor->data().nbytes()), &half_data, kFloat16Len);
    if (ret_code != 0) {
      MS_LOG(EXCEPTION) << "Failed to copy data into Tensor.";
    }
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

ValueNodePtr CreateShapeValueNode(const FuncGraphPtr &func_graph, const std::vector<int64_t> &shape,
                                  bool is_pynative = false) {
  MS_LOG(INFO) << "CreateShapeValueNode start.";
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  ValuePtr shape_value = nullptr;
  AbstractBasePtr abstract = nullptr;
  if (is_pynative) {
    // pynative mode need to create tensor
    int64_t shape_dim = SizeToLong(shape.size());
    std::vector<int64_t> shape_vec_shape = {shape_dim};
    auto shape_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, shape_vec_shape);
    MS_EXCEPTION_IF_NULL(shape_tensor);
    auto data_ptr = shape_tensor->data_c();
    MS_EXCEPTION_IF_NULL(data_ptr);
    auto elem_num = shape.size() * kInt64Len;
    auto ret_code = memcpy_s(data_ptr, static_cast<size_t>(shape_tensor->data().nbytes()), &shape[0], elem_num);
    if (ret_code != 0) {
      MS_LOG(EXCEPTION) << "Failed to copy data into Tensor.";
    }
    shape_value = shape_tensor;
    abstract = std::make_shared<abstract::AbstractTensor>(kInt64, shape_vec_shape);
  } else {
    std::vector<ValuePtr> dim_values{};
    abstract::AbstractBasePtrList abs{};
    for (const auto &dim : shape) {
      dim_values.push_back(MakeValue(dim));
      abs.push_back(std::make_shared<abstract::AbstractScalar>(dim));
    }
    shape_value = std::make_shared<ValueTuple>(dim_values);
    abstract = std::make_shared<abstract::AbstractTuple>(abs);
  }
  MS_EXCEPTION_IF_NULL(shape_value);
  MS_EXCEPTION_IF_NULL(abstract);
  auto shape_value_node = kernel_graph->NewValueNode(abstract, shape_value);
  MS_EXCEPTION_IF_NULL(shape_value_node);
  kernel_graph->AddValueNodeToGraph(shape_value_node);
  return shape_value_node;
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
}  // namespace

const BaseRef DropoutUnifyMindIR::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Y = std::make_shared<Var>();
  auto prim = std::make_shared<Primitive>(kDropoutOpName);
  auto ref = VectorRef({prim, X});
  return VectorRef({prim::kPrimTupleGetItem, ref, Y});
}

const AnfNodePtr DropoutUnifyMindIR::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto tuple_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_cnode);
  auto dropout_node = tuple_cnode->input(1);
  MS_EXCEPTION_IF_NULL(dropout_node);

  auto inputx_type_id = GetInputXDataType(dropout_node);
  auto inputx_shape = GetInputXShape(dropout_node);
  auto shape_value = CreateShapeValueNode(func_graph, inputx_shape);
  auto keep_prob_value = CreateKeepPorbValueNode(func_graph, dropout_node, inputx_type_id);

  // CreateDropoutGenMask
  std::vector<AnfNodePtr> dropout_gen_mask_inputs{NewValueNode(std::make_shared<Primitive>(kDropoutGenMaskOpName)),
                                                  shape_value, keep_prob_value};
  CNodePtr dropout_gen_mask = func_graph->NewCNode(dropout_gen_mask_inputs);
  MS_EXCEPTION_IF_NULL(dropout_gen_mask);
  AnfAlgo::CopyNodeAttrs(node, dropout_gen_mask);
  auto output_shape = CalDropoutGenMaskOutput(inputx_shape);
  auto gen_mask_abstract = std::make_shared<abstract::AbstractTensor>(kUInt8, output_shape);
  MS_EXCEPTION_IF_NULL(gen_mask_abstract);
  dropout_gen_mask->set_abstract(gen_mask_abstract);
  dropout_gen_mask->set_scope(node->scope());

  // CreateDropoutDoMask
  MS_EXCEPTION_IF_NULL(dropout_node);
  auto dropout_cnode = dropout_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dropout_cnode);
  auto dropout_input = dropout_cnode->input(1);
  std::vector<AnfNodePtr> dropout_do_mask_inputs{NewValueNode(std::make_shared<Primitive>(kDropoutDoMaskOpName)),
                                                 dropout_input, dropout_gen_mask, keep_prob_value};
  auto dropout_do_mask = func_graph->NewCNode(dropout_do_mask_inputs);
  MS_EXCEPTION_IF_NULL(dropout_do_mask);
  auto do_mask_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(inputx_type_id), inputx_shape);
  dropout_do_mask->set_abstract(do_mask_abstract);
  dropout_do_mask->set_scope(node->scope());

  return dropout_do_mask;
}

const BaseRef DropoutGradUnifyMindIR::DefinePattern() const {
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

const AnfNodePtr DropoutGradUnifyMindIR::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto dropout_grad_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dropout_grad_cnode);
  auto getitem1_node = dropout_grad_cnode->input(2);
  MS_EXCEPTION_IF_NULL(getitem1_node);
  auto getitem1_cnode = getitem1_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(getitem1_cnode);
  auto dropout_node = getitem1_cnode->input(1);
  MS_EXCEPTION_IF_NULL(dropout_node);
  auto dropout_cnode = dropout_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dropout_cnode);

  auto inputx_type_id = GetInputXDataType(dropout_node);
  auto inputx_shape = GetInputXShape(dropout_node);
  auto shape_value = CreateShapeValueNode(func_graph, inputx_shape);
  auto keep_prob_value = CreateKeepPorbValueNode(func_graph, dropout_node, inputx_type_id);

  // CreateDropoutGenMask
  std::vector<AnfNodePtr> dropout_gen_mask_inputs{NewValueNode(std::make_shared<Primitive>(kDropoutGenMaskOpName)),
                                                  shape_value, keep_prob_value};
  CNodePtr dropout_gen_mask = func_graph->NewCNode(dropout_gen_mask_inputs);
  MS_EXCEPTION_IF_NULL(dropout_gen_mask);
  AnfAlgo::CopyNodeAttrs(node, dropout_gen_mask);
  auto output_shape = CalDropoutGenMaskOutput(inputx_shape);
  auto gen_mask_abstract = std::make_shared<abstract::AbstractTensor>(kUInt8, output_shape);
  MS_EXCEPTION_IF_NULL(gen_mask_abstract);
  dropout_gen_mask->set_abstract(gen_mask_abstract);
  dropout_gen_mask->set_scope(node->scope());

  // CreateDropoutDoMask-forward
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto iter = node_users.find(dropout_node);
  CNodePtr dropout_do_mask1 = nullptr;
  if (iter != node_users.end()) {
    for (auto &node_index : iter->second) {
      auto used_node = node_index.first;
      if (AnfAlgo::CheckPrimitiveType(used_node, prim::kPrimTupleGetItem)) {
        // check if Dropout's first output, which is used by forward, is used
        if (AnfAlgo::GetTupleGetItemOutIndex(used_node->cast<CNodePtr>()) == 0) {
          // if Dropout's first output is used, create forward DropoutDoMask
          auto dropout_input = dropout_cnode->input(1);
          std::vector<AnfNodePtr> dropout_do_mask1_inputs{
            NewValueNode(std::make_shared<Primitive>(kDropoutDoMaskOpName)), dropout_input, dropout_gen_mask,
            keep_prob_value};
          dropout_do_mask1 = func_graph->NewCNode(dropout_do_mask1_inputs);
          MS_EXCEPTION_IF_NULL(dropout_do_mask1);
          auto do_mask_abstract1 =
            std::make_shared<abstract::AbstractTensor>(TypeIdToType(inputx_type_id), inputx_shape);
          dropout_do_mask1->set_abstract(do_mask_abstract1);
          dropout_do_mask1->set_scope(dropout_node->scope());
          (void)manager->Replace(used_node, dropout_do_mask1);
          break;
        }
      }
    }
  }
  if (dropout_do_mask1 != nullptr) {
    // Dropout is used by ControlDepend in some situation, need to replace ControlDepend.
    auto &users = manager->node_users();
    iter = users.find(dropout_node);
    if (iter != users.end()) {
      for (auto &node_index : iter->second) {
        auto used_node = node_index.first;
        if (AnfAlgo::CheckPrimitiveType(used_node, prim::kPrimControlDepend)) {
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
  auto do_mask_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(inputx_type_id), inputx_shape);
  dropout_do_mask->set_abstract(do_mask_abstract);
  dropout_do_mask->set_scope(node->scope());

  return dropout_do_mask;
}

const BaseRef DropoutUnifyMindIRPynative::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  return VectorRef({prim::kPrimDropout, X});
}

const AnfNodePtr DropoutUnifyMindIRPynative::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                     const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto dropout_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dropout_node);

  auto inputx_type_id = GetInputXDataType(dropout_node);
  auto inputx_shape = GetInputXShape(dropout_node);
  auto shape_value = CreateShapeValueNode(func_graph, inputx_shape, true);
  auto keep_prob_value = CreateKeepPorbValueNode(func_graph, dropout_node, inputx_type_id);

  // CreateDropoutGenMask
  std::vector<AnfNodePtr> dropout_gen_mask_inputs{NewValueNode(std::make_shared<Primitive>(kDropoutGenMaskOpName)),
                                                  shape_value, keep_prob_value};
  CNodePtr dropout_gen_mask = func_graph->NewCNode(dropout_gen_mask_inputs);
  MS_EXCEPTION_IF_NULL(dropout_gen_mask);
  AnfAlgo::CopyNodeAttrs(node, dropout_gen_mask);
  auto output_shape = CalDropoutGenMaskOutput(inputx_shape);
  auto gen_mask_abstract = std::make_shared<abstract::AbstractTensor>(kUInt8, output_shape);
  MS_EXCEPTION_IF_NULL(gen_mask_abstract);
  dropout_gen_mask->set_abstract(gen_mask_abstract);
  dropout_gen_mask->set_scope(node->scope());

  // CreateDropoutDoMask
  MS_EXCEPTION_IF_NULL(dropout_node);
  auto dropout_cnode = dropout_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dropout_cnode);
  auto dropout_input = dropout_cnode->input(1);
  std::vector<AnfNodePtr> dropout_do_mask_inputs{NewValueNode(std::make_shared<Primitive>(kDropoutDoMaskOpName)),
                                                 dropout_input, dropout_gen_mask, keep_prob_value};
  auto dropout_do_mask = func_graph->NewCNode(dropout_do_mask_inputs);
  MS_EXCEPTION_IF_NULL(dropout_do_mask);
  auto do_mask_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(inputx_type_id), inputx_shape);
  dropout_do_mask->set_abstract(do_mask_abstract);
  dropout_do_mask->set_scope(node->scope());

  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple), dropout_do_mask, dropout_gen_mask};
  auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}

const BaseRef DropoutGradUnifyMindIRPynative::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Y = std::make_shared<Var>();
  auto dropout_grad_prim = std::make_shared<Primitive>(kDropoutGradOpName);
  return VectorRef({dropout_grad_prim, X, Y});
}

const AnfNodePtr DropoutGradUnifyMindIRPynative::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                         const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto dropout_grad_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dropout_grad_cnode);

  auto grad_input_type_id = GetInputXDataType(dropout_grad_cnode);
  auto grad_input_shape = GetInputXShape(dropout_grad_cnode);
  auto keep_prob_value = CreateKeepPorbValueNode(func_graph, dropout_grad_cnode, grad_input_type_id);

  // CreateDropoutDoMask
  auto grad_input = dropout_grad_cnode->input(1);
  auto mask_input = dropout_grad_cnode->input(2);
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
