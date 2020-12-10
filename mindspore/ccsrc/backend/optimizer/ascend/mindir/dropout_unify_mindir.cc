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
#include <vector>
#include <memory>
#include <numeric>
#include <algorithm>
#include <functional>
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/log_adapter.h"

constexpr auto kKeepProb = "keep_prob";
constexpr auto kSeed0 = "Seed0";
constexpr auto kSeed1 = "Seed1";
constexpr auto kUint8BitSize = 8;

namespace mindspore::opt {
constexpr size_t kFloat16Len = 2;  // size of float16
namespace {
AnfNodePtr GetDropoutKeepProb(const AnfNodePtr &node, float *keep_prob) {
  MS_LOG(INFO) << "GetDropoutNodeInfo start.";
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(keep_prob);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!AnfAlgo::HasNodeAttr(kKeepProb, cnode) || !AnfAlgo::HasNodeAttr(kSeed0, cnode) ||
      !AnfAlgo::HasNodeAttr(kSeed1, cnode)) {
    MS_LOG(EXCEPTION) << "Dropout node does nothave attr: keep_prob or seed0 or seed1.";
  }
  *keep_prob = AnfAlgo::GetNodeAttr<float>(node, kKeepProb);
  MS_LOG(INFO) << "keep_prob: " << *keep_prob;
  // return dropout input. maybe tensor or pre cnode output
  return cnode->input(1);
}

ValueNodePtr CreateKeepPorbValueNode(const FuncGraphPtr &func_graph, const float &keep_prob, const TypePtr &dtype) {
  MS_LOG(INFO) << "CreateKeepPorbValueNode start.";
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<int64_t> keep_prob_shape = {};
  ShapeVector shape = {};
  auto keep_prob_tensor = std::make_shared<tensor::Tensor>(dtype->type_id(), keep_prob_shape);
  MS_EXCEPTION_IF_NULL(keep_prob_tensor);
  auto data_ptr = keep_prob_tensor->data_c();
  MS_EXCEPTION_IF_NULL(data_ptr);
  // keep_prob's datatype is same with input data
  if (dtype->type_id() == kNumberTypeFloat16) {
    float16 half_data = float16(keep_prob);
    auto ret_code = memcpy_s(data_ptr, kFloat16Len, &half_data, kFloat16Len);
    if (ret_code != 0) {
      MS_LOG(EXCEPTION) << "Failed to copy data into Tensor.";
    }
  } else {
    auto *val = reinterpret_cast<float *>(data_ptr);
    *val = keep_prob;
  }
  auto abstract = std::make_shared<abstract::AbstractTensor>(dtype, shape);
  auto keep_prob_value = kernel_graph->NewValueNode(abstract, keep_prob_tensor);
  MS_EXCEPTION_IF_NULL(keep_prob_value);
  kernel_graph->AddValueNodeToGraph(keep_prob_value);
  return keep_prob_value;
}

std::vector<int64_t> GetInputShape(const AnfNodePtr &node, const AnfNodePtr &dropout_input) {
  MS_LOG(INFO) << "GetInputShape start.";
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(dropout_input);
  std::vector<int64_t> shapes;
  if (dropout_input->isa<Parameter>()) {
    MS_LOG(INFO) << "Dropout input from parameter node.";
    // single test case
    auto dropout_input_value = dropout_input->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(dropout_input_value);
    MS_EXCEPTION_IF_NULL(dropout_input_value->Shape());
    auto shape = dropout_input_value->Shape()->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape);
    return shape->shape();
  } else if (dropout_input->isa<CNode>()) {
    MS_LOG(INFO) << "Dropout input from cnode.";
    auto dropout_input_node = dropout_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(dropout_input_node);
    auto shape_size_t = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
    std::transform(shape_size_t.begin(), shape_size_t.end(), std::back_inserter(shapes), SizeToLong);
    return shapes;
  } else {
    MS_LOG(ERROR) << "Dropout input is not parameter or cnode.";
    return {};
  }
}

ValueNodePtr CreateShapeValueNode(const FuncGraphPtr &func_graph, const std::vector<int64_t> &shape) {
  MS_LOG(INFO) << "CreateShapeValueNode start.";
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<ValuePtr> dim_values{};
  abstract::AbstractBasePtrList abs{};
  for (const auto &dim : shape) {
    dim_values.push_back(MakeValue(dim));
    abs.push_back(std::make_shared<abstract::AbstractScalar>(dim));
  }
  auto shape_value_tuple = std::make_shared<ValueTuple>(dim_values);
  MS_EXCEPTION_IF_NULL(shape_value_tuple);
  auto abstract = std::make_shared<abstract::AbstractTuple>(abs);
  MS_EXCEPTION_IF_NULL(abstract);
  auto shape_value = kernel_graph->NewValueNode(abstract, shape_value_tuple);
  MS_EXCEPTION_IF_NULL(shape_value);
  kernel_graph->AddValueNodeToGraph(shape_value);
  return shape_value;
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
  float keep_prob = 0;
  auto dropout_input = GetDropoutKeepProb(dropout_node, &keep_prob);
  auto dropout_dtype = AnfAlgo::GetOutputInferDataType(dropout_node, 0) == kNumberTypeFloat16 ? kFloat16 : kFloat32;
  auto keep_prob_value = CreateKeepPorbValueNode(func_graph, keep_prob, dropout_dtype);
  auto shape = GetInputShape(dropout_node, dropout_input);
  auto shape_value = CreateShapeValueNode(func_graph, shape);
  // CreateDropoutGenMask
  auto output_size = std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  output_size = output_size / kUint8BitSize;
  MS_LOG(INFO) << "Output_size: " << output_size;
  std::vector<AnfNodePtr> dropout_gen_mask_inputs{NewValueNode(std::make_shared<Primitive>(kDropoutGenMaskOpName)),
                                                  shape_value, keep_prob_value};
  CNodePtr dropout_gen_mask = func_graph->NewCNode(dropout_gen_mask_inputs);
  MS_EXCEPTION_IF_NULL(dropout_gen_mask);
  AnfAlgo::CopyNodeAttrs(node, dropout_gen_mask);
  ShapeVector dropout_gen_mask_output = {output_size};
  auto gen_mask_abstract = std::make_shared<abstract::AbstractTensor>(kUInt8, dropout_gen_mask_output);
  MS_EXCEPTION_IF_NULL(gen_mask_abstract);
  dropout_gen_mask->set_abstract(gen_mask_abstract);
  dropout_gen_mask->set_scope(node->scope());

  // CreateDropoutDoMask
  std::vector<AnfNodePtr> dropout_do_mask_inputs{NewValueNode(std::make_shared<Primitive>(kDropoutDoMaskOpName)),
                                                 dropout_input, dropout_gen_mask, keep_prob_value};
  auto dropout_do_mask = func_graph->NewCNode(dropout_do_mask_inputs);
  MS_EXCEPTION_IF_NULL(dropout_do_mask);
  ShapeVector dropout_do_mask_output = shape;
  auto do_mask_abstract = std::make_shared<abstract::AbstractTensor>(dropout_dtype, dropout_do_mask_output);
  dropout_do_mask->set_abstract(do_mask_abstract);
  dropout_do_mask->set_scope(node->scope());

  return dropout_do_mask;
}

const BaseRef DropoutGradUnifyMindIR::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Y = std::make_shared<Var>();
  MS_EXCEPTION_IF_NULL(X);
  MS_EXCEPTION_IF_NULL(Y);
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
  auto dropout_grad = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dropout_grad);
  auto tuple_getitem = dropout_grad->input(2);
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  auto tuple_getitem_cnode = tuple_getitem->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_getitem_cnode);
  auto dropout_node = tuple_getitem_cnode->input(1);
  MS_EXCEPTION_IF_NULL(dropout_node);
  float keep_prob = 0;
  auto dropout_input = GetDropoutKeepProb(dropout_node, &keep_prob);
  auto dropout_dtype = AnfAlgo::GetOutputInferDataType(dropout_node, 0) == kNumberTypeFloat16 ? kFloat16 : kFloat32;
  auto keep_prob_value = CreateKeepPorbValueNode(func_graph, keep_prob, dropout_dtype);
  auto shape = GetInputShape(dropout_node, dropout_input);
  auto shape_value = CreateShapeValueNode(func_graph, shape);
  // CreateDropoutGenMask
  auto output_size = std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  output_size = output_size / kUint8BitSize;
  MS_LOG(INFO) << "Output_size: " << output_size;
  std::vector<AnfNodePtr> dropout_gen_mask_inputs{NewValueNode(std::make_shared<Primitive>(kDropoutGenMaskOpName)),
                                                  shape_value, keep_prob_value};
  CNodePtr dropout_gen_mask = func_graph->NewCNode(dropout_gen_mask_inputs);
  MS_EXCEPTION_IF_NULL(dropout_gen_mask);
  AnfAlgo::CopyNodeAttrs(node, dropout_gen_mask);
  ShapeVector dropout_gen_mask_output = {output_size};
  auto gen_mask_abstract = std::make_shared<abstract::AbstractTensor>(kUInt8, dropout_gen_mask_output);
  MS_EXCEPTION_IF_NULL(gen_mask_abstract);
  dropout_gen_mask->set_abstract(gen_mask_abstract);
  dropout_gen_mask->set_scope(dropout_node->scope());
  //  AnfAlgo::CopyNodeAttrs(node, dropout_gen_mask);

  // CreateDropoutDoMask-forward
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto iter = node_users.find(dropout_node);
  if (iter != node_users.end()) {
    for (auto &node_index : iter->second) {
      // Dropout has two outputs, so output node is tuple_getitem
      auto tuple_getitem_cnode2 = node_index.first->cast<CNodePtr>();
      // check if Dropout's first output, which is used by forward, is used.
      auto getitem_index = GetValue<int64_t>(tuple_getitem_cnode2->input(2)->cast<ValueNodePtr>()->value());
      if (getitem_index == 0) {
        std::vector<AnfNodePtr> dropout_do_mask1_inputs{NewValueNode(std::make_shared<Primitive>(kDropoutDoMaskOpName)),
                                                        dropout_input, dropout_gen_mask, keep_prob_value};
        auto dropout_do_mask1 = func_graph->NewCNode(dropout_do_mask1_inputs);
        MS_EXCEPTION_IF_NULL(dropout_do_mask1);
        ShapeVector dropout_do_mask1_output = shape;
        auto do_mask_abstract1 = std::make_shared<abstract::AbstractTensor>(dropout_dtype, dropout_do_mask1_output);
        dropout_do_mask1->set_abstract(do_mask_abstract1);
        dropout_do_mask1->set_scope(dropout_node->scope());
        (void)manager->Replace(tuple_getitem_cnode2, dropout_do_mask1);
        break;
      }
    }
  }

  // CreateDropoutDoMask-backward
  if (equiv->find(grad_input_) == equiv->end()) {
    MS_LOG(EXCEPTION) << "Can not find grad_input in this pattern.";
  }
  auto grad_input = utils::cast<AnfNodePtr>((*equiv)[grad_input_]);
  std::vector<AnfNodePtr> dropout_do_mask2_inputs{NewValueNode(std::make_shared<Primitive>(kDropoutDoMaskOpName)),
                                                  grad_input, dropout_gen_mask, keep_prob_value};
  auto dropout_do_mask2 = func_graph->NewCNode(dropout_do_mask2_inputs);
  MS_EXCEPTION_IF_NULL(dropout_do_mask2);
  ShapeVector dropout_do_mask2_output = shape;
  auto do_mask_abstract2 = std::make_shared<abstract::AbstractTensor>(dropout_dtype, dropout_do_mask2_output);
  dropout_do_mask2->set_abstract(do_mask_abstract2);
  dropout_do_mask2->set_scope(node->scope());

  return dropout_do_mask2;
}
}  // namespace mindspore::opt
