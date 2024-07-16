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

#include "include/c_api/ms/node.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "c_api/src/helper.h"
#include "c_api/src/common.h"
#include "c_api/src/utils.h"
#include "base/base.h"
#include "ir/param_info.h"
#include "ir/anf.h"
#include "ir/scope.h"
#include "ir/func_graph_cloner.h"
#include "include/backend/optimizer/helper.h"
#include "kernel/oplib/oplib.h"
#include "kernel/oplib/opinfo.h"
#include "abstract/dshape.h"
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/pynative_utils.h"
#include "mindspore/core/ops/other_ops.h"

constexpr size_t firstInIdx = 1;
constexpr size_t secondInIdx = 2;
constexpr size_t switchInputNum = 3;
static const size_t maxMallocSize = GetMaxMallocSize();
NodeHandle MSNewOp(ResMgrHandle res_mgr, GraphHandle graph, const char *op_type, Handle const inputs[],
                   size_t input_num, const char *const *attr_names, ValueHandle attrs[], size_t attr_num) {
  if (res_mgr == nullptr || graph == nullptr || op_type == nullptr || inputs == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] or [op_type] or [inputs] is nullptr.";
    return nullptr;
  }
  // convert raw input pointer to source shared pointer
  auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
  if (res_fg == nullptr) {
    MS_LOG(ERROR) << "Get source pointer failed.";
    return nullptr;
  }
  auto res_mgr_ptr = reinterpret_cast<ResourceManager *>(res_mgr);
  std::vector<AnfNodePtr> cnode_inputs{};
  mindspore::AbstractBasePtrList abs_list{};
  auto prim = std::make_shared<PrimitiveImpl>(op_type);
  if (attr_names != nullptr && attrs != nullptr) {
    auto ret = OpSetAttrs(res_mgr, prim, attr_names, attrs, attr_num);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Op set attributes failed.";
      return nullptr;
    }
  }
  auto prim_node = mindspore::NewValueNode(prim);
  cnode_inputs.push_back(prim_node);
  CNodePtr cnode = nullptr;
  try {
    for (size_t i = 0; i < input_num; ++i) {
      auto input = GetSrcPtr<AnfNodePtr>(res_mgr, inputs[i]);
      MS_EXCEPTION_IF_NULL(input);
      if (input->isa<ParameterImpl>() && input->func_graph() != res_fg) {
        (void)res_fg->AddFreeVariable(input);
      }
      ConvertConstScalarInputToTensor(input);
      cnode_inputs.push_back(input);
      abs_list.push_back(input->abstract());
    }
    cnode = res_fg->NewCNodeInOrder(cnode_inputs);
    MS_EXCEPTION_IF_NULL(cnode);
    if (res_mgr_ptr->GetInfer()) {
      auto out_abs = OpInferShapeAndType(prim, abs_list);
      cnode->set_abstract(out_abs);
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph create new operator failed. Error info: " << e.what();
    return nullptr;
  }
  MS_LOG(INFO) << "Add Operator" << op_type;
  return GetRawPtr(res_mgr, cnode);
}

NodeHandle MSPackNodesTuple(ResMgrHandle res_mgr, GraphHandle graph, Handle const nodes[], size_t node_num) {
  if (res_mgr == nullptr || graph == nullptr || nodes == nullptr) {
    MS_LOG(ERROR) << "Input GraphHandle [res_mgr] or [graph] or [nodes] is nullptr.";
    return nullptr;
  }
  CNodePtr make_tuple_cnode = nullptr;
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    std::vector<AnfNodePtr> in_nodes{NewValueNode(mindspore::prim::kPrimMakeTuple)};
    mindspore::AbstractBasePtrList abs_list{};
    for (size_t i = 0; i < node_num; ++i) {
      auto in_node = GetSrcPtr<AnfNodePtr>(res_mgr, nodes[i]);
      MS_EXCEPTION_IF_NULL(in_node);
      in_nodes.push_back(in_node);
      ConvertConstScalarInputToTensor(in_node);
      abs_list.push_back(in_node->abstract());
    }
    make_tuple_cnode = res_fg->NewCNodeInOrder(in_nodes);
    MS_EXCEPTION_IF_NULL(make_tuple_cnode);
    make_tuple_cnode->set_abstract(std::make_shared<AbstractTupleImpl>(abs_list));
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph set output failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, make_tuple_cnode);
}

NodeHandle MSOpGetSpecOutput(ResMgrHandle res_mgr, GraphHandle graph, ConstNodeHandle op, size_t i) {
  if (res_mgr == nullptr || graph == nullptr || op == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] is nullptr.";
    return nullptr;
  }
  CNodePtr ret_node = nullptr;
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    auto cnode = GetSrcPtr<CNodePtr>(res_mgr, op);
    MS_EXCEPTION_IF_NULL(cnode);
    auto abs = cnode->abstract();
    if (abs == nullptr) {
      MS_LOG(ERROR) << "Input op's abstract is nullptr!";
      return nullptr;
    }
    if (abs->isa<mindspore::abstract::AbstractTuple>()) {
      auto branch_num = abs->cast<mindspore::abstract::AbstractTuplePtr>()->size();
      if (i >= branch_num) {
        MS_LOG(ERROR) << "Invalid output branch index, it should be less than " << branch_num << ", but got: " << i;
        return nullptr;
      }
      auto idx = mindspore::NewValueNode(mindspore::SizeToLong(i));
      auto abs_scalar = std::make_shared<mindspore::abstract::AbstractScalar>(mindspore::SizeToInt(i));
      idx->set_abstract(abs_scalar);
      ret_node = res_fg->NewCNodeInOrder({NewValueNode(mindspore::prim::kPrimTupleGetItem), cnode, idx});
      MS_EXCEPTION_IF_NULL(ret_node);
      ret_node->set_abstract(abs->cast<mindspore::abstract::AbstractTuplePtr>()->elements()[i]);
    } else {
      if (i >= 1) {
        MS_LOG(ERROR) << "Invalid output index. The op has only one output, so the output index should be 0, or you can"
                         " directly use this op as the output without calling this function, but got: "
                      << i;
        return nullptr;
      }
      MS_LOG(WARNING) << "The op has only one output, you can directly use this op as the output without calling this "
                         "function. Now the op itself is returned.";
      ret_node = cnode;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph get output failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, ret_node);
}

CNodePtr BuildSwitchStructure(ResMgrHandle res_mgr, GraphHandle graph, NodeHandle const switch_input[],
                              size_t input_num, bool set_fg_out) {
  MS_EXCEPTION_IF_NULL(res_mgr);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(switch_input);
  MS_EXCEPTION_IF_CHECK_FAIL(input_num == switchInputNum, "Switch's input number must be 3!");
  NodeHandle switch_op = MSNewOp(res_mgr, graph, "Switch", switch_input, input_num, NULL, NULL, 0);
  if (switch_op == nullptr) {
    MS_LOG(ERROR) << "Get Switch op failed!";
    return nullptr;
  }
  auto src_switch = GetSrcPtr<CNodePtr>(res_mgr, switch_op);
  MS_EXCEPTION_IF_NULL(src_switch);
  auto fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
  MS_EXCEPTION_IF_NULL(fg);
  CNodePtr switch_call = fg->NewCNodeInOrder({src_switch});
  MS_EXCEPTION_IF_NULL(switch_call);
  if (set_fg_out) {
    fg->set_output(switch_call);
  }
  auto first_node = GetSrcPtr<ValueNodePtr>(res_mgr, switch_input[firstInIdx]);
  MS_EXCEPTION_IF_NULL(first_node);
  auto second_node = GetSrcPtr<ValueNodePtr>(res_mgr, switch_input[secondInIdx]);
  MS_EXCEPTION_IF_NULL(second_node);
  // AddFuncGraphCNodeIndex is used to set cnode_index. A funcgraph's cnode_index is a list of pair
  // with pair-struct is (CNODE, index). The CNODE is in another funcgraph, who uses the funcgraph as its input.
  // for eg. if fg1's cnode A uses fg2 as A's first input, then fg2's conde_index is (A, 1)
  if (first_node->isa<ValueNodeImpl>()) {
    fg->AddValueNode(first_node);
    if (mindspore::IsValueNode<FuncGraphImpl>(first_node)) {
      auto used = mindspore::GetValueNode<FuncGraphPtr>(first_node);
      used->AddFuncGraphCNodeIndex(
        std::make_shared<mindspore::CNodeIndexPair>(std::make_pair(src_switch, firstInIdx + 1)));
      (void)fg->AddFuncGraphUsed(used);
    }
  }
  if (second_node->isa<ValueNodeImpl>()) {
    fg->AddValueNode(second_node);
    if (mindspore::IsValueNode<FuncGraphImpl>(second_node)) {
      auto used = mindspore::GetValueNode<FuncGraphPtr>(second_node);
      used->AddFuncGraphCNodeIndex(
        std::make_shared<mindspore::CNodeIndexPair>(std::make_pair(src_switch, secondInIdx + 1)));
      (void)fg->AddFuncGraphUsed(used);
    }
  }
  // Switch-call's abstract is equal to second branch.
  if (mindspore::IsValueNode<FuncGraphImpl>(second_node)) {
    auto sub_fg = mindspore::GetValueNode<FuncGraphPtr>(second_node);
    switch_call->set_abstract(sub_fg->output()->abstract());
  }
  return switch_call;
}

NodeHandle MSNewSwitch(ResMgrHandle res_mgr, GraphHandle graph, Handle cond, ConstGraphHandle true_br,
                       ConstGraphHandle false_br) {
  if (res_mgr == nullptr || graph == nullptr || cond == nullptr || true_br == nullptr || false_br == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] or [cond] or [true_br] or [false_br] is nullptr.";
    return nullptr;
  }
  try {
    auto src_cond = GetSrcPtr<BasePtr>(res_mgr, cond);
    MS_EXCEPTION_IF_NULL(src_cond);
    NodeHandle cond_raw_ptr = nullptr;
    if (src_cond->isa<FuncGraphImpl>()) {
      auto cond_graph = src_cond->cast<FuncGraphPtr>();
      MS_EXCEPTION_IF_NULL(cond_graph);
      auto cond_node = mindspore::NewValueNode(cond_graph);
      cond_node->set_abstract(cond_graph->ToAbstract());
      cond_raw_ptr = GetRawPtr(res_mgr, cond_node);
    } else {
      cond_raw_ptr = cond;
    }
    auto true_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, true_br);
    MS_EXCEPTION_IF_NULL(true_fg);
    auto true_node = mindspore::NewValueNode(true_fg);
    true_node->set_abstract(true_fg->ToAbstract());
    NodeHandle true_raw_ptr = GetRawPtr(res_mgr, true_node);

    auto false_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, false_br);
    MS_EXCEPTION_IF_NULL(false_fg);
    auto false_node = mindspore::NewValueNode(false_fg);
    false_node->set_abstract(false_fg->ToAbstract());
    NodeHandle false_raw_ptr = GetRawPtr(res_mgr, false_node);

    NodeHandle switch_input[] = {cond_raw_ptr, true_raw_ptr, false_raw_ptr};
    auto switch_call = BuildSwitchStructure(res_mgr, graph, switch_input, switchInputNum, false);
    MS_EXCEPTION_IF_NULL(switch_call);
    return GetRawPtr(res_mgr, switch_call);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "New Switch node failed. Error info: " << e.what();
    return nullptr;
  }
}

void HandleFVInWhileGraph(const FuncGraphPtr &main_fg, const FuncGraphPtr &body_fg, const FuncGraphPtr &after_fg) {
  std::vector<AnfNodePtr> fv_to_restore{};
  auto body_fvs = body_fg->free_variables();
  for (const auto &fv : body_fvs) {
    auto fv_node = fv.first;
    MS_EXCEPTION_IF_NULL(fv_node);
    if (fv_node->func_graph() != main_fg &&
        std::find(fv_to_restore.begin(), fv_to_restore.end(), fv_node) == fv_to_restore.end()) {
      fv_to_restore.push_back(fv_node);
    }
  }
  auto after_fvs = after_fg->free_variables();
  for (const auto &fv : after_fvs) {
    auto fv_node = fv.first;
    MS_EXCEPTION_IF_NULL(fv_node);
    if (fv_node->func_graph() != main_fg &&
        std::find(fv_to_restore.begin(), fv_to_restore.end(), fv_node) == fv_to_restore.end()) {
      fv_to_restore.push_back(fv_node);
    }
  }

  (void)mindspore::LiftingClone(main_fg);

  auto main_manager = Manage(main_fg);
  std::vector<AnfNodePtr> new_main_params{};
  auto main_params = main_fg->parameters();
  for (const auto &main_param : main_params) {
    auto src_main_param = main_param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(src_main_param);
    auto found_in_fv_list =
      find_if(fv_to_restore.begin(), fv_to_restore.end(), [&main_param](const AnfNodePtr &fv_param) {
        return !main_param->ToString().empty() && main_param->ToString() == fv_param->ToString();
      });
    if (found_in_fv_list != fv_to_restore.end()) {
      (void)main_manager->Replace(main_param, *found_in_fv_list);
    } else if (src_main_param->has_default()) {
      auto const_input = mindspore::NewValueNode(src_main_param->default_param());
      const_input->set_abstract(src_main_param->abstract());
      (void)main_manager->Replace(main_param, const_input);
    } else {
      new_main_params.push_back(main_param);
    }
  }
  main_fg->set_parameters(new_main_params);
}

NodeHandle MSNewWhile(ResMgrHandle res_mgr, GraphHandle graph, Handle cond, GraphHandle body_graph,
                      GraphHandle after_graph) {
  if (res_mgr == nullptr || graph == nullptr || cond == nullptr || body_graph == nullptr || after_graph == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] or [cond] or [body_graph] or [after_graph] is nullptr.";
    return nullptr;
  }
  try {
    auto src_cond = GetSrcPtr<BasePtr>(res_mgr, cond);
    MS_EXCEPTION_IF_NULL(src_cond);
    NodeHandle cond_raw_ptr = nullptr;
    GraphHandle cond_graph = nullptr;
    FuncGraphPtr src_cond_graph = nullptr;
    auto main_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    if (src_cond->isa<FuncGraphImpl>()) {
      cond_graph = cond;
      src_cond_graph = src_cond->cast<FuncGraphPtr>();
      MS_EXCEPTION_IF_NULL(src_cond_graph);
      auto cond_node = src_cond_graph->output();
      MS_EXCEPTION_IF_NULL(cond_node);
      cond_raw_ptr = GetRawPtr(res_mgr, cond_node);
    } else {
      auto cond_fg = std::make_shared<FuncGraphImpl>();
      MS_EXCEPTION_IF_NULL(cond_fg);
      cond_graph = GetRawPtr(res_mgr, cond_fg);
      MS_EXCEPTION_IF_NULL(cond_graph);
      src_cond_graph = GetSrcPtr<FuncGraphPtr>(res_mgr, cond_graph);
      MS_EXCEPTION_IF_NULL(src_cond_graph);
      (void)main_fg->AddFuncGraphUsed(src_cond_graph);
      if (src_cond->isa<CNodeImpl>()) {
        auto cond_node = src_cond->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cond_node);
        auto new_cond = src_cond_graph->NewCNodeInOrder(cond_node->inputs());
        MS_EXCEPTION_IF_NULL(new_cond);
        new_cond->set_abstract(cond_node->abstract());
        cond_raw_ptr = GetRawPtr(res_mgr, new_cond);
      } else {
        cond_raw_ptr = cond;
      }
    }

    auto body_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, body_graph);
    MS_EXCEPTION_IF_NULL(body_fg);
    auto body_node = mindspore::NewValueNode(body_fg);
    body_node->set_abstract(body_fg->ToAbstract());
    NodeHandle body_raw_ptr = GetRawPtr(res_mgr, body_node);

    auto after_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, after_graph);
    MS_EXCEPTION_IF_NULL(after_fg);
    auto after_node = mindspore::NewValueNode(after_fg);
    after_node->set_abstract(after_fg->ToAbstract());
    NodeHandle after_raw_ptr = GetRawPtr(res_mgr, after_node);

    NodeHandle switch_input[] = {cond_raw_ptr, body_raw_ptr, after_raw_ptr};
    (void)BuildSwitchStructure(res_mgr, cond_graph, switch_input, switchInputNum, true);

    // handle main graph call
    NodeHandle main_func_call = MSNewFuncCallNode(res_mgr, graph, cond_graph, nullptr, 0);
    auto src_call = GetSrcPtr<AnfNodePtr>(res_mgr, main_func_call);
    main_fg->set_output(src_call);

    // handle free parameters in while graphs
    HandleFVInWhileGraph(main_fg, body_fg, after_fg);

    // handle multi outputs in body graph
    auto sub_graph_node = mindspore::NewValueNode(src_cond_graph);
    sub_graph_node->set_abstract(src_cond_graph->ToAbstract());
    std::vector<AnfNodePtr> sub_input_nodes{sub_graph_node};
    auto body_out_node = body_fg->output();
    MS_EXCEPTION_IF_NULL(body_out_node);
    if (IsPrimitiveCNode(body_out_node, mindspore::prim::kPrimMakeTuple)) {
      auto body_out_cnode = body_out_node->cast<CNodePtr>();
      for (size_t i = 1; i < body_out_cnode->size(); i++) {
        sub_input_nodes.push_back(body_out_cnode->input(i));
      }
    } else {
      sub_input_nodes.push_back(body_out_node);
    }
    auto body_func_call = body_fg->NewCNodeInOrder(sub_input_nodes);
    MS_EXCEPTION_IF_NULL(src_cond_graph->output());
    MS_EXCEPTION_IF_NULL(body_func_call);
    body_func_call->set_abstract(src_cond_graph->output()->abstract());
    body_fg->set_output(body_func_call);
    return main_func_call;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "New While node failed. Error info: " << e.what();
    return nullptr;
  }
}

std::vector<BaseShapePtr> CustomOpInferShape(const CustomOpInfo &info, const std::vector<AbstractBasePtr> &input_args) {
  auto dyn_arr_deleter = [](int64_t **x, size_t dims) {
    std::for_each(x, x + dims, std::default_delete<int64_t[]>());
    delete[] x;
  };
  if (info.output_shapes != nullptr) {
    if (info.output_dims == nullptr) {
      MS_LOG(ERROR) << "Output dims must be given if output shapes are specified!";
      return {};
    }
    auto infer_shape = BuildShape(info.output_shapes, info.output_dims, info.output_num);
    return infer_shape;
  } else if (info.shape_infer_func != nullptr) {
    size_t input_num = info.input_num;
    size_t output_num = info.output_num;
    MS_ERROR_IF_TRUE_W_RET_N_LOG(input_num * sizeof(size_t) > maxMallocSize, {},
                                 "The input_num is too large for memory allocation.");
    MS_ERROR_IF_TRUE_W_RET_N_LOG(output_num * sizeof(size_t) > maxMallocSize, {},
                                 "The output_num is too large for memory allocation.");
    auto out_dims_arr = std::make_unique<size_t[]>(output_num);
    std::unique_ptr<int64_t *, std::function<void(int64_t **)>> out_shapes_arr(
      new (std::nothrow) int64_t *[output_num](), std::bind(dyn_arr_deleter, std::placeholders::_1, output_num));
    for (size_t i = 0; i < output_num; i++) {
      (out_shapes_arr.get())[i] = new int64_t[MAX_DIMS];
    }
    auto in_dims_arr = std::make_unique<size_t[]>(input_num);
    std::unique_ptr<int64_t *, std::function<void(int64_t **)>> in_shapes_arr(
      new (std::nothrow) int64_t *[input_num](), std::bind(dyn_arr_deleter, std::placeholders::_1, input_num));
    for (size_t i = 0; i < input_num; i++) {
      auto in_shape = input_args[i]->BuildShape();
      MS_EXCEPTION_IF_NULL(in_shape);
      auto in_shape_ptr = in_shape->cast<ShapePtr>();
      MS_EXCEPTION_IF_NULL(in_shape_ptr);
      auto in_shape_vec = in_shape_ptr->shape();
      auto in_shape_dim = in_shape_vec.size();
      in_dims_arr[i] = in_shape_dim;
      MS_ERROR_IF_TRUE_W_RET_N_LOG(in_shape_dim * sizeof(size_t) > maxMallocSize, {},
                                   "The in_shape_dim is too large for memory allocation.");
      (in_shapes_arr.get())[i] = new int64_t[in_shape_dim];
      for (size_t j = 0; j < in_shape_dim; j++) {
        (in_shapes_arr.get())[i][j] = in_shape_vec[j];
      }
    }
    auto ret = info.shape_infer_func(in_shapes_arr.get(), in_dims_arr.get(), input_num, out_shapes_arr.get(),
                                     out_dims_arr.get(), output_num);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Failed to call the shape infer function of custom op!";
      return {};
    }
    auto infer_shape = BuildShape(out_shapes_arr.get(), out_dims_arr.get(), output_num);
    return infer_shape;
  } else {
    MS_LOG(ERROR) << "Either output shape or output shape infer function must be specified!";
    return {};
  }
}

std::vector<TypePtr> CustomOpInferType(const CustomOpInfo &info, const std::vector<AbstractBasePtr> &input_args) {
  if (info.output_dtypes != nullptr) {
    auto infer_dtype = BuildType(info.output_dtypes, info.output_num);
    return infer_dtype;
  } else if (info.shape_infer_func != nullptr) {
    size_t input_num = info.input_num;
    size_t output_num = info.output_num;
    auto in_dtypes_arr = std::make_unique<DataTypeC[]>(input_num);
    auto out_dtypes_arr = std::make_unique<DataTypeC[]>(output_num);
    for (size_t i = 0; i < input_num; i++) {
      auto in_type = input_args[i]->BuildType();
      MS_EXCEPTION_IF_NULL(in_type);
      auto real_type = in_type;
      if (in_type->isa<TensorTypeImpl>()) {
        auto tensor_type = in_type->cast<TensorTypePtr>();
        real_type = tensor_type->element();
      }
      auto in_type_id = (enum DataTypeC)(real_type->type_id());
      in_dtypes_arr[i] = in_type_id;
    }
    STATUS ret = info.dtype_infer_func(in_dtypes_arr.get(), input_num, out_dtypes_arr.get(), output_num);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Failed to call the dtype infer function of custom op!";
      return {};
    }
    auto infer_dtype = BuildType(out_dtypes_arr.get(), output_num);
    return infer_dtype;
  } else {
    MS_LOG(ERROR) << "Either output dtype or output dtype infer function must be specified!";
    return {};
  }
}

NodeHandle MSNewCustomOp(ResMgrHandle res_mgr, GraphHandle graph, Handle const inputs[], size_t input_num,
                         CustomOpInfo info) {
  if (res_mgr == nullptr || graph == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] is nullptr.";
    return nullptr;
  }
  MS_ERROR_IF_TRUE_W_RET_N_LOG(input_num != info.input_num, nullptr,
                               "Input node number is not matched with the input number specified in custom op info.");
  auto ret = CheckCustomOpInfo(info);
  MS_ERROR_IF_TRUE_W_RET_N_LOG(ret != RET_OK, nullptr, "Invalid custom op info.");
  try {
    auto res_mgr_ptr = reinterpret_cast<ResourceManager *>(res_mgr);
    auto org_infer = res_mgr_ptr->GetInfer();
    res_mgr_ptr->SetInfer(false);
    NodeHandle custom_op =
      MSNewOp(res_mgr, graph, "Custom", inputs, info.input_num, info.attr_names, info.attr_values, info.attr_num);
    MS_ERROR_IF_TRUE_W_RET_N_LOG(custom_op == nullptr, nullptr, "Create Custom op failed!");
    res_mgr_ptr->SetInfer(org_infer);
    // Supplement necessary attributes
    ret = MSOpSetAttrString(res_mgr, custom_op, mindspore::kAttrFuncType, info.func_type);
    MS_ERROR_IF_TRUE_W_RET_N_LOG(ret != RET_OK, nullptr, "Custom op set func type attribute failed.");
    ret = MSOpSetAttrString(res_mgr, custom_op, mindspore::kAttrFuncName, info.func_name);
    MS_ERROR_IF_TRUE_W_RET_N_LOG(ret != RET_OK, nullptr, "Custom op set func name attribute failed.");
    // Build json object
    nlohmann::json json_obj = ConvertOpInfoToJson(info);
    MS_ERROR_IF_TRUE_W_RET_N_LOG(json_obj.empty(), nullptr, "Failed to convert op info to json.");
    // Create op info and set info map
    auto op_name = json_obj.at(mindspore::kernel::kOpName).get<std::string>();
    auto imply_type = json_obj.at(mindspore::kernel::kImplyType).get<std::string>();
    std::string func_name = info.func_name;
    std::string target_name = info.target;
    auto iter = mindspore::kernel::kImplyTypeStrToEnumMap.find(imply_type);
    if (iter == mindspore::kernel::kImplyTypeStrToEnumMap.end()) {
      MS_LOG(ERROR) << "Not support imply_type: " << imply_type;
      return nullptr;
    }
    auto op_info = mindspore::kernel::OpLib::DecodeOpInfo(json_obj, iter->second, "");
    if (op_info == nullptr) {
      MS_LOG(ERROR) << "Decode op info failed: func_name: " << func_name << " imply_type " << imply_type;
      return nullptr;
    }
    op_info->set_processor(imply_type);
    auto key = op_name + imply_type;
    auto &op_infos = mindspore::kernel::OpLib::GetOpInfoMap();
    (void)op_infos[iter->second].insert(std::pair<std::string, mindspore::kernel::OpInfoPtr>(key, op_info));
    // Infer shape and type
    mindspore::AbstractBasePtrList abs_list{};
    for (size_t i = 0; i < input_num; ++i) {
      auto in_node = GetSrcPtr<AnfNodePtr>(res_mgr, inputs[i]);
      MS_EXCEPTION_IF_NULL(in_node);
      abs_list.push_back(in_node->abstract());
    }
    auto infer_shape = CustomOpInferShape(info, abs_list);
    auto infer_type = CustomOpInferType(info, abs_list);
    AbstractBasePtr custom_abs = BuildAbstract(infer_shape, infer_type);
    MS_EXCEPTION_IF_NULL(custom_abs);
    auto src_op = GetSrcPtr<CNodePtr>(res_mgr, custom_op);
    MS_EXCEPTION_IF_NULL(src_op);
    src_op->set_abstract(custom_abs);
    return custom_op;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get custom op failed. Error info: " << e.what();
    return nullptr;
  }
}

NodeHandle MSOpGetInput(ResMgrHandle res_mgr, ConstNodeHandle op, size_t i) {
  if (res_mgr == nullptr || op == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] is nullptr.";
    return nullptr;
  }
  mindspore::AnfNodePtr anf_node = nullptr;
  try {
    auto src_cnode = GetSrcPtr<CNodePtr>(res_mgr, op);
    MS_EXCEPTION_IF_NULL(src_cnode);
    if (i >= src_cnode->size() - 1) {
      MS_LOG(ERROR) << "Invalid input index, it should be less than " << src_cnode->size() - 1 << ", but got: " << i;
      return nullptr;
    }
    anf_node = src_cnode->input(i + 1);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get input from CNode failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, anf_node);
}

size_t MSOpGetInputsNum(ResMgrHandle res_mgr, ConstNodeHandle op, STATUS *error) {
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || op == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  size_t input_num;
  try {
    auto src_cnode = GetSrcPtr<CNodePtr>(res_mgr, op);
    MS_EXCEPTION_IF_NULL(src_cnode);
    input_num = src_cnode->size() - 1;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph get input number failed. Error info: " << e.what();
    *error = RET_ERROR;
    return 0;
  }
  *error = RET_OK;
  return input_num;
}

STATUS MSOpGetInputs(ResMgrHandle res_mgr, ConstNodeHandle op, NodeHandle inputs[], size_t input_num) {
  if (res_mgr == nullptr || op == nullptr || inputs == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [inputs] is nullptr.";
    return RET_NULL_PTR;
  }
  try {
    auto src_cnode = GetSrcPtr<CNodePtr>(res_mgr, op);
    MS_EXCEPTION_IF_NULL(src_cnode);
    auto in_num = src_cnode->size() - 1;
    if (in_num != input_num) {
      MS_LOG(ERROR) << "Invalid input number, it should be: " << in_num << ", but got: " << input_num;
      return RET_ERROR;
    }
    auto cnode_inputs = src_cnode->inputs();
    for (size_t i = 0; i < input_num; i++) {
      inputs[i] = GetRawPtr(res_mgr, cnode_inputs[i + 1]);
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get inputs from CNode failed. Error info: " << e.what();
    return RET_ERROR;
  }
  return RET_OK;
}

size_t MSOpGetOutputDimension(ResMgrHandle res_mgr, ConstNodeHandle op, size_t output_index, STATUS *ret) {
  if (res_mgr == nullptr || op == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] is nullptr.";
    *ret = RET_NULL_PTR;
    return 0;
  }
  try {
    auto src_cnode = GetSrcPtr<CNodePtr>(res_mgr, op);
    MS_EXCEPTION_IF_NULL(src_cnode);
    std::vector<int64_t> shape = mindspore::common::AnfAlgo::GetOutputInferShape(src_cnode, output_index);
    return shape.size();
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Shape from OP/CNode failed. Error info: " << e.what();
    *ret = RET_ERROR;
    return 0;
  }
}

STATUS MSOpGetOutputShape(ResMgrHandle res_mgr, ConstNodeHandle op, int64_t shape_ret[], size_t dim,
                          size_t output_index) {
  if (res_mgr == nullptr || op == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] is nullptr.";
    return RET_NULL_PTR;
  }
  try {
    auto src_cnode = GetSrcPtr<CNodePtr>(res_mgr, op);
    MS_EXCEPTION_IF_NULL(src_cnode);
    std::vector<int64_t> shape = mindspore::common::AnfAlgo::GetOutputInferShape(src_cnode, output_index);
    MS_EXCEPTION_IF_CHECK_FAIL(
      dim >= shape.size(),
      "Input dimension less than the actual Dimension. Please ensure shape_ret have enough space.");
    (void)std::copy(shape.begin(), shape.end(), shape_ret);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Shape from OP/CNode failed. Error info: " << e.what();
    return RET_ERROR;
  }
  return RET_OK;
}

NodeHandle MSNewFuncCallNode(ResMgrHandle res_mgr, GraphHandle graph, ConstGraphHandle sub_graph, Handle const inputs[],
                             size_t input_num) {
  if (res_mgr == nullptr || graph == nullptr || sub_graph == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] or [sub_graph] is nullptr.";
    return nullptr;
  }
  CNodePtr cnode = nullptr;
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    auto res_sub_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, sub_graph);
    MS_EXCEPTION_IF_NULL(res_sub_fg);
    auto sub_node = mindspore::NewValueNode(res_sub_fg);
    sub_node->set_abstract(res_sub_fg->ToAbstract());
    std::vector<AnfNodePtr> cnode_inputs{sub_node};
    for (size_t i = 0; i < input_num; ++i) {
      auto cnode_input = GetSrcPtr<AnfNodePtr>(res_mgr, inputs[i]);
      MS_EXCEPTION_IF_NULL(cnode_input);
      cnode_inputs.push_back(cnode_input);
    }
    cnode = res_fg->NewCNodeInOrder(cnode_inputs);
    MS_EXCEPTION_IF_NULL(res_sub_fg->output());
    cnode->set_abstract(res_sub_fg->output()->abstract());
    (void)res_fg->AddFuncGraphUsed(res_sub_fg);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph create SubGraph node failed. Error info: " << e.what();
    return nullptr;
  }
  MS_LOG(INFO) << "Add function call node";
  return GetRawPtr(res_mgr, cnode);
}

NodeHandle MSNewPlaceholder(ResMgrHandle res_mgr, GraphHandle graph, DataTypeC type, const int64_t shape[],
                            size_t shape_size) {
  if (res_mgr == nullptr || graph == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] is nullptr.";
    return nullptr;
  }
  ParameterPtr param = nullptr;
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    param = res_fg->add_parameter();
    auto type_ptr = mindspore::TypeIdToType(mindspore::TypeId(type));
    AbstractBasePtr abs = GetAbstract(type_ptr, shape, shape_size, true);
    param->set_abstract(abs);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph add parameter failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, param);
}

NodeHandle MSNewVariableScalarFloat32(ResMgrHandle res_mgr, GraphHandle graph, float value) {
  if (res_mgr == nullptr || graph == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] is nullptr.";
    return nullptr;
  }
  ParameterPtr param = nullptr;
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    param = GetScalarParam<float>(res_fg, value, mindspore::kNumberTypeFloat32);
    MS_EXCEPTION_IF_NULL(param);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "New Scalar Variable failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, param);
}

NodeHandle MSNewVariableScalarInt32(ResMgrHandle res_mgr, GraphHandle graph, int value) {
  if (res_mgr == nullptr || graph == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] is nullptr.";
    return nullptr;
  }
  ParameterPtr param = nullptr;
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    param = GetScalarParam<float>(res_fg, value, mindspore::kNumberTypeInt32);
    MS_EXCEPTION_IF_NULL(param);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "New Scalar Variable failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, param);
}

NodeHandle MSNewVariableArray(ResMgrHandle res_mgr, GraphHandle graph, void *data, DataTypeC type,
                              const int64_t shape[], size_t shape_size, size_t data_len) {
  if (res_mgr == nullptr || graph == nullptr || data == nullptr || shape == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] or [data] or [shape] is nullptr.";
    return nullptr;
  }
  ParameterPtr param = nullptr;
  ShapeVector shape_vec(shape, shape + shape_size);
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    param = res_fg->add_parameter();
    auto tensor = std::make_shared<TensorImpl>(mindspore::TypeId(type), shape_vec, data, data_len);
    tensor->set_param_info(std::make_shared<mindspore::ParamInfo>());
    param->set_abstract(tensor->ToAbstract());
    param->set_default_param(tensor);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "New Tensor Variable failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, param);
}

NodeHandle MSNewVariableFromTensor(ResMgrHandle res_mgr, GraphHandle graph, ConstTensorHandle tensor) {
  if (res_mgr == nullptr || graph == nullptr || tensor == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] or [tensor] is nullptr.";
    return nullptr;
  }
  ParameterPtr param = nullptr;
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    auto tensor_impl = GetSrcPtr<TensorPtr>(res_mgr, tensor);
    MS_EXCEPTION_IF_NULL(tensor_impl);
    param = res_fg->add_parameter();
    param->set_abstract(tensor_impl->ToAbstract());
    param->set_default_param(tensor_impl);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "New Tensor Variable failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, param);
}

size_t MSVariableArrayGetDataSize(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error) {
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  try {
    auto node_impl = GetSrcPtr<ParameterPtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->default_param();
    MS_EXCEPTION_IF_NULL(val);
    auto tensor = val->cast<TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    size_t data_size = tensor->Size();
    *error = RET_OK;
    return data_size;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Tensor Variable get data failed. Error info: " << e.what();
    *error = RET_ERROR;
    return 0;
  }
}

void *MSVariableArrayGetData(ResMgrHandle res_mgr, ConstNodeHandle node) {
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    return nullptr;
  }
  try {
    auto node_impl = GetSrcPtr<ParameterPtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->default_param();
    MS_EXCEPTION_IF_NULL(val);
    auto tensor = val->cast<TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    void *data = tensor->data_c();
    return data;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Tensor Variable get data failed. Error info: " << e.what();
    return nullptr;
  }
}

NodeHandle MSNewConstantArray(ResMgrHandle res_mgr, void *data, DataTypeC type, const int64_t shape[],
                              size_t shape_size, size_t data_len) {
  if (res_mgr == nullptr || data == nullptr || shape == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [data] or [shape] is nullptr.";
    return nullptr;
  }
  ShapeVector shape_vec(shape, shape + shape_size);
  ValueNodePtr value_node = nullptr;
  try {
    auto tensor = std::make_shared<TensorImpl>(mindspore::TypeId(type), shape_vec, data, data_len);
    tensor->set_param_info(std::make_shared<mindspore::ParamInfo>());
    value_node = mindspore::NewValueNode(tensor);
    value_node->set_abstract(tensor->ToAbstract());
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "New Tensor Variable failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, value_node);
}

NodeHandle MSNewConstantFromTensor(ResMgrHandle res_mgr, TensorHandle tensor) {
  if (res_mgr == nullptr || tensor == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [tensor] is nullptr.";
    return nullptr;
  }
  ValueNodePtr value_node = nullptr;
  try {
    auto tensor_impl = GetSrcPtr<TensorPtr>(res_mgr, tensor);
    MS_EXCEPTION_IF_NULL(tensor_impl);
    value_node = mindspore::NewValueNode(tensor_impl);
    value_node->set_abstract(tensor_impl->ToAbstract());
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "New Tensor Variable failed. Error info: " << e.what();
    return nullptr;
  }
  return GetRawPtr(res_mgr, value_node);
}

size_t MSConstantArrayGetDataSize(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error) {
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    auto tensor = val->cast<TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    size_t data_size = tensor->Size();
    *error = RET_OK;
    return data_size;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Tensor Constant get data failed. Error info: " << e.what();
    *error = RET_ERROR;
    return 0;
  }
}

void *MSConstantArrayGetData(ResMgrHandle res_mgr, ConstNodeHandle node) {
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    return nullptr;
  }
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    auto tensor = val->cast<TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    void *data = tensor->data_c();
    return data;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Tensor Constant get data failed. Error info: " << e.what();
    return nullptr;
  }
}

NodeHandle MSNewConstantScalarFloat32(ResMgrHandle res_mgr, float value) {
  MS_LOG(INFO) << "New Float32 Scalar Value!s";
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto value_node = mindspore::NewValueNode(value);
  value_node->set_abstract(std::make_shared<AbstractScalarImpl>(value));
  return GetRawPtr(res_mgr, value_node);
}

NodeHandle MSNewConstantScalarBool(ResMgrHandle res_mgr, bool value) {
  MS_LOG(INFO) << "New Bool Scalar Value!";
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto value_node = mindspore::NewValueNode(value);
  value_node->set_abstract(std::make_shared<AbstractScalarImpl>(value));
  return GetRawPtr(res_mgr, value_node);
}

NodeHandle MSNewConstantScalarInt32(ResMgrHandle res_mgr, int value) {
  MS_LOG(INFO) << "New Int32 Scalar Value!";
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto value_node = mindspore::NewValueNode(value);
  value_node->set_abstract(std::make_shared<AbstractScalarImpl>(value));
  return GetRawPtr(res_mgr, value_node);
}

NodeHandle MSNewConstantScalarInt64(ResMgrHandle res_mgr, int64_t value) {
  MS_LOG(INFO) << "New Int64 Scalar Value!";
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto value_node = mindspore::NewValueNode(value);
  value_node->set_abstract(std::make_shared<AbstractScalarImpl>(value));
  return GetRawPtr(res_mgr, value_node);
}

NodeHandle MSNewConstantString(ResMgrHandle res_mgr, const char *str) {
  MS_LOG(INFO) << "New String Scalar Value!";
  if (res_mgr == nullptr || str == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [str] is nullptr.";
    return nullptr;
  }
  string str_val(str);
  auto value_node = mindspore::NewValueNode(str_val);
  value_node->set_abstract(std::make_shared<AbstractScalarImpl>(str_val));
  return GetRawPtr(res_mgr, value_node);
}

NodeHandle MSNewConstantTupleInt64(ResMgrHandle res_mgr, const int64_t vec[], size_t size) {
  MS_LOG(INFO) << "New Vector Value!";
  if (res_mgr == nullptr || vec == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [vec] is nullptr.";
    return nullptr;
  }
  auto value_node = mindspore::NewValueNode(std::vector<int64_t>(vec, vec + size));
  mindspore::AbstractBasePtrList abs_list = {};
  for (size_t i = 0; i < size; i++) {
    AbstractBasePtr base = std::make_shared<AbstractScalarImpl>(vec[i]);
    abs_list.push_back(base);
  }
  auto abstract = std::make_shared<AbstractTupleImpl>(abs_list);
  value_node->set_abstract(abstract);
  return GetRawPtr(res_mgr, value_node);
}

NodeHandle MSNewConstantType(ResMgrHandle res_mgr, DataTypeC type) {
  MS_LOG(INFO) << "New Type Value: " << type;
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto type_ptr = mindspore::TypeIdToType(mindspore::TypeId(type));
  auto value_node = mindspore::NewValueNode(type_ptr);
  auto abstract = std::make_shared<AbstractTypeImpl>(type_ptr);
  value_node->set_abstract(abstract);
  return GetRawPtr(res_mgr, value_node);
}

int MSConstantScalarGetValueInt32(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error) {
  MS_LOG(INFO) << "Get Int32 Scalar Value!";
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  int ret_val = 0;
  *error = RET_OK;
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    if (val->isa<TensorImpl>()) {
      auto val_tensor = val->cast<TensorPtr>();
      auto data = val_tensor->data_c();
      MS_EXCEPTION_IF_NULL(data);
      ret_val = static_cast<int *>(data)[0];
    } else if (val->isa<Int32ImmImpl>()) {
      auto val_imm = val->cast<Int32ImmPtr>();
      ret_val = val_imm->value();
    } else {
      MS_LOG(ERROR) << "Input node has invalid value type: " << val->type_name();
      *error = RET_ERROR;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Int32 Scalar value failed. Error info: " << e.what();
    *error = RET_ERROR;
  }
  return ret_val;
}

float MSConstantScalarGetValueFloat32(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error) {
  MS_LOG(INFO) << "Get Float32 Scalar Value!";
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  float ret_val = 0;
  *error = RET_OK;
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    if (val->isa<TensorImpl>()) {
      auto val_tensor = val->cast<TensorPtr>();
      auto data = val_tensor->data_c();
      MS_EXCEPTION_IF_NULL(data);
      ret_val = static_cast<float *>(data)[0];
    } else if (val->isa<Float32ImmImpl>()) {
      auto val_imm = val->cast<Float32ImmPtr>();
      ret_val = val_imm->value();
    } else {
      MS_LOG(ERROR) << "Input node has invalid value type: " << val->type_name();
      *error = RET_ERROR;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Float32 Scalar value failed. Error info: " << e.what();
    *error = RET_ERROR;
  }
  return ret_val;
}

bool MSConstantScalarGetValueBool(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error) {
  MS_LOG(INFO) << "Get Bool Scalar Value!";
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return false;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return false;
  }
  int ret_val = false;
  *error = RET_OK;
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    if (val->isa<TensorImpl>()) {
      auto val_tensor = val->cast<TensorPtr>();
      auto data = val_tensor->data_c();
      MS_EXCEPTION_IF_NULL(data);
      ret_val = static_cast<bool *>(data)[0];
    } else if (val->isa<BoolImmImpl>()) {
      auto val_imm = val->cast<BoolImmPtr>();
      ret_val = val_imm->value();
    } else {
      MS_LOG(ERROR) << "Input node has invalid value type: " << val->type_name();
      *error = RET_ERROR;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Bool Scalar value failed. Error info: " << e.what();
    *error = RET_ERROR;
  }
  return ret_val;
}

int64_t MSConstantScalarGetValueInt64(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error) {
  MS_LOG(INFO) << "Get Int64 Scalar Value!";
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  int64_t ret_val = 0;
  *error = RET_OK;
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    if (val->isa<TensorImpl>()) {
      auto val_tensor = val->cast<TensorPtr>();
      auto data = val_tensor->data_c();
      MS_EXCEPTION_IF_NULL(data);
      ret_val = static_cast<int64_t *>(data)[0];
    } else if (val->isa<Int64ImmImpl>()) {
      auto val_imm = val->cast<Int64ImmPtr>();
      ret_val = val_imm->value();
    } else {
      MS_LOG(ERROR) << "Input node has invalid value type: " << val->type_name();
      *error = RET_ERROR;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Int64 Scalar value failed. Error info: " << e.what();
    *error = RET_ERROR;
  }
  return ret_val;
}

STATUS MSConstantStringGetValue(ResMgrHandle res_mgr, ConstNodeHandle node, char str_buf[], size_t str_len) {
  MS_LOG(INFO) << "Get String Constant Value!";
  if (res_mgr == nullptr || node == nullptr || str_buf == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] or [str_buf] is nullptr.";
    return RET_NULL_PTR;
  }
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    auto val_str = val->cast<StringImmPtr>();
    std::string ret_val = val_str->value();
    size_t valid_size = ret_val.size() < str_len - 1 ? ret_val.size() : str_len - 1;
    for (size_t i = 0; i < valid_size; i++) {
      str_buf[i] = ret_val.c_str()[i];
    }
    str_buf[valid_size] = '\0';
    return RET_OK;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get String Constant value failed. Error info: " << e.what();
    return RET_ERROR;
  }
}

size_t MSConstantTupleGetSize(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error) {
  MS_LOG(INFO) << "Get Tuple Constant size!";
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    auto val_tuple = val->cast<ValueTuplePtr>();
    auto tuple_size = val_tuple->size();
    *error = RET_OK;
    return tuple_size;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Tuple Constant size failed. Error info: " << e.what();
    *error = RET_ERROR;
    return 0;
  }
}

STATUS MSConstantTupleGetValueInt64(ResMgrHandle res_mgr, ConstNodeHandle node, int64_t vec[], size_t size) {
  MS_LOG(INFO) << "Get Tuple Constant Value!";
  if (res_mgr == nullptr || node == nullptr || vec == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] or [vec] is nullptr.";
    return RET_NULL_PTR;
  }
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    auto val_tuple = val->cast<ValueTuplePtr>();
    auto val_list = val_tuple->value();
    if (val_list.size() != size) {
      MS_LOG(ERROR) << "Invalid input vector length, it should be: " << val_list.size() << ", but got: " << size;
      return RET_ERROR;
    }
    for (size_t i = 0; i < size; i++) {
      auto val_imm = val_list[i]->cast<Int64ImmPtr>();
      vec[i] = val_imm->value();
    }
    return RET_OK;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get String Constant value failed. Error info: " << e.what();
    return RET_ERROR;
  }
}

DataTypeC MSConstantTypeGetValue(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error) {
  MS_LOG(INFO) << "Get Type Constant Value!";
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return MS_INVALID_TYPE;
  }
  if (res_mgr == nullptr || node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] is nullptr.";
    *error = RET_NULL_PTR;
    return MS_INVALID_TYPE;
  }
  try {
    auto node_impl = GetSrcPtr<ValueNodePtr>(res_mgr, node);
    MS_EXCEPTION_IF_NULL(node_impl);
    auto val = node_impl->value();
    MS_EXCEPTION_IF_NULL(val);
    auto val_type = val->cast<TypePtr>();
    auto ret_val = static_cast<DataTypeC>(val_type->type_id());
    *error = RET_OK;
    return ret_val;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Type Constant value failed. Error info: " << e.what();
    *error = RET_ERROR;
    return MS_INVALID_TYPE;
  }
}

PrimitivePtr GetOpPrim(ResMgrHandle res_mgr, ConstNodeHandle node) {
  auto src_node = GetSrcPtr<CNodePtr>(res_mgr, node);
  auto node_input = src_node->input(0);
  if (node_input == nullptr) {
    MS_LOG(ERROR) << "The node's input is nullptr.";
    return nullptr;
  }
  auto prim_node = node_input->cast<ValueNodePtr>();
  if (prim_node == nullptr) {
    MS_LOG(ERROR) << "The node's input is with invalid type.";
    return nullptr;
  }
  auto node_value = prim_node->value();
  if (node_value == nullptr) {
    MS_LOG(ERROR) << "The node's value is nullptr.";
    return nullptr;
  }
  auto prim = node_value->cast<PrimitivePtr>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "The node's value is with invalid type.";
    return nullptr;
  }
  return prim;
}

STATUS MSOpSetAttrScalarFloat32(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, float value) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }
  prim->set_attr(attr_name, mindspore::MakeValue(value));
  return RET_OK;
}

STATUS MSOpSetAttrScalarBool(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, bool value) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }
  prim->set_attr(attr_name, mindspore::MakeValue(value));
  return RET_OK;
}

STATUS MSOpSetAttrScalarInt32(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, int32_t value) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }
  prim->set_attr(attr_name, mindspore::MakeValue(value));
  return RET_OK;
}

STATUS MSOpSetAttrScalarInt64(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, int64_t value) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }
  prim->set_attr(attr_name, mindspore::MakeValue(value));
  return RET_OK;
}

STATUS MSOpSetAttrType(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, DataTypeC value) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }
  auto cxx_type = mindspore::TypeId(value);
  prim->set_attr(attr_name, mindspore::TypeIdToType(cxx_type));
  return RET_OK;
}

STATUS MSOpSetAttrTypeArray(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, DataTypeC value[],
                            size_t vec_size) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }
  std::vector<mindspore::ValuePtr> vec_value;
  mindspore::TypeId cxx_type;
  for (size_t i = 0; i < vec_size; i++) {
    cxx_type = mindspore::TypeId(value[i]);
    vec_value.push_back(mindspore::TypeIdToType(cxx_type));
  }
  prim->set_attr(attr_name, mindspore::MakeValue(vec_value));
  return RET_OK;
}

STATUS MSOpSetAttrArray(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, void *value, size_t vec_size,
                        DataTypeC data_type) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr || value == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] or [value_vec] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }

  switch (data_type) {
    case MS_BOOL: {
      std::vector<bool> vec_value(static_cast<bool *>(value), static_cast<bool *>(value) + vec_size);
      prim->set_attr(attr_name, mindspore::MakeValue(vec_value));
      break;
    }
    case MS_INT32: {
      std::vector<int32_t> vec_value(static_cast<int32_t *>(value), static_cast<int32_t *>(value) + vec_size);
      prim->set_attr(attr_name, mindspore::MakeValue(vec_value));
      break;
    }
    case MS_INT64: {
      std::vector<int64_t> vec_value(static_cast<int64_t *>(value), static_cast<int64_t *>(value) + vec_size);
      prim->set_attr(attr_name, mindspore::MakeValue(vec_value));
      break;
    }
    case MS_FLOAT32: {
      std::vector<float> vec_value(static_cast<float *>(value), static_cast<float *>(value) + vec_size);
      prim->set_attr(attr_name, mindspore::MakeValue(vec_value));
      break;
    }
    default:
      MS_LOG(ERROR) << "Unrecognized datatype w/ DataTypeC ID: " << data_type << " , Attribute name: " << attr_name
                    << std::endl;
      return RET_ERROR;
  }
  return RET_OK;
}

STATUS MSOpSetAttrStringArray(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, const char *value[],
                              size_t vec_size) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr || value == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] or [value_vec] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }

  std::vector<mindspore::ValuePtr> vec_value;
  for (size_t i = 0; i < vec_size; i++) {
    vec_value.push_back(mindspore::MakeValue(value[i]));
  }
  prim->set_attr(attr_name, std::make_shared<mindspore::ValueList>(vec_value));
  return RET_OK;
}

STATUS MSOpSetAttrString(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, const char *value) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr || value == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] or [value_vec] is nullptr.";
    return RET_NULL_PTR;
  }
  auto prim = GetOpPrim(res_mgr, op);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get primitive node failed";
    return RET_NULL_PTR;
  }
  std::string value_str(value);
  prim->set_attr(attr_name, mindspore::MakeValue(value_str));
  return RET_OK;
}

int64_t MSOpGetAttrScalarInt64(ResMgrHandle res_mgr, ConstNodeHandle op, const char *attr_name, STATUS *error) {
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  std::string attr_name_str(attr_name);
  try {
    auto prim = GetOpPrim(res_mgr, op);
    MS_EXCEPTION_IF_NULL(prim);
    auto value = prim->GetAttr(attr_name_str);
    auto value_int64 = value->cast<Int64ImmPtr>();
    MS_EXCEPTION_IF_NULL(value_int64);
    auto ret_val = value_int64->value();
    *error = RET_OK;
    return ret_val;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << " Get Attribute failed. Error info: " << e.what();
    *error = RET_ERROR;
    return 0;
  }
}

STATUS MSOpGetAttrArrayInt64(ResMgrHandle res_mgr, ConstNodeHandle op, const char *attr_name, int64_t values[],
                             size_t value_num) {
  if (res_mgr == nullptr || op == nullptr || attr_name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [op] or [attr_name] is nullptr.";
    return RET_NULL_PTR;
  }
  std::string attr_name_str(attr_name);
  try {
    auto prim = GetOpPrim(res_mgr, op);
    MS_EXCEPTION_IF_NULL(prim);
    auto value = prim->GetAttr(attr_name_str);
    MS_EXCEPTION_IF_NULL(value);
    auto value_tuple = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    auto value_list = value_tuple->value();
    if (value_list.size() != value_num) {
      MS_LOG(ERROR) << "Invalid input vector length, it should be: " << value_list.size() << ", but got: " << value_num;
      return RET_ERROR;
    }
    for (size_t i = 0; i < value_num; i++) {
      auto val_imm = value_list[i]->cast<Int64ImmPtr>();
      values[i] = val_imm->value();
    }
    return RET_OK;
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Get Attribute failed. Error info: " << e.what();
    return RET_ERROR;
  }
}

STATUS MSOpSetName(ResMgrHandle res_mgr, NodeHandle node, const char *name) {
  if (res_mgr == nullptr || node == nullptr || name == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] or [name] is nullptr.";
    return RET_NULL_PTR;
  }
  auto node_impl = GetSrcPtr<CNodePtr>(res_mgr, node);
  if (node_impl == nullptr) {
    MS_LOG(ERROR) << "Get source pointer failed. Please check whether the input node is an operator node.";
    return RET_ERROR;
  }
  node_impl->set_fullname_with_scope(name);
  return RET_OK;
}

STATUS MSNodeGetName(ResMgrHandle res_mgr, ConstNodeHandle node, char str_buf[], size_t str_len) {
  if (res_mgr == nullptr || node == nullptr || str_buf == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [node] or [str_buf] is nullptr.";
    return RET_NULL_PTR;
  }
  auto node_impl = GetSrcPtr<AnfNodePtr>(res_mgr, node);
  if (node_impl == nullptr) {
    MS_LOG(ERROR) << "Get source pointer failed.";
    return RET_ERROR;
  }
  auto name = node_impl->fullname_with_scope();
  size_t valid_size = name.size() < str_len - 1 ? name.size() : str_len - 1;
  for (size_t i = 0; i < valid_size; i++) {
    str_buf[i] = name.c_str()[i];
  }
  str_buf[valid_size] = '\0';
  return RET_OK;
}

// dynamic op / eager mode
std::shared_ptr<InnerOpInfo> GenerateInnerInfo(ResMgrHandle res_mgr, const char *op_type, TensorHandle const inputs[],
                                               size_t input_num, size_t output_num, const DynamicOpInfo &extra_info) {
  MS_EXCEPTION_IF_NULL(op_type);
  MS_EXCEPTION_IF_NULL(inputs);
  std::vector<ValuePtr> src_inputs{};
  std::vector<ShapeVector> out_shapes{};
  std::vector<DataTypeC> out_dtypes{};
  std::vector<std::pair<std::string, ValuePtr>> attrs_pair{};
  for (size_t i = 0; i < input_num; i++) {
    auto input = GetSrcPtr<ValuePtr>(res_mgr, inputs[i]);
    if (input == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid input. Index: " << i;
    }
    (void)src_inputs.emplace_back(input);
  }
  if (extra_info.output_shapes != nullptr && extra_info.output_dtypes != nullptr) {
    for (size_t i = 0; i < output_num; i++) {
      MS_EXCEPTION_IF_NULL(extra_info.output_dims);
      size_t dim = extra_info.output_dims[i];
      ShapeVector out_shape{};
      MS_EXCEPTION_IF_NULL(extra_info.output_shapes[i]);
      for (size_t j = 0; j < dim; j++) {
        (void)out_shape.emplace_back(extra_info.output_shapes[i][j]);
      }
      (void)out_shapes.emplace_back(out_shape);
      (void)out_dtypes.emplace_back(extra_info.output_dtypes[i]);
    }
  }
  for (size_t i = 0; i < extra_info.attr_num; i++) {
    MS_EXCEPTION_IF_NULL(extra_info.attr_names[i]);
    auto value = GetSrcPtr<ValuePtr>(res_mgr, extra_info.attr_values[i]);
    if (value == nullptr) {
      MS_LOG(ERROR) << "Get attribute's source pointer failed, attribute index: " << i;
    }
    (void)attrs_pair.emplace_back(std::make_pair(extra_info.attr_names[i], value));
  }
  return std::make_shared<InnerOpInfo>(op_type, src_inputs, out_shapes, out_dtypes, attrs_pair);
}

STATUS CheckExtraInfo(const DynamicOpInfo &extra_info) {
  MS_ERROR_IF_TRUE_W_RET_N_LOG(extra_info.attr_num < 0, RET_ERROR, "The attr_num must be non-zero!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(
    extra_info.attr_num == 0 && (extra_info.attr_names != nullptr || extra_info.attr_values != nullptr), RET_ERROR,
    "The attr_name and attr_values must be nullptr if attr_num is 0!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(
    extra_info.attr_num != 0 && (extra_info.attr_names == nullptr || extra_info.attr_values == nullptr), RET_ERROR,
    "The attr_name and attr_values must be specified if attr_num is non-negative!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(extra_info.output_dims != nullptr && extra_info.output_shapes == nullptr, RET_ERROR,
                               "The output_shapes must be not nullptr if output_dims is non-zero!");
  return RET_OK;
}

STATUS OpRunInfoSetInputs(ResMgrHandle res_mgr, TensorHandle const inputs[], size_t input_num,
                          FrontendOpRunInfoPtr op_run_info) {
  auto prim = op_run_info->op_grad_info->op_prim;
  MS_EXCEPTION_IF_NULL(prim);
  op_run_info->input_size = input_num;
  op_run_info->op_grad_info->input_value.resize(input_num);
  for (size_t i = 0; i < input_num; i++) {
    auto in_arg = GetSrcPtr<ValuePtr>(res_mgr, inputs[i]);
    if (in_arg == nullptr) {
      MS_LOG(ERROR) << "Invalid input. Index: " << i;
      return RET_ERROR;
    }
    op_run_info->op_grad_info->input_value[i] = in_arg;
  }
  return RET_OK;
}

STATUS DynamicOpInfer(size_t output_num, FrontendOpRunInfoPtr op_run_info, const DynamicOpInfo &extra_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  // get abstract
  op_run_info->op_grad_info->input_abs.resize(op_run_info->input_size);
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    auto input_value = op_run_info->op_grad_info->input_value[i];
    op_run_info->op_grad_info->input_abs[i] = input_value->ToAbstract();
  }
  // do infer
  AbstractBasePtr out_abs = nullptr;
  auto prim = op_run_info->op_grad_info->op_prim;
  if (extra_info.output_shapes != nullptr && extra_info.output_dims != nullptr && extra_info.output_dtypes != nullptr) {
    auto shape = BuildShape(extra_info.output_shapes, extra_info.output_dims, output_num);
    auto type = BuildType(extra_info.output_dtypes, output_num);
    out_abs = BuildAbstract(shape, type);
  } else {
    MS_LOG(INFO) << "Output shapes and dtypes info is not specified completely, using inner infer.";
    prim->BeginRecordAddAttr();
    out_abs = OpInferShapeAndType(prim, op_run_info->op_grad_info->input_abs);
    prim->EndRecordAddAttr();
  }
  MS_EXCEPTION_IF_NULL(out_abs);
  op_run_info->base_op_run_info.abstract = out_abs;
  return RET_OK;
}

const OpBackendPtr &DynamicOpGetBackend(ResMgrHandle res_mgr) {
  auto res_mgr_ptr = reinterpret_cast<ResourceManager *>(res_mgr);
  MS_EXCEPTION_IF_NULL(res_mgr_ptr);
  return res_mgr_ptr->GetOpBackend();
}

ValuePtr DynamicOpRun(ResMgrHandle res_mgr, const FrontendOpRunInfoPtr &op_run_info) {
  MS_LOG(DEBUG) << "DynamicOpRun start";
  MS_EXCEPTION_IF_NULL(op_run_info);
  auto ms_context = mindspore::MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(mindspore::MS_CTX_DEVICE_ID);
  ms_context->set_param<bool>(mindspore::MS_CTX_ENABLE_PYNATIVE_INFER, true);
  mindspore::pynative::PyNativeAlgo::DataConvert::GetInputTensor(op_run_info, nullptr);
  auto backend_op_run_info = std::make_shared<mindspore::BackendOpRunInfo>(
    op_run_info->base_op_run_info, std::make_shared<mindspore::Primitive>(*op_run_info->op_grad_info->op_prim), true,
    false);

  mindspore::VectorRef outputs;
  const auto &op_backend = DynamicOpGetBackend(res_mgr);
  MS_EXCEPTION_IF_NULL(op_backend);
  py::scoped_interpreter py_scope;
  if (op_run_info->base_op_run_info.use_dynamic_shape_process) {
    mindspore::AnfAlgo::SetDynamicAttrToPrim(backend_op_run_info->op_prim);
  }
  op_backend->Run(backend_op_run_info, op_run_info->base_op_run_info.device_target, device_id, &outputs);

  if (op_run_info->base_op_run_info.has_dynamic_output) {
    op_run_info->base_op_run_info.abstract = backend_op_run_info->base_op_run_info.abstract;
  }
  bool is_out_sequence = (op_run_info->base_op_run_info.abstract == nullptr ||
                          op_run_info->base_op_run_info.abstract->isa<mindspore::abstract::AbstractSequence>());
  const auto &result = mindspore::pynative::PyNativeAlgo::DataConvert::VectorRefToValue(
    outputs, op_run_info->requires_grad, is_out_sequence);
  ms_context->set_param<bool>(mindspore::MS_CTX_ENABLE_PYNATIVE_INFER, false);
  MS_LOG(DEBUG) << "DynamicOpRun end";
  return result;
}

STATUS MSRunOpWithInfo(ResMgrHandle res_mgr, const char *op_type, TensorHandle const inputs[], size_t input_num,
                       TensorHandle outputs[], size_t output_num, DynamicOpInfo extra_info) {
  MS_ERROR_IF_TRUE_W_RET_N_LOG(res_mgr == nullptr, RET_NULL_PTR, "Input Handle [res_mgr] is nullptr!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(inputs == nullptr, RET_NULL_PTR, "Input Handle [inputs] is nullptr!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(outputs == nullptr, RET_NULL_PTR, "Input Handle [outputs] is nullptr!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(input_num == 0, RET_NULL_PTR, "Input [input_num] must be non-zero!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(output_num == 0, RET_NULL_PTR, "Input [output_num] must be non-zero!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(CheckExtraInfo(extra_info) != RET_OK, RET_NULL_PTR, "Input [extra_info] is invalid!");
  try {
    auto res_mgr_ptr = reinterpret_cast<ResourceManager *>(res_mgr);
    FrontendOpRunInfoPtr op_run_info = nullptr;
    auto op_info = GenerateInnerInfo(res_mgr, op_type, inputs, input_num, output_num, extra_info);
    auto cached_run_info = res_mgr_ptr->GetOpRunInfoFromCache(op_info);
    if (cached_run_info != nullptr) {
      op_run_info = cached_run_info;
      // set inputs
      auto ret = OpRunInfoSetInputs(res_mgr, inputs, input_num, op_run_info);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Dynamic Op set inputs failed.";
        return RET_ERROR;
      }
    } else {
      // create op_run_info
      op_run_info = std::make_shared<mindspore::pynative::FrontendOpRunInfo>();
      op_run_info->base_op_run_info.op_name = op_type;
      op_run_info->requires_grad = false;
      auto ms_context = mindspore::MsContext::GetInstance();
      auto cur_target = ms_context->get_param<std::string>(mindspore::MS_CTX_DEVICE_TARGET);
      op_run_info->base_op_run_info.device_target = cur_target;
      // create prim
      auto prim = std::make_shared<PrimitiveImpl>(op_type);
      op_run_info->op_grad_info->op_prim = prim;
      // set inputs
      bool is_dynamic_shape =
        op_run_info->base_op_run_info.has_dynamic_output || op_run_info->base_op_run_info.use_dynamic_shape_process;
      mindspore::pynative::PyNativeAlgo::Common::GetConstInputToAttr(prim, op_type, cur_target, is_dynamic_shape,
                                                                     &op_run_info->input_to_attr);
      auto ret = OpRunInfoSetInputs(res_mgr, inputs, input_num, op_run_info);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Dynamic Op set inputs failed.";
        return RET_ERROR;
      }
      // set args
      if (extra_info.attr_names != nullptr && extra_info.attr_values != nullptr) {
        ret = OpSetAttrs(res_mgr, prim, extra_info.attr_names, extra_info.attr_values, extra_info.attr_num);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "Dynamic Op set attributes failed.";
          return RET_ERROR;
        }
      }
      // infer and set abstract
      ret = DynamicOpInfer(output_num, op_run_info, extra_info);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Dynamic Op infer shape and type failed.";
        return RET_ERROR;
      }
      // cache op run info
      res_mgr_ptr->CacheOpRunInfo(op_info, op_run_info);
    }

    // run op
    op_run_info->real_out = DynamicOpRun(res_mgr, op_run_info);
    if (op_run_info->real_out->isa<ValueSequenceImpl>()) {
      const auto &result_v_list = op_run_info->real_out->cast<ValueSequencePtr>();
      if (result_v_list->size() == 1 && op_run_info->base_op_run_info.abstract != nullptr &&
          !op_run_info->base_op_run_info.abstract->isa<mindspore::abstract::AbstractSequence>()) {
        op_run_info->real_out = result_v_list->value().front();
      }
    }

    // clear used input tensor
    op_run_info->base_op_run_info.expanded_input_values.clear();
    op_run_info->base_op_run_info.input_types.clear();

    // get output tensor
    const std::vector<TensorPtr> &ref_outputs = ConvertOutputToTensor(op_run_info->real_out);
    if (ref_outputs.size() != output_num) {
      MS_LOG(ERROR) << "Invalid outputs number, it should be: " << ref_outputs.size() << ", but got: " << output_num;
      return RET_ERROR;
    }
    for (size_t i = 0; i < output_num; i++) {
      outputs[i] = GetRawPtr(res_mgr, ref_outputs[i]);
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Run op failed. Error info: " << e.what();
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS MSRunOp(ResMgrHandle res_mgr, const char *op_type, TensorHandle const inputs[], size_t input_num,
               TensorHandle outputs[], size_t output_num) {
  MS_ERROR_IF_TRUE_W_RET_N_LOG(res_mgr == nullptr, RET_NULL_PTR, "Input Handle [res_mgr] is nullptr!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(inputs == nullptr, RET_NULL_PTR, "Input Handle [inputs] is nullptr!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(outputs == nullptr, RET_NULL_PTR, "Input Handle [outputs] is nullptr!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(input_num == 0, RET_NULL_PTR, "Input [input_num] must be non-zero!");
  MS_ERROR_IF_TRUE_W_RET_N_LOG(output_num == 0, RET_NULL_PTR, "Input [output_num] must be non-zero!");
  DynamicOpInfo extra_info = {NULL, NULL, 0, NULL, NULL, NULL};
  return MSRunOpWithInfo(res_mgr, op_type, inputs, input_num, outputs, output_num, extra_info);
}
