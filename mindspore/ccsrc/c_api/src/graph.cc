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

#include "c_api/include/graph.h"
#include "c_api/src/helper.h"
#include "c_api/src/common.h"
#include "c_api/src/utils.h"
#include "base/base.h"
#include "ops/core_ops.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "ir/func_graph_cloner.h"
#include "utils/ms_context.h"
#include "backend/graph_compiler/backend.h"
#include "pipeline/jit/pass.h"

GraphHandle MSFuncGraphCreate(ResMgrHandle res_mgr) {
  if (res_mgr == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] is nullptr.";
    return nullptr;
  }
  auto fg = std::make_shared<FuncGraphImpl>();
  return GetRawPtr(res_mgr, fg);
}

NodeHandle MSFuncGraphGetInput(ResMgrHandle res_mgr, ConstGraphHandle graph, size_t i) {
  if (res_mgr == nullptr || graph == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [cnode] is nullptr.";
    return nullptr;
  }
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    auto fg_inputs = res_fg->get_inputs();
    if (i >= fg_inputs.size()) {
      MS_LOG(ERROR) << "Invalid input index, it should be less than " << fg_inputs.size() << ", but got: " << i;
      return nullptr;
    }
    return GetRawPtr(res_mgr, fg_inputs[i]);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph get inputs failed. Error info: " << e.what();
    return nullptr;
  }
}

size_t MSFuncGraphGetInputNum(ResMgrHandle res_mgr, ConstGraphHandle graph, STATUS *error) {
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || graph == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  size_t input_num;
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    input_num = res_fg->get_inputs().size();
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph get input number failed. Error info: " << e.what();
    *error = RET_ERROR;
    return 0;
  }
  *error = RET_OK;
  return input_num;
}

STATUS MSFuncGraphGetInputs(ResMgrHandle res_mgr, ConstGraphHandle graph, NodeHandle inputs[], size_t input_num) {
  if (res_mgr == nullptr || graph == nullptr || inputs == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] or [inputs] is nullptr.";
    return RET_NULL_PTR;
  }
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    auto fg_inputs = res_fg->get_inputs();
    if (fg_inputs.size() != input_num) {
      MS_LOG(ERROR) << "Invalid input number, it should be: " << fg_inputs.size() << ", but got: " << input_num;
      return RET_ERROR;
    }
    for (size_t i = 0; i < input_num; i++) {
      inputs[i] = GetRawPtr(res_mgr, fg_inputs[i]);
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph get inputs failed. Error info: " << e.what();
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS MSFuncGraphSetOutput(ResMgrHandle res_mgr, GraphHandle graph, ConstNodeHandle op_node, bool force_new_ret) {
  if (res_mgr == nullptr || graph == nullptr || op_node == nullptr) {
    MS_LOG(ERROR) << "Input GraphHandle [res_mgr] or [graph] or [op_node] is nullptr.";
    return RET_NULL_PTR;
  }
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    auto res_anfnode = GetSrcPtr<AnfNodePtr>(res_mgr, op_node);
    MS_EXCEPTION_IF_NULL(res_anfnode);
    res_fg->set_output(res_anfnode, force_new_ret);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph set output failed. Error info: " << e.what();
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS MSFuncGraphSetOutputs(ResMgrHandle res_mgr, GraphHandle graph, Handle const outputs[], size_t output_num,
                             bool force_new_ret) {
  if (res_mgr == nullptr || graph == nullptr || outputs == nullptr) {
    MS_LOG(ERROR) << "Input GraphHandle [res_mgr] or [graph] or [outputs] is nullptr.";
    return RET_NULL_PTR;
  }
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    std::vector<AnfNodePtr> out_nodes{NewValueNode(mindspore::prim::kPrimMakeTuple)};
    mindspore::AbstractBasePtrList abs_list{};
    for (size_t i = 0; i < output_num; ++i) {
      auto out_node = GetSrcPtr<AnfNodePtr>(res_mgr, outputs[i]);
      MS_EXCEPTION_IF_NULL(out_node);
      out_nodes.push_back(out_node);
      ConvertConstScalarInputToTensor(out_node);
      abs_list.push_back(out_node->abstract());
    }
    auto make_tuple_cnode = res_fg->NewCNode(out_nodes);
    make_tuple_cnode->set_abstract(std::make_shared<AbstractTupleImpl>(abs_list));
    res_fg->set_output(make_tuple_cnode, force_new_ret);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph set output failed. Error info: " << e.what();
    return RET_ERROR;
  }
  return RET_OK;
}

NodeHandle MSFuncGraphGetOutput(ResMgrHandle res_mgr, ConstGraphHandle graph, size_t i) {
  if (res_mgr == nullptr || graph == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] is nullptr.";
    return nullptr;
  }
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    auto out_node = res_fg->output();
    if (IsPrimitiveCNode(out_node, mindspore::prim::kPrimMakeTuple)) {
      auto out_cnode = out_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(out_cnode);
      auto out_num = out_cnode->size() - 1;
      if (i >= out_num) {
        MS_LOG(ERROR) << "Invalid output index, it should be less than " << out_num << ", but got: " << i;
        return nullptr;
      }
      return GetRawPtr(res_mgr, out_cnode->input(i + 1));
    } else {
      if (i >= 1) {
        MS_LOG(ERROR)
          << "Invalid output index. The graph has only one output, so the output index should be 0, but got: " << i;
        return nullptr;
      }
      return GetRawPtr(res_mgr, out_node);
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph get output failed. Error info: " << e.what();
    return nullptr;
  }
}

size_t MSFuncGraphGetOutputNum(ResMgrHandle res_mgr, ConstGraphHandle graph, STATUS *error) {
  if (error == nullptr) {
    MS_LOG(ERROR) << "Input status flag [error] is nullptr.";
    return 0;
  }
  if (res_mgr == nullptr || graph == nullptr) {
    MS_LOG(ERROR) << "Input GraphHandle [res_mgr] or [graph] or [error] is nullptr.";
    *error = RET_NULL_PTR;
    return 0;
  }
  size_t out_num = 0;
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    auto out_node = res_fg->output();
    if (IsPrimitiveCNode(out_node, mindspore::prim::kPrimMakeTuple)) {
      auto out_cnode = out_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(out_cnode);
      out_num = out_cnode->size() - 1;
    } else {
      out_num = 1;
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph set output failed. Error info: " << e.what();
    *error = RET_ERROR;
    return 0;
  }
  return out_num;
}

STATUS MSFuncGraphGetOutputs(ResMgrHandle res_mgr, ConstGraphHandle graph, NodeHandle outputs[], size_t output_num) {
  if (res_mgr == nullptr || graph == nullptr || outputs == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] or [inputs] is nullptr.";
    return RET_NULL_PTR;
  }
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    size_t out_num = 0;
    auto out_node = res_fg->output();
    auto out_cnode = out_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(out_cnode);
    if (IsPrimitiveCNode(out_node, mindspore::prim::kPrimMakeTuple)) {
      out_num = out_cnode->size() - 1;
    } else {
      out_num = 1;
    }
    if (out_num != output_num) {
      MS_LOG(ERROR) << "Invalid output number, it should be: " << out_num << ", but got: " << output_num;
      return RET_ERROR;
    }
    for (size_t i = 0; i < output_num; i++) {
      outputs[i] = GetRawPtr(res_mgr, out_cnode->input(i + 1));
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph get inputs failed. Error info: " << e.what();
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS MSFuncGraphReplace(ResMgrHandle res_mgr, GraphHandle graph, ConstNodeHandle old_node, ConstNodeHandle new_node) {
  if (res_mgr == nullptr || graph == nullptr || old_node == nullptr || new_node == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] or [old_node] or [new_node] is nullptr.";
    return RET_NULL_PTR;
  }
  try {
    auto res_fg = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(res_fg);
    auto manager_ptr = mindspore::Manage(res_fg, true);
    MS_EXCEPTION_IF_NULL(manager_ptr);
    auto res_old_anfnode = GetSrcPtr<AnfNodePtr>(res_mgr, old_node);
    MS_EXCEPTION_IF_NULL(res_old_anfnode);
    auto res_new_anfnode = GetSrcPtr<AnfNodePtr>(res_mgr, new_node);
    MS_EXCEPTION_IF_NULL(res_new_anfnode);
    manager_ptr->Replace(res_old_anfnode, res_new_anfnode);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph replace failed. Error info: " << e.what();
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS MSFuncGraphCompile(ResMgrHandle res_mgr, GraphHandle graph) {
  if (res_mgr == nullptr || graph == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [graph] is nullptr.";
    return RET_NULL_PTR;
  }
  auto res_mgr_ptr = reinterpret_cast<ResourceManager *>(res_mgr);
  auto context_ptr = mindspore::MsContext::GetInstance();
  try {
    auto func_graph = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(func_graph);
    auto fg_mgr = mindspore::MakeManager();
    fg_mgr->AddFuncGraph(func_graph, true);
    MS_EXCEPTION_IF_NULL(fg_mgr);
    func_graph->set_manager(fg_mgr);
    (void)mindspore::LiftingClone(func_graph);
    context_ptr->Refresh();
    std::string backend_name = context_ptr->backend_policy();
    std::string target = context_ptr->get_param<std::string>(mindspore::MS_CTX_DEVICE_TARGET);
    uint32_t device_id = context_ptr->get_param<uint32_t>(mindspore::MS_CTX_DEVICE_ID);
    auto backend = std::make_shared<mindspore::compile::MindRTBackend>(backend_name, target, device_id);
    res_mgr_ptr->SetBackend(backend);
    if (target == mindspore::kAscendDevice &&
        context_ptr->get_param<int>(mindspore::MS_CTX_EXECUTION_MODE) == mindspore::kPynativeMode) {
      backend->set_is_multi_graph_sink(false);
    }
    // TODO(XianglongZeng): SetRunMode()
    auto actor_info = backend->CompileGraphs(func_graph);
    res_mgr_ptr->SetResult(mindspore::pipeline::kOutput, actor_info);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph compile failed. Error info: " << e.what();
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS MSFuncGraphRun(ResMgrHandle res_mgr, GraphHandle graph, TensorHandle const inputs[], size_t input_num,
                      TensorHandle outputs[], size_t outputs_num) {
  if (res_mgr == nullptr || inputs == nullptr || outputs == nullptr) {
    MS_LOG(ERROR) << "Input Handle [res_mgr] or [inputs] or [outputs] is nullptr.";
    return RET_NULL_PTR;
  }
  mindspore::VectorRef args;
  for (size_t i = 0; i < input_num; i++) {
    auto in_tensor = GetSrcPtr<TensorPtr>(res_mgr, inputs[i]);
    if (in_tensor == nullptr) {
      MS_LOG(ERROR) << "Invalid input. Index: " << i;
      return RET_NULL_PTR;
    }
    args.push_back(in_tensor);
  }
  auto res_mgr_ptr = reinterpret_cast<ResourceManager *>(res_mgr);
  auto raw_info = res_mgr_ptr->GetResult(mindspore::pipeline::kOutput);
  auto bc_ptr = res_mgr_ptr->GetBackend();
  auto mindrt_bc_ptr = std::dynamic_pointer_cast<mindspore::compile::MindRTBackend>(bc_ptr);
  mindspore::VectorRef out_vec;
  try {
    auto func_graph = GetSrcPtr<FuncGraphPtr>(res_mgr, graph);
    MS_EXCEPTION_IF_NULL(func_graph);
    auto params_anf = func_graph->parameters();
    for (auto p : params_anf) {
      auto param = p->cast<ParameterPtr>();
      if (param->has_default()) {
        auto value_ptr = param->default_param();
        auto tensor_ptr = value_ptr->cast<TensorPtr>();
        args.push_back(tensor_ptr);
      }
    }
    const auto actor_info = raw_info.cast<mindspore::compile::ActorInfo>();
    mindrt_bc_ptr->RunGraph(actor_info, args, &out_vec);
    const std::vector<TensorPtr> &ref_outputs = ConvertOutputToTensor(out_vec);
    if (ref_outputs.size() != outputs_num) {
      MS_LOG(ERROR) << "Invalid outputs number, it should be: " << ref_outputs.size() << ", but got: " << outputs_num;
      return RET_ERROR;
    }
    for (size_t i = 0; i < outputs_num; ++i) {
      outputs[i] = GetRawPtr(res_mgr, ref_outputs[i]);
    }
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "FuncGraph compile failed. Error info: " << e.what();
    return RET_ERROR;
  }
  return RET_OK;
}
