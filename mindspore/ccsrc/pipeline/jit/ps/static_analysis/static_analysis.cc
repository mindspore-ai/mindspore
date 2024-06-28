/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include <algorithm>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_set>
#include <utility>
#include <atomic>
#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "abstract/abstract_value.h"
#include "pipeline/jit/ps/fallback.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "pipeline/jit/ps/static_analysis/prim.h"
#include "frontend/operator/ops.h"
#include "utils/ms_exception.h"
#include "utils/compile_config.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/jit/ps/static_analysis/evaluator.h"
#include "pipeline/jit/ps/debug/trace.h"
#include "include/common/fallback.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/ps/static_analysis/async_eval_result.h"
#include "frontend/operator/ops_front_infer_function.h"
#include "frontend/operator/composite/composite.h"
#include "ops/op_def.h"

namespace mindspore {
namespace abstract {
// Record current depth of function call stack, including `stack_frame_depth`.
std::atomic<size_t> function_call_depth;
// Record current depth of stack frames call.
std::atomic<size_t> stack_frame_depth;

void ResetFunctionCallDepth() { function_call_depth = 0; }

void IncreaseFunctionCallDepth() { (void)(++function_call_depth); }

void DecreaseFunctionCallDepth() {
  if (function_call_depth == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "Current function call depth is already 0, can not decrease it.";
  }
  function_call_depth--;
}

size_t FunctionCallDepth() { return function_call_depth; }

void ResetStackFrameDepth() { stack_frame_depth = 0; }

void IncreaseStackFrameDepth() { (void)(++stack_frame_depth); }

void DecreaseStackFrameDepth() {
  if (stack_frame_depth == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "Current stack frame depth is already 0, can not decrease it.";
  }
  stack_frame_depth--;
}

size_t StackFrameDepth() { return stack_frame_depth; }

namespace {
void ExecEvaluator(EvaluatorPtr eval, AnalysisEnginePtr engine, ConfigPtrList args_conf_list, AnfNodeConfigPtr out_conf,
                   std::string thread_id, AsyncAbstractPtr async_result_branch, AsyncAbstractPtr async_result_main,
                   AsyncInferTaskPtr async_task, trace::TraceGraphEvalStack graph_evals,
                   trace::TraceCNodeEvalStack trace_c_node_evals) {
  MS_EXCEPTION_IF_NULL(eval);
  MS_EXCEPTION_IF_NULL(async_task);
  AnalysisSchedule::set_thread_id(thread_id);
  // Restore trace stack for dump stack when there is exception.
  trace::TraceEvalCNodeStackPrepare(trace_c_node_evals);
  trace_c_node_evals.clear();
  trace::TraceGraphEvalStackPrepare(graph_evals);
  graph_evals.clear();

  try {
    // Wait for Signal to run
    MS_LOG(DEBUG) << async_task.get() << "  " << eval->ToString() << " waiting.";
    (void)async_task->GetResult();
    MS_LOG(DEBUG) << async_task.get() << "  " << eval->ToString() << " running.";

    // Acquire GIL for eval to callback python.
    EvalResultPtr result;
    {
      MS_LOG(DEBUG) << eval->ToString() << "_" << AnalysisSchedule::thread_id() << " begin.";
      py::gil_scoped_acquire py_guard;
      result = eval->Run(engine, args_conf_list, out_conf);
    }
    MS_LOG(DEBUG) << eval->ToString() << "_" << AnalysisSchedule::thread_id() << " end.";
    MS_EXCEPTION_IF_NULL(result);
    MS_EXCEPTION_IF_NULL(result->abstract());

    // Check the branch value to be compatible with the other branch value.
    AnalysisResultCacheMgr::GetInstance().CheckSwitchValueJoinable(out_conf, result->abstract());
    // Broaden the result of switch(c,t,f)()
    auto broaden_abstract = result->abstract()->Broaden();

    MS_EXCEPTION_IF_NULL(async_result_branch);
    MS_EXCEPTION_IF_NULL(async_result_main);
    // Notify the thread of waiting for branch value and the main thread to continue.
    async_result_branch->set_result(broaden_abstract);
    async_result_main->set_result(broaden_abstract);
    MS_LOG(DEBUG) << GetInferThread() << " async :" << eval->ToString()
                  << " asyncResult address = " << async_result_branch.get();
    if (async_result_branch->TryGetResult()) {
      MS_LOG(DEBUG) << "value = " << (async_result_branch->TryGetResult())->ToString();
    } else {
      MS_LOG(DEBUG) << "value = null.";
    }
  } catch (const std::exception &ex) {
    MS_EXCEPTION_IF_NULL(out_conf->node());
    MS_LOG(INFO) << GetInferThread() << "Eval node: " << out_conf->node()->ToString() << "  " << eval->ToString()
                 << " threw exception: " << ex.what();
    AnalysisSchedule::GetInstance().HandleException(ex);
  }
  trace::ClearTraceStack();
  ClearThreadLocal();
  MS_LOG(DEBUG) << AnalysisSchedule::thread_id() << " exited.";
  // Thread number will be drop when thread exits.
  AnalysisSchedule::GetInstance().DecreaseThreadCount();
}

AbstractBasePtr BuildAsyncAbstractRecursively(const AbstractBasePtr &orig_abs,
                                              const std::vector<AsyncAbstractPtr> &pending_async_abstract_list,
                                              const std::vector<std::size_t> &index) {
  MS_EXCEPTION_IF_NULL(orig_abs);
  auto sequence_abs = dyn_cast_ptr<AbstractSequence>(orig_abs);
  if (sequence_abs != nullptr) {
    const auto &orig_elements = sequence_abs->elements();
    AbstractBasePtrList new_elements;
    for (size_t i = 0; i < orig_elements.size(); ++i) {
      MS_EXCEPTION_IF_NULL(orig_elements[i]);
      if (orig_elements[i]->isa<AbstractFuncAtom>()) {
        AbstractFuncAtomPtrList abs_func_list{orig_elements[i]->cast<AbstractFuncAtomPtr>()};
        for (size_t j = 0; j < pending_async_abstract_list.size(); ++j) {
          std::vector<std::size_t> new_index(index);
          new_index.push_back(i);
          auto async_func = AsyncAbstractFuncAtom::MakeShared(pending_async_abstract_list[j], new_index);
          abs_func_list.push_back(async_func);
        }
        new_elements.push_back(AbstractFunction::MakeAbstractFunction(abs_func_list));
      } else if (orig_elements[i]->isa<AbstractSequence>()) {
        std::vector<std::size_t> new_index(index);
        new_index.push_back(i);
        new_elements.push_back(BuildAsyncAbstractRecursively(orig_elements[i], pending_async_abstract_list, new_index));
      } else {
        new_elements.push_back(orig_elements[i]);
      }
    }
    static const auto enable_eliminate_unused_element = (common::GetCompileConfig("ENABLE_DDE") != "0");
    AbstractBasePtr new_abs;
    if (orig_abs->isa<AbstractTuple>()) {
      new_abs = std::make_shared<AbstractTuple>(
        new_elements, (enable_eliminate_unused_element ? sequence_abs->sequence_nodes() : nullptr));
    } else if (orig_abs->isa<AbstractList>()) {
      new_abs = std::make_shared<AbstractList>(
        new_elements, (enable_eliminate_unused_element ? sequence_abs->sequence_nodes() : nullptr));
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "FirstResult is not AbstractTuple or AbstractList, but: " << orig_abs->ToString();
    }
    return new_abs;
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Orig abstract is not AbstractTuple or AbstractList, but: " << orig_abs->ToString();
}

void BuildPossibleSpecs(const AbstractBasePtr &first_result,
                        const std::vector<AsyncAbstractPtr> &branch_async_abstract_list,
                        AbstractBasePtrList *out_abs_list) {
  MS_EXCEPTION_IF_NULL(out_abs_list);
  MS_EXCEPTION_IF_NULL(first_result);
  std::vector<AsyncAbstractPtr> pending_async_abstract_list;
  std::size_t len = branch_async_abstract_list.size();

  for (size_t i = 0; i < len; ++i) {
    AbstractBasePtr result;
    MS_EXCEPTION_IF_NULL(branch_async_abstract_list[i]);
    if (enable_waiting_branch_eval()) {
      result = branch_async_abstract_list[i]->GetResult();
    } else {
      result = branch_async_abstract_list[i]->TryGetResult();
    }

    if (result) {
      if (result->isa<AsyncAbstractFuncAtom>()) {
        branch_async_abstract_list[i]->ClearPossibleResult();
        pending_async_abstract_list.push_back(branch_async_abstract_list[i]);
        MS_LOG(DEBUG) << "Pending add: " << branch_async_abstract_list[i].get() << "_"
                      << branch_async_abstract_list[i]->ToString();
      } else {
        out_abs_list->push_back(result);
      }
    } else {
      pending_async_abstract_list.push_back(branch_async_abstract_list[i]);
      MS_LOG(DEBUG) << "Pending add: " << branch_async_abstract_list[i].get() << "_"
                    << branch_async_abstract_list[i]->ToString();
    }
  }

  if (first_result->isa<AbstractFunction>()) {
    for (std::size_t j = 0; j < pending_async_abstract_list.size(); ++j) {
      auto async_func = AsyncAbstractFuncAtom::MakeShared(pending_async_abstract_list[j], std::vector<size_t>{0});
      out_abs_list->push_back(async_func);
      MS_LOG(DEBUG) << "out_abs_list add: " << async_func.get() << "_" << async_func->ToString();
    }
  } else if (first_result->isa<AbstractSequence>()) {
    const auto &new_first_result =
      BuildAsyncAbstractRecursively(first_result, pending_async_abstract_list, std::vector<size_t>());
    MS_LOG(DEBUG) << GetInferThread() << " Try to replace old first with new one, old: " << first_result->ToString()
                  << ", new: " << new_first_result->ToString();
    std::replace_if(
      out_abs_list->begin(), out_abs_list->end(),
      [first_result](const auto &element) { return element == first_result; }, new_first_result);
  } else {
    MS_LOG(DEBUG) << GetInferThread() << " wait for normal async result";
  }
}

EvalResultPtr ConvertToPyInterpretCall(const CNodePtr &cnode, const AnfNodeConfigPtr &conf,
                                       const AnfNodePtr &func_node = nullptr) {
  auto fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto out_node = conf->node();
  MS_EXCEPTION_IF_NULL(out_node);
  std::stringstream script_buffer;
  AnfNodePtrList local_key_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  AnfNodePtrList local_value_inputs = {NewValueNode(prim::kPrimMakeTuple)};

  // Handle call function
  const std::string call_func_str = "__call_func_str__";
  constexpr size_t call_func_index = 0;
  script_buffer << call_func_str << "(";
  (void)local_key_inputs.emplace_back(NewValueNode(call_func_str));
  if (func_node == nullptr) {
    (void)local_value_inputs.emplace_back(cnode->input(call_func_index));
  } else {
    (void)local_value_inputs.emplace_back(func_node);
  }

  // Handle inputs.
  const std::string call_prefix = "__input_";
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto cur_node = cnode->input(i);
    if (IsPrimitiveCNode(cur_node, prim::kPrimMakeKeywordArg)) {
      const std::string value_cur_str = call_prefix + "_value_" + std::to_string(i - 1) + "__";
      constexpr size_t key_inputs_index = 1;
      constexpr size_t value_inputs_index = 2;
      constexpr size_t expect_inputs_size = 3;
      if (cur_node->cast<CNodePtr>()->size() != expect_inputs_size) {
        MS_LOG(INTERNAL_EXCEPTION) << "The make_keyword_arg node should have " << expect_inputs_size
                                   << " inputs, but got " << cnode->size();
      }
      auto key_node = cur_node->cast<CNodePtr>()->input(key_inputs_index);
      if (!IsValueNode<StringImm>(key_node)) {
        MS_LOG(INTERNAL_EXCEPTION) << "The key in make_keyword args must be string, but got "
                                   << key_node->DebugString();
      }
      auto key_string = GetValue<std::string>(GetValueNode(key_node));
      std::string key_value_str = key_string + "=" + value_cur_str;
      (void)local_key_inputs.emplace_back(NewValueNode(value_cur_str));
      script_buffer << key_value_str << ",";
      auto value_node = cur_node->cast<CNodePtr>()->input(value_inputs_index);
      (void)local_value_inputs.emplace_back(value_node);
    } else {
      const std::string cur_str = call_prefix + std::to_string(i - 1) + "__";
      script_buffer << cur_str << ",";
      (void)local_key_inputs.emplace_back(NewValueNode(cur_str));
      (void)local_value_inputs.emplace_back(cur_node);
    }
  }
  script_buffer << ")";
  const auto &script = script_buffer.str();
  auto local_key_node = fg->NewCNode(local_key_inputs);
  auto local_value_node = fg->NewCNode(local_value_inputs);
  auto local_dict_node = fg->NewCNode({NewValueNode(prim::kPrimMakeDict), local_key_node, local_value_node});
  auto obj_call_node =
    fallback::CreatePyInterpretCNode(fg, script, py::dict(), local_dict_node, out_node->debug_info());
  MS_LOG(DEBUG) << "Created obj_call_node: " << obj_call_node->DebugString();
  AnalysisEnginePtr eng = conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(obj_call_node, conf->context(), conf->func_graph());
  return eng->ForwardConfig(conf, fn_conf);
}

EvalResultPtr ParsePyObjToFunc(const py::object &py_fn, const CNodePtr &cnode, const AnfNodeConfigPtr &conf) {
  FuncGraphPtr func_fg = nullptr;
  {
    MS_LOG_TRY_CATCH_SCOPE;
    func_fg = parse::ParsePythonCode(py_fn);
  }
  if (func_fg != nullptr) {
    auto fg = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    func_fg->set_manager(fg->manager());

    std::vector<AnfNodePtr> new_cnode_inputs;
    (void)new_cnode_inputs.emplace_back(NewValueNode(func_fg));
    for (std::size_t i = 1; i < cnode->size(); ++i) {
      (void)new_cnode_inputs.emplace_back(cnode->input(i));
    }
    auto new_cnode = fg->NewCNodeInOrder(new_cnode_inputs);
    new_cnode->set_debug_info(cnode->debug_info());

    AnalysisEnginePtr eng = conf->engine();
    MS_EXCEPTION_IF_NULL(eng);
    AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_cnode, conf->context(), conf->func_graph());
    return eng->ForwardConfig(conf, fn_conf);
  } else {
    return ConvertToPyInterpretCall(cnode, conf);
  }
}

std::string GetClassName(const py::object &cls_obj) {
  if (py::hasattr(cls_obj, "__class__")) {
    return py::getattr(py::getattr(cls_obj, "__class__"), "__name__").cast<py::str>();
  }
  return py::getattr(cls_obj, "__name__").cast<py::str>();
}

EvalResultPtr ConvertCallPyObjCallFunc(const CNodePtr &cnode, const AbstractBasePtr &abs,
                                       const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(abs);
  auto val = abs->BuildValue();
  MS_EXCEPTION_IF_NULL(val);
  auto warp_obj = dyn_cast_ptr<parse::PyObjectWrapper>(val);
  MS_EXCEPTION_IF_NULL(warp_obj);
  py::object cls_obj = warp_obj->obj();
  auto class_name = GetClassName(cls_obj);
  py::object call_obj = py::none();
  const std::string construct_func_name = "construct";
  if (py::hasattr(cls_obj, common::SafeCStr(construct_func_name)) && py::isinstance<Cell>(cls_obj)) {
    call_obj = py::getattr(cls_obj, common::SafeCStr(construct_func_name));
  } else {
    const std::string call_func_name = "__call__";
    if (py::hasattr(cls_obj, common::SafeCStr(call_func_name))) {
      call_obj = py::getattr(cls_obj, common::SafeCStr(call_func_name));
    }
  }
  if (py::isinstance<py::none>(call_obj)) {
    MS_EXCEPTION(ValueError) << class_name << "is not a callable object";
  }
  return ParsePyObjToFunc(call_obj, cnode, conf);
}

EvalResultPtr ConvertMsClassObjToFunc(const CNodePtr &cnode, const AbstractBasePtr &abs, const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(abs);
  auto val = abs->BuildValue();
  MS_EXCEPTION_IF_NULL(val);
  auto class_val = dyn_cast_ptr<parse::MsClassObject>(val);
  MS_EXCEPTION_IF_NULL(class_val);
  py::object cls_obj = class_val->obj();
  const std::string call_func_name = "__call__";
  if (!py::hasattr(cls_obj, common::SafeCStr(call_func_name))) {
    MS_EXCEPTION(ValueError) << class_val->name() << " has no " << call_func_name
                             << " function, please check the code.";
  }
  py::object call_obj = py::getattr(cls_obj, common::SafeCStr(call_func_name));
  return ParsePyObjToFunc(call_obj, cnode, conf);
}

bool CheckFuncSideEffect(const AbstractFunctionPtr &func) {
  // Check if func graph contains isolated side-effect, and sync.
  auto func_graph_abs = dyn_cast_ptr<FuncGraphAbstractClosure>(func);
  if (func_graph_abs != nullptr) {
    MS_EXCEPTION_IF_NULL(func_graph_abs->func_graph());
    return func_graph_abs->func_graph()->has_side_effect_node();
  } else {
    auto meta_func_graph_abs = dyn_cast_ptr<MetaFuncGraphAbstractClosure>(func);
    if (meta_func_graph_abs != nullptr) {
      MS_EXCEPTION_IF_NULL(meta_func_graph_abs->meta_func_graph());
      return meta_func_graph_abs->meta_func_graph()->has_side_effect_node();
    }
    if (func->isa<abstract::PartialAbstractClosure>()) {
      const auto &abstract_partial_func = func->cast<abstract::PartialAbstractClosurePtr>();
      const auto &abstract_fn = abstract_partial_func->fn();
      MS_EXCEPTION_IF_NULL(abstract_fn);
      return CheckFuncSideEffect(abstract_fn);
    }
  }
  return false;
}

AbstractFuncAtomPtr GetRealFuncAtom(const AbstractFuncAtomPtr &possible_func) {
  MS_EXCEPTION_IF_NULL(possible_func);
  auto real_atom = possible_func;
  const auto &async_abs_func = possible_func->cast_ptr<AsyncAbstractFuncAtom>();
  if (async_abs_func != nullptr) {
    auto real_func = async_abs_func->GetUnique();
    real_atom = dyn_cast<AbstractFuncAtom>(real_func);
    MS_EXCEPTION_IF_NULL(real_atom);
    MS_LOG(DEBUG) << "Real AsyncAbstractFuncAtom is: " << real_atom->ToString();
  }
  return real_atom;
}

template <typename T>
bool Match(const ValuePtr &prim) {
  return prim->isa<T>();
}
using MetaFgMatchFunc = std::function<bool(const ValuePtr &)>;

bool MatchMetaFg(const ValuePtr &prim) {
  static const std::vector<MetaFgMatchFunc> meta_fg_ops{
    Match<prim::GradOperation>,
    Match<prim::VmapOperation>,
    Match<prim::Shard>,
  };
  return std::any_of(meta_fg_ops.cbegin(), meta_fg_ops.cend(),
                     [&prim](const MetaFgMatchFunc &match_func) { return match_func(prim); });
}

void RemoveSequenceFromOrderList(const CNodePtr &origin_cnode) {
  constexpr size_t sequence_input_pos = 2;
  if (origin_cnode->size() <= sequence_input_pos) {
    return;
  }
  auto seq_node = origin_cnode->input(sequence_input_pos);
  auto prim = GetCNodePrimitiveWithoutDoSignature(seq_node);
  if (prim != nullptr &&
      (IsPrimitiveEquals(prim, prim::kPrimMakeTuple) || IsPrimitiveEquals(prim, prim::kPrimMakeList))) {
    auto seq_cnode = dyn_cast<CNode>(seq_node);
    MS_EXCEPTION_IF_NULL(seq_cnode);
    seq_cnode->func_graph()->EraseUnusedNodeInOrder(seq_cnode);
  }
}

AbstractBasePtr GetEvalResult(const AnfNodePtr &node, const AnalysisEnginePtr &engine, const AnfNodeConfigPtr &conf) {
  AnfNodeConfigPtr func_conf = std::make_shared<AnfNodeConfig>(engine, node, conf->context(), conf->func_graph());
  auto possible_func_eval_result = func_conf->ObtainEvalResult();
  MS_EXCEPTION_IF_NULL(possible_func_eval_result);
  return possible_func_eval_result->abstract();
}

bool IsFuncGraphAbstractInput(const CNodePtr &origin_cnode, const AnalysisEnginePtr &engine,
                              const AnfNodeConfigPtr &conf) {
  auto possible_func = GetEvalResult(origin_cnode->input(1), engine, conf);
  if (possible_func == nullptr || !possible_func->isa<FuncGraphAbstractClosure>()) {
    return false;
  }
  // Check whether it is a high order scene such as GradOperation(GradOperation(net)), the meta_unpack_prepare doesn't
  // handle before. To handle this later.
  if (!origin_cnode->input(1)->isa<CNode>()) {
    return true;
  }
  auto input1_cnode = origin_cnode->input(1)->cast<CNodePtr>();
  auto possible_prim = GetEvalResult(input1_cnode->input(0), engine, conf);
  if (possible_prim == nullptr || !possible_prim->isa<PrimitiveAbstractClosure>()) {
    return true;
  }
  auto value = GetValueWithoutDoSignature(possible_prim->cast<PrimitiveAbstractClosurePtr>()->prim());
  return !MatchMetaFg(value);
}

// {{meta_fg, g, w}, Ys} => {{meta_fg, {UnpackGraph, g, Ys}, w}, Ys}
// {UnpackCall, {meta_fg, g, w}, Ys} => {UnpackCall, {meta_fg, {UnpackGraph, g, Ys}, w}, Ys}
AnfNodePtr InsertUnpackGraph(const CNodePtr &origin_cnode, const ValuePtr &value, const AnfNodeConfigPtr &conf,
                             const AnalysisEnginePtr &engine) {
  // origin_cnode is {meta_fg, g, ...}
  const size_t inputs_x_minimum_size = 2;
  if (origin_cnode->size() < inputs_x_minimum_size) {
    return nullptr;
  }

  if (value == nullptr || !MatchMetaFg(value)) {
    return nullptr;
  }

  if (!IsFuncGraphAbstractInput(origin_cnode, engine, conf)) {
    return nullptr;
  }

  auto manager = conf->engine()->func_graph_manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto node_users = manager->node_users()[origin_cnode];
  if (node_users.empty()) {
    return nullptr;
  }
  auto meta_user = node_users.begin()->first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(meta_user);
  int index = node_users.begin()->second;
  if (index != 0 && index != 1) {
    return nullptr;
  }

  bool need_unpack_args = false;
  if (index == 1) {
    // The meta_fg user node should be UnpackCall.
    auto input0_value = GetValueWithoutDoSignature(meta_user->input(0));
    if (input0_value == nullptr || !input0_value->isa<prim::UnpackCall>()) {
      return nullptr;
    }
    need_unpack_args = true;
  }
  // Create UnpackGraph node.
  bool sens_param = false;
  if (value->isa<prim::GradOperation>()) {
    sens_param = value->cast<prim::GradOperationPtr>()->sens_param();
    RemoveSequenceFromOrderList(origin_cnode);
  }
  auto unpack_graph = std::make_shared<prim::UnpackGraphPrimitive>(sens_param, need_unpack_args);
  std::vector<AnfNodePtr> unpack_graph_inputs{NewValueNode(unpack_graph), origin_cnode->input(1)};
  const auto &meta_user_inputs = meta_user->inputs();
  constexpr int64_t unpack_inputs_begin_index = 2;
  int64_t offset = (need_unpack_args ? unpack_inputs_begin_index : 1);
  (void)std::transform(meta_user_inputs.begin() + offset, meta_user_inputs.end(),
                       std::back_inserter(unpack_graph_inputs),
                       [](const AnfNodePtr &node) -> AnfNodePtr { return node; });
  auto fg = origin_cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto unpack_graph_node = fg->NewCNodeBefore(meta_user, unpack_graph_inputs);
  // Create new call_node.
  auto new_cnode_inputs = origin_cnode->inputs();
  new_cnode_inputs[1] = unpack_graph_node;
  auto new_cnode = fg->NewCNodeBefore(meta_user, new_cnode_inputs);
  return new_cnode;
}
}  // namespace

EvalResultPtr PrimitiveEvalCache::Get(const PrimitivePtr &prim, const AbstractBasePtrList &args) const {
  MS_EXCEPTION_IF_NULL(prim);
  std::lock_guard<std::mutex> guard(mutex_);
  auto cache_iter = prim_cache_.find(prim->name());
  if (cache_iter == prim_cache_.end()) {
    return nullptr;
  }
  auto &cache = cache_iter->second;
  auto iter = cache.find(PrimitiveEvalCacheKey{prim->attrs(), args});
  if (iter == cache.end()) {
    return nullptr;
  }
  return iter->second;
}

void PrimitiveEvalCache::Put(const PrimitivePtr &prim, AttrValueMap &&attrs, const AbstractBasePtrList &args,
                             const EvalResultPtr &result) {
  MS_EXCEPTION_IF_NULL(prim);
  std::lock_guard<std::mutex> guard(mutex_);
  (void)prim_cache_[prim->name()].emplace(PrimitiveEvalCacheKey{std::move(attrs), args}, result);
}

void PrimitiveEvalCache::Clear() {
  std::lock_guard<std::mutex> guard(mutex_);
  prim_cache_.clear();
}

AnalysisResult AnalysisEngine::Run(const FuncGraphPtr &func_graph, const AbstractBasePtrList &args_abs_list) {
  StaticAnalysisException::Instance().ClearException();
  AnalysisResult result;
  try {
    MS_EXCEPTION_IF_NULL(func_graph);
    ConfigPtrList args_conf_list;
    (void)std::transform(args_abs_list.begin(), args_abs_list.end(), std::back_inserter(args_conf_list),
                         [](const AbstractBasePtr &arg) -> ConfigPtr { return std::make_shared<VirtualConfig>(arg); });
    MS_EXCEPTION_IF_NULL(func_graph_manager_);
    func_graph_manager_->AddFuncGraph(func_graph);
    root_func_graph_ = func_graph;

    // Running the analyzer.
    ResetFunctionCallDepth();
    ResetStackFrameDepth();
    // Create a new root dummy context for the new analysis session.
    AnalysisContextPtr dummy_context = AnalysisContext::NewDummyContext();
    MS_LOG(DEBUG) << func_graph->ToString() << ": Run begin.";
    AnalysisContextPtr root_context = Run(func_graph, dummy_context, args_conf_list);
    AnalysisSchedule::GetInstance().Wait();
    MS_EXCEPTION_IF_NULL(root_context);
    auto root_context_fg = root_context->func_graph();
    MS_EXCEPTION_IF_NULL(root_context_fg);
    AnfNodeConfigPtr output_conf = MakeConfig(root_context_fg->get_return(), root_context, root_context_fg);
    MS_LOG(DEBUG) << func_graph->ToString() << ": Run finished.";

    MS_EXCEPTION_IF_NULL(output_conf);
    auto eval_result = output_conf->ObtainEvalResult();
    result.eval_result = eval_result;
    result.context = root_context;
  } catch (const std::exception &ex) {
    MS_LOG(INFO) << "Eval " << func_graph->ToString() << " threw exception.\n" << ex.what();
    AnalysisSchedule::GetInstance().HandleException(ex);
  }
  AnalysisSchedule::GetInstance().Wait();
  MS_LOG(DEBUG) << func_graph->ToString() << ": Run end.";
  // Set the sequence nodes' elements use flags all true.
  SetSequenceElementsUseFlagsRecursively(result.eval_result->abstract(), true);
  MS_LOG(DEBUG) << func_graph->ToString() << ":SetSequenceElementsUseFlagsRecursively Run end.";
  return result;
}

AnalysisContextPtr AnalysisEngine::Run(const FuncGraphPtr &func_graph, const AnalysisContextPtr &context,
                                       const ConfigPtrList &args_conf_list) {
  auto evaluator = std::make_shared<FuncGraphEvaluator>(func_graph, context);
  (void)evaluator->Run(shared_from_this(), args_conf_list, nullptr);
  return root_context_;
}

EvalResultPtr ObtainEvalResultFromCache(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  static AnalysisResultCacheMgr &cache_mgr = AnalysisResultCacheMgr::GetInstance();
  auto result = cache_mgr.GetValue(conf);
  if (result != nullptr) {
    MS_EXCEPTION_IF_NULL(result->abstract());
    MS_LOG(DEBUG) << "Evaluate cache found for NodeConfig: " << conf->ToString()
                  << ", result: " << result->abstract().get() << "/" << result->abstract()->ToString();
    return result;
  }
  return nullptr;
}

EvalResultPtr AnalysisEngine::ObtainEvalResultWithCache(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  auto result = ObtainEvalResultFromCache(conf);
  if (result != nullptr) {
    return result;
  }
  MS_LOG(DEBUG) << "Evaluate cache miss for NodeConfig: " << conf->ToString();
  result = ObtainEvalResultWithoutCache(conf);
  return result;
}

EvalResultPtr AnalysisEngine::ObtainEvalResultWithoutCache(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  EvalResultPtr result = nullptr;
  result = Eval(conf);
  if (result == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Evaluate for NodeConfig " << conf->ToString() << " get nullptr";
  }
  MS_EXCEPTION_IF_NULL(result->abstract());
  MS_LOG(DEBUG) << "Always Evaluate node for NodeConfig: " << conf->ToString()
                << ", result: " << result->abstract().get() << "/" << result->abstract()->ToString();
  SaveEvalResultInCache(conf, result);
  return result;
}

void AnalysisEngine::SaveEvalResultInCache(const AnfNodeConfigPtr &conf, const EvalResultPtr &result) const {
  MS_EXCEPTION_IF_NULL(conf);
  MS_EXCEPTION_IF_NULL(result);
  static AnalysisResultCacheMgr &cache_mgr = AnalysisResultCacheMgr::GetInstance();
  auto iter = cache_mgr.GetCache().find(conf);
  if (iter != cache_mgr.GetCache().end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    MS_EXCEPTION_IF_NULL(iter->second->abstract());
    MS_LOG(DEBUG) << "Found previous result for NodeConfig: " << conf->ToString()
                  << ", result: " << iter->second->abstract().get() << "/" << iter->second->abstract()->ToString();
    // Update sequence nodes info, if matched in cache.
    static const auto enable_eliminate_unused_element = (common::GetCompileConfig("ENABLE_DDE") != "0");
    if (enable_eliminate_unused_element) {
      auto new_sequence = dyn_cast<AbstractSequence>(result->abstract());
      auto old_sequence = dyn_cast<AbstractSequence>(iter->second->abstract());
      if (old_sequence != nullptr && new_sequence != nullptr) {
        MS_LOG(DEBUG) << "Before synchronize sequence nodes use flags for NodeConfig: " << conf->ToString()
                      << ", old_sequence: " << old_sequence->ToString()
                      << ", new_sequence: " << new_sequence->ToString();
        SynchronizeSequenceElementsUseFlagsRecursively(old_sequence, new_sequence);
        MS_LOG(DEBUG) << "After synchronize sequence nodes use flags for NodeConfig: " << conf->ToString()
                      << ", old_sequence: " << old_sequence->ToString()
                      << ", new_sequence: " << new_sequence->ToString();
      }
    }
  }
  MS_EXCEPTION_IF_NULL(result->abstract());
  MS_LOG(DEBUG) << "Save result for NodeConfig: " << conf->ToString() << ", result: " << result->abstract().get() << "/"
                << result->abstract()->ToString();
  cache_mgr.SetValue(conf, result);
}

void SynchronizeSequenceElementsUseFlagsForFuncGraphArgs(const AnalysisEnginePtr &engine, const FuncGraphPtr &fg,
                                                         const CNodePtr &cnode,
                                                         const AbstractFunctionPtr &base_func_graph_func,
                                                         const AnalysisContextPtr &fg_context) {
  // Get the evaluator for func graph.
  auto evaluator = engine->GetEvaluatorFor(base_func_graph_func);
  MS_EXCEPTION_IF_NULL(evaluator);

  AbstractBasePtrList args_abs_list;
  for (std::size_t i = 1; i < cnode->size(); i++) {
    auto config = engine->MakeConfig(cnode->input(i), fg_context, fg);
    auto result = config->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(result);
    auto abs = result->abstract();
    args_abs_list.push_back(abs);
  }

  // Check if already evaluated before.
  MS_EXCEPTION_IF_NULL(evaluator->evaluator_cache_mgr());
  auto &cache = evaluator->evaluator_cache_mgr()->GetCache();
  auto iter = cache.find(args_abs_list);
  if (iter != cache.end()) {
    MS_EXCEPTION_IF_NULL(fg_context);
    MS_LOG(DEBUG) << "Eval before, current_node: " << cnode->DebugString() << ", context: " << fg_context->ToString()
                  << ", args: " << args_abs_list;
    // Update inputs sequence nodes info, if matched in cache.
    for (std::size_t i = 0; i < args_abs_list.size(); ++i) {
      auto new_sequence = dyn_cast<AbstractSequence>(args_abs_list[i]);
      auto old_sequence = dyn_cast<AbstractSequence>(iter->first[i]);
      if (old_sequence != nullptr && new_sequence != nullptr) {
        MS_LOG(DEBUG) << "Before synchronize sequence nodes use flags, old_sequence: " << old_sequence->ToString()
                      << ", new_sequence: " << new_sequence->ToString();
        SynchronizeSequenceElementsUseFlagsRecursively(old_sequence, new_sequence);
        MS_LOG(DEBUG) << "After synchronize sequence nodes use flags, old_sequence: " << old_sequence->ToString()
                      << ", new_sequence: " << new_sequence->ToString();
      }
    }
  }
}

EvalResultPtr AnalysisEngine::Eval(const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  AnfNodePtr node = conf->node();
  EvalResultPtr eval_result = nullptr;
#ifdef DEBUG
  compute_conf_stack_.push_back(node);
  std::ostringstream buffer;
  buffer << "Compute Config Begin:";
  for (auto iter : compute_conf_stack_) {
    buffer << " -> " << iter->DebugString();
  }
  MS_LOG(DEBUG) << buffer.str();
#endif
  MS_LOG(DEBUG) << "Begin Eval NodeConfig " << conf->ToString();
  MS_EXCEPTION_IF_NULL(node);
  if (node->abstract() != nullptr) {
    MS_LOG(DEBUG) << "Return old abstract: " << node->DebugString();
    eval_result = std::make_shared<EvalResult>(node->abstract(), std::make_shared<AttrValueMap>());
  } else if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    auto abstract = EvalValueNode(value_node, conf);
    eval_result = std::make_shared<EvalResult>(abstract, std::make_shared<AttrValueMap>());
  } else if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    trace::TraceEvalCNodeEnter(conf);
    MS_LOG(DEBUG) << "Begin Eval CNode: " << cnode->DebugString();
    eval_result = EvalCNode(cnode, conf);
    MS_LOG(DEBUG) << "End Eval CNode: " << cnode->DebugString();
    trace::TraceEvalCNodeLeave();
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Illegal AnfNode for evaluating, node: " << node->DebugString()
                               << "(type:" << node->type_name() << "), fg: "
                               << (node->func_graph() != nullptr ? node->func_graph()->ToString() : "nullgraph")
                               << " conf: " << conf->ToString();
  }

#ifdef DEBUG
  compute_conf_stack_.pop_back();
  if (eval_result == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Compute Config failed, node: " << node->DebugString()
                               << " NodeInfo: " << trace::GetDebugInfo(node->debug_info());
  }
#endif
  MS_EXCEPTION_IF_NULL(eval_result->abstract());
  MS_LOG(DEBUG) << "End Eval NodeConfig " << conf->ToString() << ", res: " << eval_result->abstract()->ToString();
  return eval_result;
}

AbstractBasePtr AnalysisEngine::EvalValueNode(const ValueNodePtr &value_node, const AnfNodeConfigPtr &conf) const {
  MS_EXCEPTION_IF_NULL(conf);
  MS_EXCEPTION_IF_NULL(value_node);
  auto out = ToAbstract(value_node->value(), conf->context(), conf);
  if (value_node->has_new_value() && out->isa<AbstractTensor>()) {
    out = out->Broaden();
  }
  return out;
}

AnfNodeConfigPtr AnalysisEngine::GetForwardConfig(const AnfNodeConfigPtr &conf) const {
  MS_EXCEPTION_IF_NULL(conf);
  AnfNodeConfigPtr new_conf = conf;
  auto conf_iter = anfnode_config_map().find(conf);
  while (conf_iter != anfnode_config_map().end()) {
    new_conf = conf_iter->second;
    MS_EXCEPTION_IF_NULL(new_conf);
    conf_iter = anfnode_config_map().find(new_conf);
  }
  return new_conf;
}

EvalResultPtr AnalysisEngine::InterpretedNodeCall(const CNodePtr &cnode, const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "CNode inputs should not be empty, CNode: " << cnode->DebugString();
  }

  // Check if the operator input is PyExecute CNode.
  const auto &func_node = cnode->input(0);
  MS_EXCEPTION_IF_NULL(func_node);
  constexpr auto recursive_level = 2;
  MS_LOG(DEBUG) << "Current CNode: " << cnode->DebugString(recursive_level)
                << ", func_node: " << func_node->DebugString(recursive_level);
  auto prim = GetCNodePrimitiveWithoutDoSignature(func_node);
  if (!IsPrimitiveEquals(prim, prim::kPrimResolve) && !IsPrimitiveEquals(prim, prim::kPrimGetAttr) &&
      !IsPrimitiveEquals(prim, prim::kPrimPyExecute) && !IsPrimitiveEquals(prim, prim::kPrimPyInterpret)) {
    // Optimize the performance.
    return nullptr;
  }
  AnfNodeConfigPtr func_conf = MakeConfig(func_node, conf->context(), conf->func_graph());
  MS_EXCEPTION_IF_NULL(func_conf);
  const auto &forwarded_conf = GetForwardConfig(func_conf);
  if (!IsPrimitiveCNode(forwarded_conf->node(), prim::kPrimPyExecute) &&
      !IsPrimitiveCNode(forwarded_conf->node(), prim::kPrimPyInterpret)) {
    return nullptr;
  }

  if (IsPrimitiveEquals(prim, prim::kPrimResolve)) {
    return ConvertToPyInterpretCall(cnode, conf, forwarded_conf->node());
  }
  // Forward getattr CNode call to PyInterpreted CNode.
  return ConvertToPyInterpretCall(cnode, conf);
}

AbstractBasePtr AnalysisEngine::GetCNodeOperatorAbstract(const CNodePtr &cnode, const AnalysisContextPtr &context,
                                                         const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "CNode inputs should not be empty, CNode: " << cnode->DebugString();
  }
  auto &func_node = cnode->input(0);
  MS_EXCEPTION_IF_NULL(func_node);
  MS_LOG(DEBUG) << "Current CNode function: " << func_node->DebugString();
  AnfNodeConfigPtr func_conf = MakeConfig(func_node, context, func_graph);
  MS_EXCEPTION_IF_NULL(func_conf);
  // Keep it in a local variable, otherwise smart pointer will free it.
  auto possible_func_eval_result = func_conf->ObtainEvalResult();
  MS_EXCEPTION_IF_NULL(possible_func_eval_result);
  auto &possible_func = possible_func_eval_result->abstract();
  if (possible_func == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "No abstract, func_conf: " << func_conf->ToString();
  }
  return possible_func;
}

EvalResultPtr AnalysisEngine::ConvertClassTypeToFunc(const CNodePtr &cnode, const AbstractBasePtr &abs,
                                                     const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(cnode);
  const auto inputs_size = cnode->size();
  AbstractBasePtrList input_abs;
  input_abs.reserve(inputs_size - 1);
  for (std::size_t i = 1; i < inputs_size; ++i) {
    const AnfNodePtr &node = cnode->input(i);
    auto cur_config = MakeConfig(node, conf->context(), conf->func_graph());
    const auto &cur_eval_result = cur_config->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(cur_eval_result);
    auto cur_abs = cur_eval_result->abstract();
    MS_EXCEPTION_IF_NULL(cur_abs);
    input_abs.push_back(cur_abs);
  }
  bool has_non_graph_input = std::any_of(input_abs.begin(), input_abs.end(), [](const AbstractBasePtr &abs) {
    MS_EXCEPTION_IF_NULL(abs);
    return abs->isa<abstract::AbstractAny>() || abs->BuildValue()->isa<parse::InterpretedObject>();
  });
  if (has_non_graph_input) {
    return ConvertToPyInterpretCall(cnode, conf);
  }
  MS_EXCEPTION_IF_NULL(abs);
  auto val = abs->BuildValue();
  MS_EXCEPTION_IF_NULL(val);
  auto class_val = dyn_cast_ptr<parse::ClassType>(val);
  MS_EXCEPTION_IF_NULL(class_val);
  const auto &class_name = class_val->name();
  std::vector<AnfNodePtr> new_cnode_inputs;
  auto fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);

  std::map<std::string, ValueNodePtr> list_or_tuple_func_map = {
    {"class 'list'", NewValueNode(std::make_shared<prim::ListFunc>("list_func"))},
    {"class 'tuple'", NewValueNode(std::make_shared<prim::TupleFunc>("tuple_func"))}};
  auto iter = list_or_tuple_func_map.find(class_name);
  if (iter != list_or_tuple_func_map.end()) {
    (void)new_cnode_inputs.emplace_back(iter->second);
  } else {
    auto class_obj = class_val->obj();
    py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
    auto py_fn =
      python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_CONVERT_CLASS_TO_FUNCTION, py::str(class_name), class_obj);
    if (py::isinstance<py::none>(py_fn)) {
      return ConvertToPyInterpretCall(cnode, conf);
    }
    auto func_fg = parse::ParsePythonCode(py_fn);
    MS_EXCEPTION_IF_NULL(func_fg);
    func_fg->set_manager(fg->manager());
    (void)new_cnode_inputs.emplace_back(NewValueNode(func_fg));
  }

  for (std::size_t i = 1; i < cnode->size(); ++i) {
    (void)new_cnode_inputs.emplace_back(cnode->input(i));
  }
  auto new_cnode = fg->NewCNodeInOrder(new_cnode_inputs);
  new_cnode->set_debug_info(cnode->debug_info());
  AnalysisEnginePtr eng = conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_cnode, conf->context(), conf->func_graph());
  return eng->ForwardConfig(conf, fn_conf);
}

EvalResultPtr AnalysisEngine::EvalCNode(const CNodePtr &cnode, const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(conf);
  MS_EXCEPTION_IF_NULL(cnode);

  // Handle the interpreted node call here.
  const auto &interpreted_eval_result = InterpretedNodeCall(cnode, conf);
  if (interpreted_eval_result != nullptr) {
    return interpreted_eval_result;
  }

  AbstractBasePtr possible_func = GetCNodeOperatorAbstract(cnode, conf->context(), conf->func_graph());
  MS_EXCEPTION_IF_NULL(possible_func->BuildType());
  if (possible_func->IsSameTypeId(AbstractUndetermined::kTypeId)) {
    MS_LOG(DEBUG) << "EvalCNode eval Undetermined";
    return std::make_shared<EvalResult>(possible_func->Clone(), std::make_shared<AttrValueMap>());
  }

  if (possible_func->isa<AbstractClass>()) {
    return ConvertMsClassObjToFunc(cnode, possible_func, conf);
  }
  if (possible_func->isa<AbstractScalar>()) {
    // Convert class to function, such as list(xxx).
    auto val = possible_func->BuildValue();
    MS_EXCEPTION_IF_NULL(val);
    if (val->isa<parse::ClassType>()) {
      return ConvertClassTypeToFunc(cnode, possible_func, conf);
    }
    if (val->isa<parse::InterpretedObject>()) {
      return ConvertCallPyObjCallFunc(cnode, possible_func, conf);
    }
  }

  if (possible_func->isa<AbstractAny>()) {
    return ConvertToPyInterpretCall(cnode, conf);
  }

  if (possible_func->isa<PrimitiveAbstractClosure>()) {
    auto value = GetValueWithoutDoSignature(possible_func->cast<PrimitiveAbstractClosurePtr>()->prim());
    auto new_cnode = InsertUnpackGraph(cnode, value, conf, shared_from_this());
    if (new_cnode != nullptr) {
      AnalysisEnginePtr eng = conf->engine();
      MS_EXCEPTION_IF_NULL(eng);
      AnfNodeConfigPtr new_conf = eng->MakeConfig(new_cnode, conf->context(), conf->func_graph());
      return eng->ForwardConfig(conf, new_conf);
    }
  }

  auto func = dyn_cast_ptr<AbstractFunction>(possible_func);
  if (func == nullptr) {
    MS_LOG(ERROR) << "Can not cast to a AbstractFunction from " << possible_func->ToString() << ".";
    MS_LOG(ERROR) << "It's called at: " << cnode->DebugString();
    MS_EXCEPTION(ValueError) << "The object is not callable. Please check code.";
  }

  // Make arguments config list.
  bool contains_side_effect = false;
  const auto inputs_size = cnode->size();
  ConfigPtrList args_conf_list;
  args_conf_list.reserve(inputs_size - 1);
  // Ignore the first node which is function name.
  for (std::size_t i = 1; i < inputs_size; ++i) {
    const AnfNodePtr &node = cnode->input(i);
    (void)args_conf_list.emplace_back(MakeConfig(node, conf->context(), conf->func_graph()));
    if (check_side_effect()) {
      auto input_cnode = dyn_cast_ptr<CNode>(node);
      if (input_cnode != nullptr) {
        contains_side_effect = contains_side_effect || input_cnode->has_side_effect_node();
      }
    }
  }

  // Find evaluators.
  std::vector<EvaluatorPtr> evaluators;
  func->Visit([this, &evaluators, &cnode](const AbstractFuncAtomPtr &possible_func) {
    const auto &real_func_atom = GetRealFuncAtom(possible_func);
    auto evaluator = this->GetEvaluatorFor(real_func_atom);
    evaluator->set_bound_node(cnode);
    (void)evaluators.emplace_back(std::move(evaluator));
  });

  // Run evaluators.
  auto eval_result = ExecuteEvaluators(evaluators, conf, args_conf_list);
  // Check if func graph contains isolated side-effect, and sync.
  if (check_side_effect()) {
    func->Visit([&contains_side_effect](const AbstractFuncAtomPtr &possible_func) {
      const auto &real_func_atom = GetRealFuncAtom(possible_func);
      bool func_has_side_effect = CheckFuncSideEffect(real_func_atom);
      if (func_has_side_effect) {
        contains_side_effect = true;
      }
    });
    if (contains_side_effect) {
      MS_EXCEPTION_IF_NULL(conf->func_graph());
      MS_LOG(DEBUG) << "Found side-effect, cnode: " << cnode->DebugString()
                    << ", func_graph: " << conf->func_graph()->ToString();
      cnode->set_has_side_effect_node(true);
      conf->func_graph()->set_has_side_effect_node(true);
      eval_result->set_has_side_effect_node(true);
    }
  }
  return eval_result;
}

EvalResultPtr AnalysisEngine::Execute(const AbstractFunctionPtr &func, const AbstractBasePtrList &args_abs_list) {
  MS_EXCEPTION_IF_NULL(func);
  ConfigPtrList args_conf_list;
  (void)std::transform(args_abs_list.begin(), args_abs_list.end(), std::back_inserter(args_conf_list),
                       [](const AbstractBasePtr &arg) -> ConfigPtr { return std::make_shared<VirtualConfig>(arg); });
  std::vector<EvaluatorPtr> infs;
  MS_EXCEPTION_IF_NULL(func);
  auto build_evaluator = [this, &infs](const AbstractFuncAtomPtr &poss) {
    auto evaluator = this->GetEvaluatorFor(poss);
    infs.push_back(evaluator);
  };
  func->Visit(build_evaluator);
  return ExecuteEvaluators(infs, nullptr, args_conf_list);
}

void AnalysisEngine::ClearEvaluatorCache() {
  py::gil_scoped_acquire gil;
  for (auto &element : evaluators_) {
    EvaluatorPtr evaluator = element.second;
    if (evaluator == nullptr || evaluator->evaluator_cache_mgr() == nullptr) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(evaluator->evaluator_cache_mgr());
    evaluator->evaluator_cache_mgr()->Clear();
  }
  for (auto &element : prim_constructors_) {
    EvaluatorPtr evaluator = element.second;
    if (evaluator == nullptr || evaluator->evaluator_cache_mgr() == nullptr) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(evaluator->evaluator_cache_mgr());
    evaluator->evaluator_cache_mgr()->Clear();
  }
  for (auto &element : prim_py_evaluators_) {
    EvaluatorPtr evaluator = element.second;
    if (evaluator == nullptr || evaluator->evaluator_cache_mgr() == nullptr) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(evaluator->evaluator_cache_mgr());
    evaluator->evaluator_cache_mgr()->Clear();
  }
  // Release exception to avoid hup at exit.
  StaticAnalysisException::Instance().ClearException();
  // Reset the EnvironGet sparse option.
  EnvSetSparseResultMgr::GetInstance().Set(false);
}

void AnalysisEngine::Clear() {
  AnalysisResultCacheMgr::GetInstance().Clear();
  anfnode_config_map_.clear();
  eval_trace_.clear();
  evaluators_.clear();
  prim_py_evaluators_.clear();
  constructors_app_.clear();
  continued_evals_.clear();
  root_context_ = nullptr;
}

EvaluatorPtr GetPyEvaluator(const PrimitivePtr &prim, const AnalysisEnginePtr &engine) {
  auto prim_py = dyn_cast<PrimitivePy>(prim);
  if (prim_py != nullptr) {
    auto is_constexpr = prim_py->HasAttr(GRAPH_FLAG_CONSTEXPR_PRIM);
    if (is_constexpr) {
      return std::make_shared<ConstexprEvaluator>(prim_py);
    }
    if (engine == nullptr) {
      return std::make_shared<PythonPrimEvaluator>(prim_py);
    }

    const auto &iter = engine->prim_py_evaluators_.find(prim_py);
    if (iter != engine->prim_py_evaluators_.end()) {
      return iter->second;
    }
    auto evaluator = std::make_shared<PythonPrimEvaluator>(prim_py);
    engine->prim_py_evaluators_[prim_py] = evaluator;
    return evaluator;
  }
  MS_LOG(ERROR) << "The primitive with python evaluator should be a python primitive.";
  return nullptr;
}

inline StandardPrimEvaluatorPtr GetStandardPrimEvaluator(const PrimitivePtr &prim) {
  auto eval_impl_opt = GetFrontendPrimitiveInferImpl(prim);
  if (eval_impl_opt.has_value()) {
    // Find prim infer function in the prim function map return a standard evaluator
    auto eval_impl = eval_impl_opt.value();
    if (eval_impl.IsImplInferShapeAndType() && !IsPrimitiveEquals(prim, prim::kPrimMakeTuple) &&
        !IsPrimitiveEquals(prim, prim::kPrimMakeList)) {
      return std::make_shared<StandardPrimEvaluator>(prim, eval_impl);
    }
  }

  return nullptr;
}

EvaluatorPtr GetPrimEvaluator(const PrimitivePtr &prim, const AnalysisEnginePtr &engine) {
  // Custom Primitive with python infer_shape, infer_type
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->isa<prim::DoSignaturePrimitive>()) {
    return std::make_shared<DoSignatureEvaluator>(prim);
  }
  if (prim->isa<prim::UnpackGraphPrimitive>()) {
    return std::make_shared<UnpackGraphEvaluator>(prim);
  }
  if (IsPrimitiveEquals(prim, prim::kPrimMixedPrecisionCast)) {
    return std::make_shared<MixedPrecisionCastEvaluator>(prim);
  }
  if (IsPrimitiveEquals(prim, prim::kPrimPyExecute)) {
    return std::make_shared<PyExecuteEvaluator>();
  }
  static const bool enable_pre_lift = (common::GetCompileConfig("PRE_LIFT") == "1");
  if (enable_pre_lift && IsPrimitiveEquals(prim, prim::kPrimSwitch)) {
    return std::make_shared<SwitchEvaluator>();
  }

  if (prim->isa<prim::DoTransPrimitiveFunction>()) {
    return std::make_shared<DoTransPrimitiveFunctionEvaluator>(prim);
  }
  // Primitive is defined in OpTable.
  if (mindspore::ops::IsPrimitiveFunction(prim->name())) {
    if (prim->isa<PrimitivePy>()) {
      return std::make_shared<PrimitiveArgsToInputsEvaluator>(prim);
    }
    return std::make_shared<PrimitiveFunctionEvaluator>(prim);
  }

  auto standard_evaluator = GetStandardPrimEvaluator(prim);
  if (standard_evaluator != nullptr) {
    return standard_evaluator;
  }

  // Use python infer function if the infer function not founded in the map return a python evaluator
  EvaluatorPtr evaluator = nullptr;
  if (prim->HasPyEvaluator()) {
    return GetPyEvaluator(prim, engine);
  }

  // Delete this when the infer value can be mapped to the CPU backend operator.
  if (PrimNeedFrontendInferValue(prim)) {
    return nullptr;
  }

  // Return a default evaluator
  if (engine == nullptr) {
    // If engine is nullptr, get constructor from default.
    const PrimEvaluatorMap &prim_evaluator_map = GetPrimEvaluatorConstructors();
    auto iter = prim_evaluator_map.find(prim);
    if (iter != prim_evaluator_map.end()) {
      evaluator = iter->second;
    }
  } else {
    // If engine is given, get constructor from engine resource.
    const PrimEvaluatorMap &prim_evaluator_map = engine->PrimConstructors();
    auto iter = prim_evaluator_map.find(prim);
    if (iter != prim_evaluator_map.end()) {
      evaluator = iter->second;
    }
  }

  if (evaluator == nullptr) {
    MS_LOG(DEBUG) << "The evaluator of the primitive is not defined (" << prim->name() << ").";
  }
  return evaluator;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<PrimitiveAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  const auto &primitive = func->prim();
  if (func->tracking_id() == 0) {
    // Create primitive evaluator if tracking_id == 0.
    auto [iter, is_new] = evaluators_.emplace(func, nullptr);

    if (is_new) {
      iter->second = GetPrimEvaluator(primitive, shared_from_this());
      if (iter->second == nullptr) {
        MS_LOG(EXCEPTION) << "Operator '" << primitive->name()
                          << "' is invalid, or no matching evaluator could be found.";
      }
    }
    return iter->second;
  }
  // Use TrackedEvaluator if tracking_id != 0.
  auto iter = evaluators_.find(func);
  if (iter != evaluators_.end()) {
    return iter->second;
  }
  auto prim_without_tracking_id = std::make_shared<PrimitiveAbstractClosure>(primitive, 0);
  EvaluatorPtr prim_evaluator = _GetEvaluatorFor(prim_without_tracking_id);
  static const bool enable_pre_lift = (common::GetCompileConfig("PRE_LIFT") == "1");
  if (enable_pre_lift && IsPrimitiveEquals(primitive, prim::kPrimSwitch)) {
    auto result = evaluators_.emplace(func, prim_evaluator);
    return result.first->second;
  } else {
    auto tracked_evaluator = std::make_shared<TrackedEvaluator>(prim_evaluator);
    auto result = evaluators_.emplace(func, std::move(tracked_evaluator));
    return result.first->second;
  }
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<FuncGraphAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  auto [iter, is_new] = evaluators_.emplace(func, nullptr);
  if (is_new) {
    iter->second = std::make_shared<FuncGraphEvaluator>(func->func_graph(), func->context());
  }
  return iter->second;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<MetaFuncGraphAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  auto [iter, is_new] = evaluators_.emplace(func, nullptr);
  if (is_new) {
    iter->second = std::make_shared<MetaFuncGraphEvaluator>(func->meta_func_graph(), func->GetScope());
  }
  return iter->second;
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<JTransformedAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  const auto &primal_func = func->fn();
  auto primal_evaluator = GetEvaluatorFor(primal_func);
  return std::make_shared<JEvaluator>(primal_evaluator, primal_func);
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<VmapTransformedAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  const auto &primal_func = func->fn();
  const auto &in_axes = func->in_axes();
  const auto &out_axes = func->out_axes();
  size_t cell_size = func->cell_size();
  auto primal_evaluator = GetEvaluatorFor(primal_func);
  return std::make_shared<VmapEvaluator>(primal_evaluator, primal_func, in_axes, out_axes, cell_size);
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<TaylorTransformedAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  const auto &primal_func = func->fn();
  auto primal_evaluator = GetEvaluatorFor(primal_func);
  return std::make_shared<TaylorEvaluator>(primal_evaluator, primal_func);
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<ShardTransformedAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  const auto &primal_func = func->fn();
  auto primal_evaluator = GetEvaluatorFor(primal_func);
  return std::make_shared<ShardEvaluator>(primal_evaluator, primal_func);
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<VirtualAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  return std::make_shared<VirtualEvaluator>(func->args_abs_list(), func->output());
}

EvaluatorPtr AnalysisEngine::_GetEvaluatorFor(const std::shared_ptr<PartialAbstractClosure> &func) {
  MS_EXCEPTION_IF_NULL(func);
  auto primal_func = func->fn();
  auto part_pair = std::make_pair(primal_func, func->args());
  auto iter = constructors_app_.find(part_pair);
  if (iter != constructors_app_.end()) {
    return iter->second;
  }
  EvaluatorPtr partial_evaluator = nullptr;
  if (func->need_append_to_end()) {
    partial_evaluator = std::make_shared<PartialToEndEvaluator>(primal_func);
  } else {
    auto primal_evaluator = GetEvaluatorFor(primal_func);
    partial_evaluator = std::make_shared<PartialAppEvaluator>(primal_evaluator, func->args());
  }
  auto result = constructors_app_.emplace(std::move(part_pair), std::move(partial_evaluator));
  return result.first->second;
}

EvaluatorPtr AnalysisEngine::GetEvaluatorFor(const AbstractFunctionPtr &func) {
  MS_EXCEPTION_IF_NULL(func);
  MS_LOG(DEBUG) << "GetEvaluatorFor: " << func->ToString() << " tracking_id: " << func->tracking_id();

  if (func->isa<PrimitiveAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<PrimitiveAbstractClosure>(func));
  }
  if (func->isa<FuncGraphAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<FuncGraphAbstractClosure>(func));
  }
  if (func->isa<MetaFuncGraphAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<MetaFuncGraphAbstractClosure>(func));
  }
  if (func->isa<JTransformedAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<JTransformedAbstractClosure>(func));
  }
  if (func->isa<VmapTransformedAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<VmapTransformedAbstractClosure>(func));
  }
  if (func->isa<TaylorTransformedAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<TaylorTransformedAbstractClosure>(func));
  }
  if (func->isa<ShardTransformedAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<ShardTransformedAbstractClosure>(func));
  }
  if (func->isa<VirtualAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<VirtualAbstractClosure>(func));
  }
  if (func->isa<PartialAbstractClosure>()) {
    return _GetEvaluatorFor(std::static_pointer_cast<PartialAbstractClosure>(func));
  }

  MS_LOG(INTERNAL_EXCEPTION) << "Cannot GetEvaluator from " << func->type_name();
}

EvalResultPtr AnalysisEngine::ForwardConfig(const AnfNodeConfigPtr &orig_conf, const AnfNodeConfigPtr new_conf) {
  MS_EXCEPTION_IF_NULL(orig_conf);
  MS_EXCEPTION_IF_NULL(new_conf);
  // If always_eval_flag is true in BaseFuncGraphEvaluaotr, then the CNode with same orig_conf may be forwarded
  // again, so update the config_map with new_conf;
  anfnode_config_map_[orig_conf] = new_conf;
  MS_LOG(DEBUG) << "Forward orig_conf: " << orig_conf->ToString() << ", to new_conf: " << new_conf->ToString();
  MS_EXCEPTION_IF_NULL(orig_conf->node());
  MS_EXCEPTION_IF_NULL(new_conf->node());
  auto old_cnode = orig_conf->node()->cast_ptr<CNode>();
  auto new_cnode = new_conf->node()->cast<CNodePtr>();
  if (old_cnode != nullptr && new_cnode != nullptr) {
    if (old_cnode->func_graph() == new_cnode->func_graph()) {
      MS_LOG(DEBUG) << "Try to remove forward node from order list, forward node: " << new_cnode->DebugString()
                    << ", as origin node should be in order list, origin_node: " << old_cnode->DebugString();
      old_cnode->func_graph()->EraseUnusedNodeInOrder(new_cnode);
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "Forward orig_node to different func_graph, old_node: " << old_cnode->DebugString()
                                 << ", new_node: " << new_cnode->DebugString();
    }
  }
  (void)forward_count_++;
  auto res = ObtainEvalResultWithCache(new_conf);
  (void)forward_count_--;
  return res;
}

EvalResultPtr AnalysisEngine::ExecuteEvaluators(const std::vector<EvaluatorPtr> &evaluators,
                                                const AnfNodeConfigPtr &out_conf, const ConfigPtrList &args_conf_list) {
  if (evaluators.size() == 1) {
    auto &eval = evaluators[0];
    MS_EXCEPTION_IF_NULL(eval);
    return eval->Run(shared_from_this(), args_conf_list, out_conf);
  }
  static const bool enable_single_thread = (common::GetCompileConfig("SINGLE_EVAL") == "1");
  if (enable_single_thread) {
    return ExecuteMultipleEvaluators(evaluators, out_conf, args_conf_list);
  }
  return ExecuteMultipleEvaluatorsMultiThread(evaluators, out_conf, args_conf_list);
}

void AnalysisEngine::SetUndeterminedFlag(const std::string &thread_id, const FuncGraph &fg) {
  static std::mutex fg_lock;
  std::lock_guard<std::mutex> infer_lock(fg_lock);
  MS_LOG(DEBUG) << "Record undetermined flag of fg:" << fg.ToString() << ", thread id:" << thread_id;
  func_graph_undetermined_flags_[&fg].push_front(thread_id);
}

void AnalysisEngine::SetIgnoreValueFlag(const std::string &thread_id, FuncGraph *fg) {
  MS_EXCEPTION_IF_NULL(fg);
  auto it = func_graph_undetermined_flags_.find(fg);
  if (it == func_graph_undetermined_flags_.cend()) {
    return;
  }
  for (const auto &id : it->second) {
    if (thread_id.find(id) != std::string::npos && thread_id != id) {
      MS_LOG(DEBUG) << "Set ignore value of fg:" << fg->ToString() << ", thread id:" << thread_id;
      fg->set_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE, true);
      return;
    }
  }
}

EvaluatorPtr AnalysisEngine::HandleNestedRecursion(const std::vector<EvaluatorPtr> &evaluators,
                                                   const EvaluatorPtr &eval, const AbstractBasePtrList &args_abs_list,
                                                   const EvalTraceRevIter &it, bool *continue_flag) {
  MS_EXCEPTION_IF_NULL(continue_flag);
  MS_EXCEPTION_IF_NULL(eval);
  *continue_flag = false;
  // Find latest entry function to handle nested recursion.
  EvaluatorPtr latest_entry = eval;
  auto latest_entry_iter = eval_trace_.crbegin();
  for (auto r_it = eval_trace_.crbegin(); *r_it != *it;) {
    auto it_temp = std::find(evaluators.cbegin(), evaluators.cend(), r_it->evaluator_);
    if (it_temp != evaluators.cend()) {
      latest_entry = *it_temp;
      latest_entry_iter = r_it;
      break;
    }
    latest_entry_iter = ++r_it;
  }
  if (latest_entry != eval) {
    MS_LOG(DEBUG) << "Continue Evaluator " << eval->ToString();
    *continue_flag = true;
    return latest_entry;
  }

  bool has_undetermined = false;
  // Check whether sub loop has untraced undetermined evaluator.
  std::unordered_set<EvaluatorArgs, EvaluatorArgsHasher, EvaluatorArgsEqual> undetermined_evals;
  for (auto r_it = eval_trace_.crbegin(); r_it != latest_entry_iter; r_it++) {
    (void)undetermined_evals.insert(*r_it);
  }
  MS_LOG(DEBUG) << "undetermined_evals size(): " << undetermined_evals.size();

  for (const auto &u_eval : undetermined_evals) {
    MS_EXCEPTION_IF_NULL(u_eval.evaluator_);
    MS_LOG(DEBUG) << u_eval.evaluator_->ToString() << "check undetermined.";
    auto &alternate_evaluator = multi_poss_[u_eval.evaluator_];
    MS_EXCEPTION_IF_NULL(alternate_evaluator);
    auto eval_cache = alternate_evaluator->evaluator_cache_mgr();
    MS_EXCEPTION_IF_NULL(eval_cache);
    const auto &alt_eval_args = EvaluatorArgs(alternate_evaluator, args_abs_list);
    auto is_not_undetermined_eval = (undetermined_evals.find(alt_eval_args) == undetermined_evals.cend());
    auto is_not_continued_eval = (continued_evals_.find(u_eval) == continued_evals_.cend());
    auto args_not_evaluated = (eval_cache->GetValue(args_abs_list) == nullptr);
    if (is_not_undetermined_eval && (args_not_evaluated || is_not_continued_eval)) {
      MS_LOG(DEBUG) << u_eval.evaluator_->ToString() << "has undetermined.";
      has_undetermined = true;
      break;
    }
  }
  if (!has_undetermined) {
    MS_LOG(DEBUG) << eval->ToString() << "has no undetermined.";
    *continue_flag = true;
    return latest_entry;
  }

  return latest_entry;
}

FuncGraphPtr GetFuncGraphFromBranchNode(const AnfNodePtr &branch_node) {
  MS_EXCEPTION_IF_NULL(branch_node);
  auto fg = GetValueNode<FuncGraphPtr>(branch_node);
  if (fg != nullptr) {
    return fg;
  }
  if (IsPrimitiveCNode(branch_node, prim::kPrimPartial)) {
    fg = GetValueNode<FuncGraphPtr>(branch_node->cast<CNodePtr>()->input(kPartialGraphIndex));
  }
  if (fg != nullptr) {
    return fg;
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Unexpected branch node: " << branch_node->DebugString();
}

std::string JoinBranchesFailedInfo(const AbstractBasePtr &abs, const AbstractBasePtr &last_out_abs,
                                   const AnfNodePtr &node, const std::string &error_info) {
  constexpr int recursive_level = 2;
  std::ostringstream buffer;
  buffer << "Cannot join the return values of different branches, perhaps you need to make them equal.\n"
         << error_info
         << "#dmsg#Framework Error Message:#dmsg#The abstract type of the return value of the current branch is:\n"
         << abs->ToString() << ",\n and that of the previous branch is:\n"
         << last_out_abs->ToString() << ".\n"
         << "The node is " << node->DebugString(recursive_level);
  if (!node->isa<CNode>()) {
    buffer << "\n";
    return buffer.str();
  }
  auto input_node = node->cast_ptr<CNode>()->input(0);
  if (IsPrimitiveCNode(input_node, prim::kPrimSwitch)) {
    // {prim::kPrimSwitch, cond, true_branch, false_branch}
    const auto &cnode = input_node->cast_ptr<CNode>();
    auto true_out = GetFuncGraphFromBranchNode(cnode->input(kSwitchTrueBranchIndex))->get_return();
    auto false_out = GetFuncGraphFromBranchNode(cnode->input(kSwitchFalseBranchIndex))->get_return();
    buffer << ", true branch: " << cnode->input(kSwitchTrueBranchIndex)->ToString() << "\n"
           << trace::GetDebugInfoStr(true_out->debug_info())
           << "\n, false branch: " << cnode->input(kSwitchFalseBranchIndex)->ToString() << "\n"
           << trace::GetDebugInfoStr(false_out->debug_info());
  } else if (IsPrimitiveCNode(input_node, prim::kPrimSwitchLayer)) {
    // {prim::kPrimSwitchLayer, X, {prim::kPrimMakeTuple, branch1, branch2, ...}}
    constexpr int branch_index = 2;
    const auto &tuple_node = input_node->cast_ptr<CNode>()->input(branch_index);
    if (IsPrimitiveCNode(tuple_node, prim::kPrimMakeTuple)) {
      const auto &cnode = tuple_node->cast_ptr<CNode>();
      for (size_t i = 1; i < cnode->size(); i++) {
        auto out_node = GetValueNode<FuncGraphPtr>(cnode->input(i))->get_return();
        MS_EXCEPTION_IF_NULL(out_node);
        buffer << ", branch" << i << ": " << cnode->input(i)->ToString() << "\n"
               << trace::GetDebugInfoStr(out_node->debug_info());
      }
    }
  } else {
    buffer << trace::GetDebugInfoStr(node->debug_info());
  }
  buffer << "\n";
  return buffer.str();
}

void SetUseFlagsForJoinedAny(const AbstractBasePtrList &out_abs_list) {
  for (const auto &abs : out_abs_list) {
    SetSequenceElementsUseFlagsRecursively(abs, true);
  }
}

EvalResultPtr AnalysisEngine::ProcessEvalResults(const AbstractBasePtrList &out_abs_list, const AnfNodePtr &node) {
  if (out_abs_list.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "There is an endless loop for evaluator.";
  }

  if (out_abs_list.size() == 1) {
    MS_EXCEPTION_IF_NULL(out_abs_list[0]);
    // If only one result derived, then broaden it to avoid wrong constant propagation.
    return std::make_shared<EvalResult>(out_abs_list[0]->Broaden(), std::make_shared<AttrValueMap>());
  }
  MS_EXCEPTION_IF_NULL(node);

  // Return Any if some branch returns Any.
  if (std::any_of(out_abs_list.cbegin(), out_abs_list.cend(), [](const AbstractBasePtr &abs) {
        MS_EXCEPTION_IF_NULL(abs);
        return abs->isa<AbstractAny>() && !abs->isa<AbstractNegligible>();
      })) {
    MS_LOG(INFO) << "The branches outputs contain Any output.\nJoin them to Any output.";
    return std::make_shared<EvalResult>(std::make_shared<AbstractAny>(), std::make_shared<AttrValueMap>());
  }

  AbstractBasePtr last_out_abs = out_abs_list[0];
  MS_EXCEPTION_IF_NULL(last_out_abs);
  AbstractBasePtr joined_abs = out_abs_list[0];
  for (size_t i = 1; i < out_abs_list.size(); ++i) {
    const auto &abs = out_abs_list[i];
    MS_EXCEPTION_IF_NULL(abs);
    try {
      MS_LOG(DEBUG) << "Join node: " << node->DebugString() << ", " << joined_abs->ToString() << ", and "
                    << abs->ToString();
      MS_LOG_TRY_CATCH_SCOPE;
      joined_abs = joined_abs->Join(abs);
    } catch (const py::type_error &ex) {
      auto error_info = ExtractLoggingInfo(ex.what());
      const auto info = JoinBranchesFailedInfo(abs, last_out_abs, node, error_info);
      MS_LOG(INFO) << info;
      auto joined_any = std::make_shared<AbstractJoinedAny>();
      joined_any->set_exception(AbstractJoinedAny::ExceptionType::kTypeError);
      joined_any->set_message(info);
      SetUseFlagsForJoinedAny(out_abs_list);
      return std::make_shared<EvalResult>(joined_any, std::make_shared<AttrValueMap>());
    } catch (const py::value_error &ex) {
      auto error_info = ExtractLoggingInfo(ex.what());
      const auto info = JoinBranchesFailedInfo(abs, last_out_abs, node, error_info);
      MS_LOG(INFO) << info;
      auto joined_any = std::make_shared<AbstractJoinedAny>();
      joined_any->set_exception(AbstractJoinedAny::ExceptionType::kValueError);
      joined_any->set_message(info);
      SetUseFlagsForJoinedAny(out_abs_list);
      return std::make_shared<EvalResult>(joined_any, std::make_shared<AttrValueMap>());
    } catch (const std::exception &ex) {
      auto error_info = ExtractLoggingInfo(ex.what());
      const auto info = JoinBranchesFailedInfo(abs, last_out_abs, node, error_info);
      MS_LOG(INFO) << info;
      auto joined_any = std::make_shared<AbstractJoinedAny>();
      joined_any->set_exception(AbstractJoinedAny::ExceptionType::kDefault);
      joined_any->set_message(info);
      // Remove it when the transform form dict to tuple is disabled in Compatible or Lax mode.
      if (joined_abs->isa<AbstractDictionary>()) {
        joined_any->set_user_data<bool>("from_dict", std::make_shared<bool>(true));
      }
      SetUseFlagsForJoinedAny(out_abs_list);
      return std::make_shared<EvalResult>(joined_any, std::make_shared<AttrValueMap>());
    }
    MS_EXCEPTION_IF_NULL(joined_abs);
    last_out_abs = abs;
  }

  MS_LOG(DEBUG) << "Multiple evaluators joined: " << joined_abs->ToString();
  return std::make_shared<EvalResult>(joined_abs, std::make_shared<AttrValueMap>());
}

EvalResultPtr AnalysisEngine::ExecuteMultipleEvaluatorsMultiThread(const std::vector<EvaluatorPtr> &evaluators,
                                                                   const AnfNodeConfigPtr &out_conf,
                                                                   const ConfigPtrList &args_conf_list) {
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  MS_EXCEPTION_IF_NULL(out_conf->func_graph());
  // Release GIL for C++
  MS_LOG(DEBUG) << out_conf->func_graph()->ToString() << "_" << std::this_thread::get_id() << " begin.";
  py::gil_scoped_release infer_gil_release;

  // Only one thread to run
  AnalysisSchedule::GetInstance().WaitForRun();

  // Wait for the last switch node to finish.
  MS_LOG(DEBUG) << GetInferThread() << "async : entry switch  " << out_conf->ToString();
  auto eval_result = AnalysisResultCacheMgr::GetInstance().GetSwitchValue(out_conf);
  if (eval_result == nullptr) {
    MS_LOG(DEBUG) << GetInferThread() << "async : Init switch  " << out_conf->node()->ToString();
    AnalysisResultCacheMgr::GetInstance().InitSwitchValue(out_conf);
  } else {
    return std::make_shared<EvalResult>(eval_result, nullptr);
  }
  auto possible_parent_fg = out_conf->node()->func_graph();
  MS_EXCEPTION_IF_NULL(possible_parent_fg);
  // Eval result of the main.
  AsyncAbstractPtr async_result_main = std::make_shared<AsyncAbstract>();
  if (possible_parent_fg->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE)) {
    async_result_main->set_ignore_value(true);
  }
  // Eval result of the branches
  std::vector<AsyncAbstractPtr> async_result_branches;
  SetUndeterminedFlag(AnalysisSchedule::thread_id(), *possible_parent_fg);
  for (auto &evaluator : evaluators) {
    static std::atomic<int> id_count{0};
    std::string thread_id = AnalysisSchedule::thread_id() + "." + std::to_string(id_count.fetch_add(1));
    MS_EXCEPTION_IF_NULL(evaluator);
    AsyncAbstractPtr async_result_branch = std::make_shared<AsyncAbstract>(async_result_main);
    // Control the order to run.
    AsyncAbstractPtr control_run_order = std::make_shared<AsyncAbstract>();
    control_run_order->set_result(std::make_shared<AbstractScalar>(1));
    AsyncInferTaskPtr async_task = AsyncInferTask::MakeShared(control_run_order, thread_id);
    AnalysisSchedule::GetInstance().IncreaseThreadCount();
    MS_LOG(DEBUG) << GetInferThread() << "async : " << evaluator->ToString();
    auto thread = std::thread(ExecEvaluator, evaluator, shared_from_this(), args_conf_list, out_conf, thread_id,
                              async_result_branch, async_result_main, async_task, trace::GetCurrentGraphEvalStack(),
                              trace::GetCNodeDebugStack());
    thread.detach();

    // Push to list of running loop
    MS_LOG(DEBUG) << "Add to schedule: " << async_task.get();
    AnalysisSchedule::GetInstance().Add2Schedule(async_task);  // Activate order witch child thread.
    (void)async_result_branches.emplace_back(std::move(async_result_branch));
  }

  size_t len = evaluators.size();
  size_t min_size = 2;
  if (len < min_size) {
    MS_LOG(EXCEPTION) << "There are at least 2 evaluators in multi thread, but got " << len << " evaluator.";
  }

  MS_LOG(DEBUG) << GetInferThread() << "async : wait for one of async to finish.  " << evaluators[0]->ToString()
                << " or  " << evaluators[1]->ToString() << "...";

  auto first_result = async_result_main->GetResult();
  MS_EXCEPTION_IF_NULL(first_result);
  MS_LOG(DEBUG) << GetInferThread() << "async main thread result of " << out_conf->node()->ToString() << " = "
                << first_result->ToString();

  AbstractBasePtrList out_abs_list;
  if (NeedWaitForBranches(first_result)) {
    MS_LOG(DEBUG) << GetInferThread() << " BuildPossibleSpecs.";
    BuildPossibleSpecs(first_result, async_result_branches, &out_abs_list);
  } else {
    for (size_t i = 0; i < len; ++i) {
      AbstractBasePtr result;
      MS_EXCEPTION_IF_NULL(async_result_branches[i]);
      if (enable_waiting_branch_eval()) {
        // wait to get the result of branch.
        result = async_result_branches[i]->GetResult();
      } else {
        // Not wait to get the result of branch.
        result = async_result_branches[i]->TryGetResult();
      }

      if (result) {
        MS_EXCEPTION_IF_NULL(evaluators[i]);
        MS_EXCEPTION_IF_NULL(result);
        MS_LOG(DEBUG) << "#" << i << ": " << GetInferThread() << " async get " << evaluators[i]->ToString()
                      << ", result: " << result->ToString() << ", args: " << args_conf_list;
        out_abs_list.push_back(result);
      }
    }
  }
  MS_LOG(DEBUG) << GetInferThread() << " finish.";
  const auto &processed_result = ProcessEvalResults(out_abs_list, out_conf->node());
  if (processed_result != nullptr) {
    // This is the final switch()() value.
    AnalysisResultCacheMgr::GetInstance().SetSwitchValue(out_conf, processed_result->abstract());
  }
  MS_LOG(DEBUG) << GetInferThread() << " join finish.";
  return processed_result;
}

EvalResultPtr AnalysisEngine::ExecuteMultipleEvaluators(const std::vector<EvaluatorPtr> &evaluators,
                                                        const AnfNodeConfigPtr &out_conf,
                                                        const ConfigPtrList &args_conf_list) {
  AbstractBasePtrList out_abs_list;
  const size_t evaluators_size = 2;
  if (evaluators.size() < evaluators_size) {
    MS_LOG(INTERNAL_EXCEPTION) << "Evaluators size is less than 2.";
  }
  multi_poss_[evaluators[0]] = evaluators[1];
  multi_poss_[evaluators[1]] = evaluators[0];
  AbstractBasePtrList args_abs_list = EvaluateArguments(args_conf_list);
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  auto possible_parent_fg = out_conf->node()->func_graph();
  MS_EXCEPTION_IF_NULL(possible_parent_fg);
  possible_parent_fg->set_flag(kFuncGraphFlagUndetermined, true);
  MS_LOG(DEBUG) << "Set graph undetermined flag for " << possible_parent_fg->ToString();
  for (const auto &eval : evaluators) {
    MS_EXCEPTION_IF_NULL(eval);
    const auto current_inf = EvaluatorArgs(eval, args_abs_list);
    MS_LOG(DEBUG) << "Check evaluator " << eval->ToString();
    // If current evaluator is under tracing, then skip current evaluator to avoid recursively evaluating.
    auto it = std::find(eval_trace_.crbegin(), eval_trace_.crend(), current_inf);
    if (it == eval_trace_.crend()) {
      eval_trace_.push_back(current_inf);
      auto eval_result = eval->Run(shared_from_this(), args_conf_list, out_conf);
      MS_EXCEPTION_IF_NULL(eval_result);
      auto eval_abstract = eval_result->abstract();
      MS_EXCEPTION_IF_NULL(eval_abstract);

      out_abs_list.push_back(eval_abstract);
      eval_trace_.pop_back();
      if (eval_trace_.empty()) {
        multi_poss_.clear();
      }
    } else {
      bool continue_flag = false;
      auto latest_entry = HandleNestedRecursion(evaluators, eval, args_abs_list, it, &continue_flag);
      if (continue_flag) {
        MS_EXCEPTION_IF_NULL(current_inf.evaluator_);
        MS_LOG(DEBUG) << "The continued_evals_ insert " << current_inf.evaluator_.get() << "/"
                      << current_inf.evaluator_->ToString();
        continued_evals_.insert(current_inf);
        continue;
      }

      // Try to travel the latest undetermined.
      if (latest_entry != eval_trace_.rbegin()->evaluator_) {
        MS_LOG(DEBUG) << "Direct run evaluator " << eval.get() << "/" << eval->ToString();
        auto eval_result = latest_entry->Run(shared_from_this(), args_conf_list, out_conf);
        MS_EXCEPTION_IF_NULL(eval_result);
        MS_EXCEPTION_IF_NULL(eval_result->abstract());
        MS_LOG(DEBUG) << "End direct evaluator " << latest_entry->ToString()
                      << ", return out_abs: " << eval_result->abstract()->ToString();
        possible_parent_fg->set_flag(kFuncGraphFlagUndetermined, false);
        return eval_result;
      }
    }
  }
  possible_parent_fg->set_flag(kFuncGraphFlagUndetermined, false);
  return ProcessEvalResults(out_abs_list, out_conf->node());
}

EvalResultPtr AnfNodeConfig::ObtainEvalResult() {
  AnfNodeConfigPtr self = shared_from_base<AnfNodeConfig>();
  return engine_.lock()->ObtainEvalResultWithCache(self);
}

AbstractBasePtr MakeAbstractClosure(const FuncGraphPtr &func_graph, const AnalysisContextPtr &context,
                                    const AnfNodePtr &anf_node) {
  AnalysisContextPtr temp_context = context;
  if (temp_context == nullptr) {
    temp_context = AnalysisContext::DummyContext();
  }
  return std::make_shared<FuncGraphAbstractClosure>(func_graph, temp_context, anf_node);
}

AbstractBasePtr MakeAbstractClosure(const MetaFuncGraphPtr &meta_func_graph, const AnfNodePtr &anf_node) {
  MetaFuncGraphAbstractClosurePtr meta_func_graph_fn;
  if (anf_node == nullptr) {
    meta_func_graph_fn = std::make_shared<MetaFuncGraphAbstractClosure>(meta_func_graph);
  } else {
    meta_func_graph_fn = std::make_shared<MetaFuncGraphAbstractClosure>(meta_func_graph, anf_node, anf_node->scope());
  }
  return meta_func_graph_fn;
}

AbstractBasePtr MakeAbstractClosure(const PrimitivePtr &primitive, const AnfNodePtr &anf_node) {
  auto prim_func = std::make_shared<PrimitiveAbstractClosure>(primitive, anf_node);
  return prim_func;
}

AbstractBasePtr ToAbstract(const ValuePtr &value, const AnalysisContextPtr &context, const AnfNodeConfigPtr &conf) {
  MS_EXCEPTION_IF_NULL(value);
  AnfNodePtr anf_node = nullptr;
  if (conf != nullptr) {
    anf_node = conf->node();
  }
  if (value->isa<Primitive>()) {
    auto prim = value->cast<PrimitivePtr>();
    return MakeAbstractClosure(prim, anf_node);
  }
  if (value->isa<FuncGraph>()) {
    auto func_graph = value->cast<FuncGraphPtr>();
    return MakeAbstractClosure(func_graph, context, anf_node);
  }
  if (value->isa<MetaFuncGraph>()) {
    auto meta_func_graph = value->cast<MetaFuncGraphPtr>();
    return MakeAbstractClosure(meta_func_graph, anf_node);
  }
  if (value->isa<ValueSequence>() && anf_node != nullptr) {
    auto abs = value->ToAbstract();
    MS_EXCEPTION_IF_NULL(abs);
    // Attach corresponding python sequence object to AbstractSequence.
    py::object py_list_obj =
      fallback::HasPyObjectInNode(anf_node) ? fallback::GetPyObjectFromNode(anf_node) : ValueToPyData(value);
    fallback::AttachPyObjToAbs(abs, py_list_obj, !fallback::HasPyObjectInNode(anf_node));
    MS_LOG(DEBUG) << "Attach python list object " << fallback::GetPyObjectPtrStr(py_list_obj)
                  << " to new abstract: " << abs->ToString();
    // Set sequence node for new AbstractSequence.
    static const auto enable_eliminate_unused_element = (common::GetCompileConfig("ENABLE_DDE") != "0");
    if (enable_eliminate_unused_element) {
      auto sequence_abs = abs->cast<AbstractSequencePtr>();
      MS_EXCEPTION_IF_NULL(sequence_abs);
      SetSequenceNodeElementsUseFlags(anf_node, std::make_shared<std::vector<bool>>(sequence_abs->elements().size()));
      std::shared_ptr<AnfNodeWeakPtrList> sequence_nodes = std::make_shared<AnfNodeWeakPtrList>();
      (void)sequence_nodes->emplace_back(AnfNodeWeakPtr(anf_node));
      sequence_abs->set_sequence_nodes(sequence_nodes);
    }
    return abs;
  }
  if (value->isa<ValueDictionary>() && anf_node != nullptr) {
    auto abs = value->ToAbstract();
    MS_EXCEPTION_IF_NULL(abs);
    // Attach corresponding python dictionary object to AbstractDictionary.
    py::object py_dict_obj =
      fallback::HasPyObjectInNode(anf_node) ? fallback::GetPyObjectFromNode(anf_node) : fallback::GeneratePyObj(abs);
    fallback::AttachPyObjToAbs(abs, py_dict_obj, !fallback::HasPyObjectInNode(anf_node));
    MS_LOG(DEBUG) << "Attach python dict object " << fallback::GetPyObjectPtrStr(py_dict_obj)
                  << " to new abstract: " << abs->ToString();
    return abs;
  }
  return value->ToAbstract();
}

AbstractBasePtr FromValueInside(const ValuePtr &value, bool broaden) {
  AbstractBasePtr a = ToAbstract(value, nullptr, nullptr);
  if (broaden) {
    a = a->Broaden();
  }
  return a;
}

EvalResultPtr EvalOnePrim(const PrimitivePtr &primitive, const AbstractBasePtrList &arg_specs) {
  auto evaluator = GetPrimEvaluator(primitive, nullptr);
  if (evaluator == nullptr) {
    MS_LOG(ERROR) << "The evaluator of the primitive is not defined (" << primitive->name() << ").";
    return nullptr;
  }
  auto trivial_evaluator = dyn_cast_ptr<TrivialPrimEvaluator>(evaluator);
  if (trivial_evaluator != nullptr) {
    return trivial_evaluator->EvalPrim(nullptr, arg_specs);
  }
  // Support MakeTuple/MakeList ops in PyNative mode.
  auto transition_evaluator = dyn_cast_ptr<TransitionPrimEvaluator>(evaluator);
  if (transition_evaluator != nullptr) {
    if (transition_evaluator->isa<MakeTupleEvaluator>() || transition_evaluator->isa<MakeListEvaluator>()) {
      return transition_evaluator->EvalPrim(nullptr, arg_specs, nullptr, nullptr);
    }
    return pipeline::AbstractAnalyze(primitive, arg_specs).eval_result;
  }
  // To add EvalPrim call of TransitionPrimEvaluator such as GetAttr.
  MS_LOG(ERROR) << "The primitive '" << primitive->ToString() << "' should be built as a TrivialPrimEvaluator, but "
                << evaluator->ToString();
  return nullptr;
}

AbstractBasePtr EvalFunctionValue(const ValuePtr &func, const AbstractBasePtrList &args_spec) {
  auto func_abs = func->ToAbstract();
  if (!func_abs->isa<AbstractFunction>()) {
    MS_LOG(EXCEPTION) << "The value : " << func->ToString() << " is not a callable object.";
  }
  if (func->isa<Primitive>() && !func->isa<prim::DoSignaturePrimitive>()) {
    return EvalOnePrim(func->cast<PrimitivePtr>(), args_spec)->abstract();
  } else {
    auto infer_graph = std::make_shared<FuncGraph>();
    std::vector<AnfNodePtr> inputs = {std::make_shared<ValueNode>(func)};
    (void)std::transform(args_spec.begin(), args_spec.end(), std::back_inserter(inputs),
                         [infer_graph](const AbstractBasePtr &) -> AnfNodePtr { return infer_graph->add_parameter(); });
    auto infer_node = infer_graph->NewCNode(inputs);
    infer_graph->set_return(infer_node);
    auto manager = Manage(infer_graph, true);
    auto engine = std::make_shared<abstract::AnalysisEngine>(abstract::GetPrimEvaluatorConstructors(), manager);
    auto res = engine->Run(infer_graph, args_spec);
    return res.eval_result->abstract();
  }
}

AnalysisContextPtr NewContext(const AnalysisContextPtr &current_context, const FuncGraphPtr &fg,
                              const AbstractBasePtrList &args_abs_list) {
  MS_EXCEPTION_IF_NULL(fg);
  auto new_context = current_context->NewContext(fg, args_abs_list);
  if (new_context == nullptr) {  // Not obtain context for fg->parent() during create context.
    FuncGraphPtr parent_graph = fg->parent();
    const auto no_parent = parent_graph == nullptr;
#ifdef ENABLE_DUMP_IR
    DumpIR(std::string("EXCEPTION_NEW_CONTEXT_CURRENT_") + (no_parent ? "0" : "1") + "_" + fg->ToString() + ".ir", fg);
    if (!no_parent) {
      DumpIR("EXCEPTION_NEW_CONTEXT_PARENT_" + parent_graph->ToString() + ".ir", parent_graph);
    }
#endif
    // If parent context is not found, we'll raise exception.
    MS_LOG(INTERNAL_EXCEPTION) << "BUG: Failed to find parent context in current context: "
                               << current_context->ToString() << ", func_graph: " << fg->ToString()
                               << ", parent_graph: " << (no_parent ? "null" : parent_graph->ToString()) << ",\n"
                               << trace::GetDebugInfoStr(fg->debug_info());
  }
  return new_context;
}
}  // namespace abstract
}  // namespace mindspore
