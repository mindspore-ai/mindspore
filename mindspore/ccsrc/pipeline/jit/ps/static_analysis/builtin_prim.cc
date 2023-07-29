/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "pipeline/jit/ps/static_analysis/builtin_prim.h"

#include "include/common/utils/convert_utils_py.h"
#include "include/common/fallback.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "pipeline/jit/ps/fallback.h"
#include "pipeline/jit/ps/parse/data_converter.h"

namespace mindspore {
namespace abstract {
bool InnerAbsEvaluator::CheckConst(const AbstractBasePtrList &args_abs_list) const {
  if (args_abs_list[0]->isa<AbstractSequence>()) {
    auto abs_seq = args_abs_list[0]->cast<AbstractSequencePtr>();
    const auto &elements = abs_seq->elements();
    for (auto ele : elements) {
      MS_EXCEPTION_IF_NULL(ele);
      if (!ele->isa<AbstractScalar>()) {
        return false;
      }
      auto const_abstract_value = ele->cast_ptr<AbstractScalar>();
      MS_EXCEPTION_IF_NULL(const_abstract_value);
      if (const_abstract_value->BuildValue() == kValueAny) {
        return false;
      }
    }
    return true;
  }
  if (args_abs_list[0]->isa<AbstractScalar>()) {
    auto const_abstract_value = args_abs_list[0]->cast_ptr<AbstractScalar>();
    MS_EXCEPTION_IF_NULL(const_abstract_value);
    return const_abstract_value->BuildValue() != kValueAny;
  }
  return false;
}

EvalResultPtr InnerAbsEvaluator::EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list,
                                          const ConfigPtr &, const AnfNodeConfigPtr &out_conf) {
  // abs(-1) = 1
  if (args_abs_list.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "abs() requires 1 argument.";
  }
  MS_EXCEPTION_IF_NULL(out_conf->node());
  auto cnode = out_conf->node()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(args_abs_list[0]);
  // Convert pyexecute.
  if (fallback::ContainsSequenceAnyType(args_abs_list[0])) {
    const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
    if (allow_fallback_runtime) {
      auto pyexecute_node = fallback::ConvertCNodeToPyExecuteForPrim(cnode, "abs");
      MS_LOG(DEBUG) << "Convert: " << cnode->DebugString() << " -> " << pyexecute_node->DebugString();
      AnfNodeConfigPtr fn_conf = engine->MakeConfig(pyexecute_node, out_conf->context(), out_conf->func_graph());
      return engine->ForwardConfig(out_conf, fn_conf);
    }
  }
  // Process constants.
  if (CheckConst(args_abs_list)) {
    auto const_value = args_abs_list[0]->BuildValue();
    if (const_value != kValueAny) {
      auto type = args_abs_list[0]->BuildType();
      MS_EXCEPTION_IF_NULL(type);
      auto py_x_data = ValueToPyData(const_value);
      py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
      py::object abs_data = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_CONST_ABS, py_x_data);
      ValuePtr abs_value = parse::data_converter::PyDataToValue(abs_data);
      auto res = std::make_shared<AbstractScalar>(abs_value, type);
      auto infer_result = std::make_shared<EvalResult>(res, std::make_shared<AttrValueMap>());
      evaluator_cache_mgr_->SetValue(args_abs_list, infer_result);
      return infer_result;
    }
  }
  // Convert abs ops.
  auto new_cnode = std::make_shared<CNode>(*cnode);
  new_cnode->set_input(0, NewValueNode(prim::kPrimAbs));
  AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
  return engine->ForwardConfig(out_conf, fn_conf);
}

bool InnerRoundEvaluator::CheckConst(const AbstractBasePtrList &args_abs_list) const {
  MS_EXCEPTION_IF_NULL(args_abs_list[0]);
  if (args_abs_list[0]->isa<AbstractTensor>()) {
    return false;
  }
  if (args_abs_list[0]->isa<AbstractSequence>()) {
    auto abs_seq = args_abs_list[0]->cast<AbstractSequencePtr>();
    const auto &elements = abs_seq->elements();
    for (auto ele : elements) {
      MS_EXCEPTION_IF_NULL(ele);
      if (!ele->isa<AbstractScalar>()) {
        return false;
      }
      auto const_abstract_value = ele->cast_ptr<AbstractScalar>();
      MS_EXCEPTION_IF_NULL(const_abstract_value);
      if (const_abstract_value->BuildValue() == kValueAny) {
        return false;
      }
    }
    if (args_abs_list.size() == 1) {
      return true;
    }
  }

  if (args_abs_list[0]->isa<AbstractScalar>()) {
    auto const_abstract_value = args_abs_list[0]->cast_ptr<AbstractScalar>();
    MS_EXCEPTION_IF_NULL(const_abstract_value);
    if (args_abs_list.size() == 1) {
      return const_abstract_value->BuildValue() != kValueAny;
    }
  }
  if (args_abs_list.size() == 1) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(args_abs_list[1]);
  if (args_abs_list[1]->isa<AbstractScalar>()) {
    auto const_abstract_value = args_abs_list[1]->cast_ptr<AbstractScalar>();
    MS_EXCEPTION_IF_NULL(const_abstract_value);
    return const_abstract_value->BuildValue() != kValueAny;
  }
  return args_abs_list[1]->isa<AbstractNone>();
}

EvalResultPtr InnerRoundEvaluator::EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list,
                                            const ConfigPtr &, const AnfNodeConfigPtr &out_conf) {
  // round(1.909, None) = round(1.909) = 2, round(1.909, 2) = 1.91
  constexpr size_t max_input_index = 2;
  if (args_abs_list.size() == 0 || args_abs_list.size() > max_input_index) {
    MS_LOG(INTERNAL_EXCEPTION) << "round() requires 1 or 2 arguments.";
  }
  MS_EXCEPTION_IF_NULL(out_conf->node());
  auto cnode = out_conf->node()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // Convert pyexecute.
  if (fallback::ContainsSequenceAnyType(args_abs_list[0]) ||
      (args_abs_list.size() == max_input_index && fallback::ContainsSequenceAnyType(args_abs_list[1]))) {
    const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
    if (allow_fallback_runtime) {
      auto pyexecute_node = fallback::ConvertCNodeToPyExecuteForPrim(cnode, "round");
      MS_LOG(DEBUG) << "Convert: " << cnode->DebugString() << " -> " << pyexecute_node->DebugString();
      AnfNodeConfigPtr fn_conf = engine->MakeConfig(pyexecute_node, out_conf->context(), out_conf->func_graph());
      return engine->ForwardConfig(out_conf, fn_conf);
    }
  }
  // Process constants.
  bool is_const = CheckConst(args_abs_list);
  if (is_const) {
    auto const_value = args_abs_list[0]->BuildValue();
    auto type = args_abs_list[0]->BuildType();
    py::tuple tuple_args(max_input_index);
    tuple_args[0] = ValueToPyData(const_value);
    tuple_args[1] = py::none();
    if (args_abs_list.size() > 1) {
      auto point_num_value = args_abs_list[1]->BuildValue();
      auto py_point_data = ValueToPyData(point_num_value);
      tuple_args[1] = py_point_data;
    }
    py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
    py::object round_data = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_CONST_ROUND, tuple_args);
    ValuePtr abs_value = parse::data_converter::PyDataToValue(round_data);
    auto res = std::make_shared<AbstractScalar>(abs_value, type);
    auto infer_result = std::make_shared<EvalResult>(res, std::make_shared<AttrValueMap>());
    evaluator_cache_mgr_->SetValue(args_abs_list, infer_result);
    return infer_result;
  }
  if (args_abs_list.size() == max_input_index) {
    MS_EXCEPTION(TypeError) << "When applying round() to tensor, only one tensor is supported as input.";
  }
  // Convert round ops.
  auto new_cnode = std::make_shared<CNode>(*cnode);
  new_cnode->set_input(0, NewValueNode(prim::kPrimRound));
  AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
  return engine->ForwardConfig(out_conf, fn_conf);
}

EvalResultPtr InnerLenEvaluator::EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list,
                                          const ConfigPtr &, const AnfNodeConfigPtr &out_conf) {
  // len([1, 2]]) = 2
  if (args_abs_list.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "len() requires 1 argument.";
  }
  MS_EXCEPTION_IF_NULL(out_conf->node());
  auto cnode = out_conf->node()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(args_abs_list[0]);
  MS_LOG(DEBUG) << "args_abs_list[0]:" << args_abs_list[0]->ToString();

  // Process constants.
  if (args_abs_list[0]->isa<AbstractScalar>()) {
    auto const_value = args_abs_list[0]->BuildValue();
    MS_EXCEPTION_IF_NULL(const_value);
    auto const_type = args_abs_list[0]->BuildType();
    MS_EXCEPTION_IF_NULL(const_type);
    if (const_value == kValueAny) {
      MS_EXCEPTION(TypeError) << "object of type " << const_type->ToString() << " has no len().";
    }
    auto py_x_data = ValueToPyData(const_value);
    py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
    py::object len_data = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_CONST_LEN, py_x_data);
    ValuePtr len_value = parse::data_converter::PyDataToValue(len_data);
    MS_EXCEPTION_IF_NULL(len_value);
    auto res = std::make_shared<AbstractScalar>(len_value);
    auto infer_result = std::make_shared<EvalResult>(res, std::make_shared<AttrValueMap>());
    evaluator_cache_mgr_->SetValue(args_abs_list, infer_result);
    return infer_result;
  }

  // Process list, tuple, dict and tensor(pyexecute).
  auto new_cnode = std::make_shared<CNode>(*cnode);
  if (args_abs_list[0]->isa<AbstractSequence>()) {
    new_cnode->set_input(0, NewValueNode(prim::kPrimSequenceLen));
  } else if (args_abs_list[0]->isa<AbstractDictionary>()) {
    new_cnode->set_input(0, NewValueNode(prim::kPrimDictLen));
  } else if (args_abs_list[0]->isa<AbstractAny>()) {
    const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
    // Convert pyexecute.
    if (allow_fallback_runtime) {
      auto pyexecute_node = fallback::ConvertCNodeToPyExecuteForPrim(cnode, "len");
      MS_LOG(DEBUG) << "Convert: " << cnode->DebugString() << " -> " << pyexecute_node->DebugString();
      AnfNodeConfigPtr fn_conf = engine->MakeConfig(pyexecute_node, out_conf->context(), out_conf->func_graph());
      return engine->ForwardConfig(out_conf, fn_conf);
    } else {
      MS_EXCEPTION(TypeError) << "The len() only supports types such as Tensor, list, tuple, dict, scalar in JIT "
                              << "kStrict mode, but got " << args_abs_list[0]->ToString() << ".";
    }
  } else if (args_abs_list[0]->isa<AbstractTensor>()) {
    new_cnode->set_input(0, NewValueNode(prim::kPrimArrayLen));
  } else {
    MS_EXCEPTION(TypeError) << "The len() only supports types such as Tensor, list, tuple, dict, scalar, and numpy"
                            << " ndarray, but got " << args_abs_list[0]->ToString() << ".";
  }
  AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
  return engine->ForwardConfig(out_conf, fn_conf);
}
}  // namespace abstract
}  // namespace mindspore
