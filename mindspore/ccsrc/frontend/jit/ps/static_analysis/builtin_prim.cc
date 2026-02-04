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

#include "frontend/jit/ps/static_analysis/builtin_prim.h"

#include "include/utils/convert_utils_py.h"
#include "include/utils/pipeline/fallback.h"
#include "primitive/math_ops.h"
#include "primitive/sequence_ops.h"
#include "frontend/jit/ps/fallback.h"
#include "frontend/jit/ps/parse/data_converter.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

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
      if (const_abstract_value->BuildValue()->ContainsValueAny()) {
        return false;
      }
    }
    return true;
  }
  if (args_abs_list[0]->isa<AbstractScalar>()) {
    auto const_abstract_value = args_abs_list[0]->cast_ptr<AbstractScalar>();
    MS_EXCEPTION_IF_NULL(const_abstract_value);
    return !const_abstract_value->BuildValue()->ContainsValueAny();
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
    if (!const_value->ContainsValueAny()) {
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
      if (const_abstract_value->BuildValue()->ContainsValueAny()) {
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
      return !const_abstract_value->BuildValue()->ContainsValueAny();
    }
  }
  if (args_abs_list.size() == 1) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(args_abs_list[1]);
  if (args_abs_list[1]->isa<AbstractScalar>()) {
    auto const_abstract_value = args_abs_list[1]->cast_ptr<AbstractScalar>();
    MS_EXCEPTION_IF_NULL(const_abstract_value);
    return !const_abstract_value->BuildValue()->ContainsValueAny();
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
  constexpr auto index_input = 0;
  constexpr auto index_decimals = 1;
  if (fallback::ContainsSequenceAnyType(args_abs_list[index_input]) ||
      (args_abs_list.size() == max_input_index && fallback::ContainsSequenceAnyType(args_abs_list[index_decimals]))) {
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
    auto const_value = args_abs_list[index_input]->BuildValue();
    auto type = args_abs_list[index_input]->BuildType();
    py::tuple tuple_args(max_input_index);
    tuple_args[index_input] = ValueToPyData(const_value);
    if (args_abs_list.size() == max_input_index) {
      auto point_num_value = args_abs_list[index_decimals]->BuildValue();
      tuple_args[index_decimals] = ValueToPyData(point_num_value);
    } else {
      tuple_args[index_decimals] = py::none();
    }
    py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
    py::object round_data = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_CONST_ROUND, tuple_args);
    ValuePtr abs_value = parse::data_converter::PyDataToValue(round_data);
    auto res = std::make_shared<AbstractScalar>(abs_value, type);
    auto infer_result = std::make_shared<EvalResult>(res, std::make_shared<AttrValueMap>());
    evaluator_cache_mgr_->SetValue(args_abs_list, infer_result);
    return infer_result;
  }
  // Convert round ops.
  AnfNodePtrList round_inputs = {NewValueNode(prim::kPrimRound), cnode->input(index_input + 1)};
  if (args_abs_list.size() == max_input_index) {
    (void)round_inputs.emplace_back(cnode->input(index_decimals + 1));
  } else {
    (void)round_inputs.emplace_back(NewValueNode(MakeValue<int64_t>(0)));
  }
  auto fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto new_cnode = fg->NewCNodeInOrder(round_inputs);
  AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
  return engine->ForwardConfig(out_conf, fn_conf);
}
}  // namespace abstract
}  // namespace mindspore
