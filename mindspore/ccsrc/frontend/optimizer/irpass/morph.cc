/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/morph.h"

#include <algorithm>
#include <map>
#include <string>

#include "frontend/jit/ps/static_analysis/auto_monad.h"
#include "include/frontend/operator/primitive_py.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"
#include "pybind11/pybind11.h"
#include "ir/anf.h"
#include "ir/value.h"
#include "ir/func_graph_flag.h"

namespace py = pybind11;

namespace mindspore {
namespace opt {
namespace irpass {

namespace {

constexpr auto kFlagMonadParameterSize = "monad_parameter_size";

AnfNodePtr CopyDefaultParamFromFg(const FuncGraphPtr &fg, const ParameterPtr &param, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(param);

  const auto default_values = fg->parameter_default_value();
  const auto iter = default_values.find(param->name());
  if (iter == default_values.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Funcgraph: " << fg->ToString() << " has no default parameter: " << param->ToString();
  } else if (iter->second->isa<Null>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Parameter `" << param->ToString() << "` of func graph `" << fg->ToString()
                               << "` has no default parameter.";
  } else {
    const auto &default_value = iter->second;
    if (default_value->isa<ValueNode>()) {
      return default_value;
    } else if (IsPrimitiveCNode(default_value, prim::kPrimResolve)) {
      const auto &resolve_node = default_value->cast<CNodePtr>();
      const auto &cnode_fg = cnode->func_graph();
      return cnode_fg->NewCNode(resolve_node->inputs());
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "Unexpected default parameter: " << param->ToString() << " found.";
    }
  }
  return nullptr;
}

size_t GetMonadParameterSize(const FuncGraphPtr &fg) {
  int64_t monad_param_size = 0;
  if (fg->has_attr(kFlagMonadParameterSize)) {
    monad_param_size = GetValue<int64_t>(fg->get_attr(kFlagMonadParameterSize));
  }
  return LongToSize(monad_param_size);
}

CNodePtr CreateNewCNode(const FuncGraphPtr &fg, const CNodePtr &cnode, size_t start_of_monad) {
  AnfNodePtrList inputs = {NewValueNode(fg)};

  size_t start_of_kwarg = 1;
  for (; start_of_kwarg < start_of_monad; ++start_of_kwarg) {
    const auto &input = cnode->input(start_of_kwarg);
    if (IsValueNode<KeywordArg>(input)) {
      break;
    }
  }

  size_t end_of_kwarg = start_of_kwarg;
  for (; end_of_kwarg < start_of_monad; ++end_of_kwarg) {
    const auto &input = cnode->input(end_of_kwarg);
    if (!IsValueNode<KeywordArg>(input)) {
      break;
    }
  }

  // arg without default value
  for (size_t idx = 1; idx < start_of_kwarg; ++idx) {
    inputs.push_back(cnode->input(idx));
  }

  // arg with default value
  std::map<std::string, ValuePtr> kwargs_map;
  for (size_t idx = start_of_kwarg; idx < end_of_kwarg; ++idx) {
    const auto &input = cnode->input(idx);
    const auto value_node = GetValueNode<KeywordArgPtr>(input);
    MS_EXCEPTION_IF_NULL(value_node);
    kwargs_map[value_node->get_key()] = value_node->get_value();
  }

  size_t param_size_without_monad = fg->parameters().size() - GetMonadParameterSize(fg);

  for (size_t idx = start_of_kwarg - 1; idx < param_size_without_monad; ++idx) {
    const auto param = fg->parameters()[idx]->cast<ParameterPtr>();
    auto iter = kwargs_map.find(param->name());
    if (iter != kwargs_map.end()) {
      inputs.push_back(NewValueNode(iter->second));
    } else {
      const auto default_param = CopyDefaultParamFromFg(fg, param, cnode);
      MS_EXCEPTION_IF_NULL(default_param);
      inputs.push_back(default_param);
    }
  }

  // add monad
  for (size_t idx = start_of_monad; idx < cnode->size(); ++idx) {
    const auto &input = cnode->input(idx);
    if (!IsMonad(input)) {
      MS_LOG(INTERNAL_EXCEPTION) << input->DebugString() << " is not a monad node.";
    }
    inputs.push_back(input);
  }

  const auto &func_graph = cnode->func_graph();
  return func_graph->NewCNode(inputs);
}

void AddMonadParameterForFuncGraph(const FuncGraphPtr &fg, const CNodePtr cnode, size_t target_fg_param_size,
                                   size_t start_of_monad) {
  size_t cnode_monad_size = cnode->size() - start_of_monad;
  size_t monad_param_size = GetMonadParameterSize(fg);
  for (size_t idx = 0; idx < cnode_monad_size; ++idx) {
    if (fg->parameters().size() < target_fg_param_size - 1) {
      const auto abs = cnode->input(start_of_monad + idx)->abstract();
      MS_EXCEPTION_IF_NULL(abs);
      std::string name = abs->isa<abstract::AbstractIOMonad>() ? "io" : "u";
      mindspore::pipeline::AddMonadParameter(fg, name, abs);
      ++monad_param_size;
    }
  }

  fg->set_attr(kFlagMonadParameterSize, MakeValue(SizeToLong(monad_param_size)));
}

FuncGraphPtr CreateMorphGraph(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_py = prim->cast_ptr<PrimitivePy>();
  if (prim_py == nullptr) {
    return nullptr;
  }

  auto py_obj = prim_py->GetPyObj();
  auto fn = py::getattr(py_obj, "__morph_fn__", py::none());
  if (fn.is_none()) {
    MS_LOG(EXCEPTION) << "Primitive " << prim->name() << " does not have __morph_fn__ attribute.";
  }

  // Since we will add monad parameters to `morph_fn` later, we set the `forbid_reuse` parameter of the
  // `ConvertToFuncGraph` function to True here.
  ValuePtr morph_fn = parse::ConvertToFuncGraph(fn, {}, parse::PYTHON_MOD_GET_PARSE_METHOD, true);
  if (morph_fn == nullptr || !morph_fn->isa<FuncGraph>()) {
    MS_LOG(EXCEPTION) << "Failed to convert morph (" << py::str(fn) << ") fn to funcgraph.";
  }

  auto morph_fg = morph_fn->cast<FuncGraphPtr>();

  auto bprop_fn = py::getattr(py_obj, "__morph_bprop_fn__", py::none());
  if (!bprop_fn.is_none()) {
    ValuePtr morph_bprop_fn = parse::ConvertToFuncGraph(bprop_fn);
    if (morph_bprop_fn == nullptr || !morph_bprop_fn->isa<FuncGraph>()) {
      MS_LOG(EXCEPTION) << "Failed to convert morph bprop fn(" << py::str(bprop_fn) << ") to funcgraph.";
    }
    auto morph_bprop_fg = morph_bprop_fn->cast<FuncGraphPtr>();
    (void)morph_fg->transforms().emplace(parse::CUSTOM_BPROP_NAME, FuncGraphTransform(morph_bprop_fg));
    (void)morph_bprop_fg->transforms().emplace("primal", FuncGraphTransform(morph_fg));
    morph_fg->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
    morph_fg->set_flag(FUNC_GRAPH_FLAG_PRIMAL_OF_BPROP, true);
  }

  return morph_fg;
}

void SetFlagForMorphFunc(const CNodePtr &cnode, const FuncGraphPtr &fg, const std::string &flag) {
  int64_t num = -1;
  if (cnode->HasAttr(flag)) {
    auto value = cnode->GetAttr(flag);
    num = GetValue<int64_t>(value);
  }
  if (num != -1) {
    MS_LOG(DEBUG) << "The fg: " << fg->ToString() << " mark " << flag << ": " << num;
    fg->set_attr(flag, MakeValue(static_cast<size_t>(num)));
    fg->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, true);
  }
}

void MarkStreamInfoForMorphFunc(const CNodePtr &cnode, const FuncGraphPtr &fg) {
  SetFlagForMorphFunc(cnode, fg, kFuncGraphFlagStreamId);
  SetFlagForMorphFunc(cnode, fg, kFuncGraphFlagStreamLimitId);
  SetFlagForMorphFunc(cnode, fg, kFuncGraphFlagCubeNum);
  SetFlagForMorphFunc(cnode, fg, kFuncGraphFlagVectorNum);
  SetFlagForMorphFunc(cnode, fg, kFuncGraphFlagStreamCtxAfter);
  SetFlagForMorphFunc(cnode, fg, kFuncGraphFlagStreamLimitCtxAfter);
}
}  // namespace

AnfNodePtr Morph::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  const auto &prim = GetCNodePrimitive(node);
  if (prim == nullptr) {
    MS_LOG(DEBUG) << "Node is not a cnode with primitive. node: " << node->DebugString() << ".";
    return nullptr;
  }

  auto metamorphosis_value = prim->GetAttr("__metamorphosis__");
  if (metamorphosis_value == nullptr) {
    return nullptr;
  }

  auto metamorphosis = GetValue<bool>(metamorphosis_value);
  if (!metamorphosis) {
    return nullptr;
  }

  auto fg = CreateMorphGraph(prim);
  if (fg == nullptr) {
    return nullptr;
  }

  bool has_recompute_scope =
    (node->scope() != nullptr && node->scope()->name().compare(0, strlen(kAttrRecompute), kAttrRecompute) == 0);
  if (has_recompute_scope) {
    parse::UpdateRecomputeScope(fg);
  }

  fg->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);

  if (fg->has_vararg() || fg->has_kwarg() || fg->kwonlyargs_count() != 0 || fg->fv_param_count() != 0) {
    MS_LOG(EXCEPTION) << "Morph function does not support varargs/kwargs/kwonlyargs and free parameters now. Please "
                         "check you funcgraph: "
                      << trace::GetDebugInfoStr(fg->debug_info());
  }

  const auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  size_t start_of_monad = 1;
  for (; start_of_monad < cnode->size(); ++start_of_monad) {
    const auto &input = cnode->input(start_of_monad);
    if (IsMonad(input)) {
      break;
    }
  }

  const size_t input_size_without_monad = start_of_monad - 1;
  const size_t param_size_without_monad = fg->parameters().size() - GetMonadParameterSize(fg);
  if (input_size_without_monad > param_size_without_monad) {
    MS_LOG(EXCEPTION) << "Too many inputs of cnode: " << cnode->DebugString() << ", max input size is "
                      << param_size_without_monad << ", but got " << input_size_without_monad;
  }

  auto new_cnode = CreateNewCNode(fg, cnode, start_of_monad);

  // add monad parameters for funcgraph
  AddMonadParameterForFuncGraph(fg, cnode, new_cnode->size(), start_of_monad);
  // Mark stream info
  MarkStreamInfoForMorphFunc(cnode, fg);

  return new_cnode;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
