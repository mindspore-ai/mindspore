/**
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
#include "frontend/expander/pack/packfunc.h"

#include <unordered_map>
#include <vector>
#include <memory>
#include <string>

#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/structure_ops.h"
#include "include/common/debug/anf_ir_dump.h"
#include "frontend/expander/pack/pack_expander.h"
#include "utils/ms_context.h"
#include "ir/func_graph_cloner.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/core/ops/packfunc.h"
#include "pipeline/jit/parse/parse.h"

namespace mindspore {
namespace expander {
bool IsPackGraph(const FuncGraphPtr &fg) { return fg->ToString().find("pack_wrap") != std::string::npos; }

PrimitivePyPtr GetPackFuncPrimitive(const FuncGraphPtr &fg) {
  auto prim = GetValueNode<PrimitivePtr>(fg->get_return()->input(1)->cast_ptr<CNode>()->input(0));
  auto do_signature = dyn_cast<prim::DoSignaturePrimitive>(prim);
  return do_signature->function()->cast<PrimitivePyPtr>();
}

FuncGraphPtr UpdateReusingGraphForPack(const FuncGraphPtr &reusing_graph, const std::vector<AnfNodePtr> &parameters) {
  auto pack_node = reusing_graph->get_return()->input(1)->cast_ptr<CNode>();
  for (size_t i = 0; i < parameters.size(); i++) {
    auto param = reusing_graph->add_parameter();
    pack_node->add_input(param);
  }
  reusing_graph->set_has_vararg(false);
  reusing_graph->set_attr("reuse", MakeValue(True));
  auto prim_py = GetPackFuncPrimitive(reusing_graph);
  prim_py->AddAttr("reuse", MakeValue(True));
  return reusing_graph;
}

void GenerateTopGraphParams(const py::object &obj, std::vector<AnfNodePtr> *parameters) {
  MS_LOG(DEBUG) << "enter GenerateTopGraphParams";
  auto trainable_parameters = py::getattr(obj, "parameters_and_names", py::none())();
  auto top_func_graph = parse::Parser::GetTopFuncGraph();
  for (auto &tr : trainable_parameters) {
    auto item = py::cast<py::tuple>(tr);
    auto value = item[1];
    auto par_name = item[0].cast<std::string>();
    auto parameter_name = py::getattr(value, "name", py::str(par_name)).cast<std::string>();
    auto exist_fv = top_func_graph->GetParameterByName(parameter_name);
    if (exist_fv) {
      parameters->push_back(exist_fv);
    } else {
      auto fv = top_func_graph->AddFvParameter(parameter_name, parse::GetParameterValue(value));
      parameters->push_back(fv);
    }
  }
}

void GetPackGraphParams(const FuncGraphPtr &fg, std::vector<AnfNodePtr> *parameters) {
  return GenerateTopGraphParams(GetPackFuncPrimitive(fg)->GetPyObj().attr("cell_obj"), parameters);
}

namespace {
bool IsAbstractDynamicShape(const std::vector<AbstractBasePtr> &input_args) {
  return std::any_of(input_args.begin(), input_args.end(),
                     [](const AbstractBasePtr &abs) { return abs->BuildShape()->IsDynamic(); });
}

bool IsAbstractOutputTensor(const AbstractBasePtr &abs) {
  if (abs->isa<abstract::AbstractTuple>()) {
    auto abs_tuple = abs->cast<abstract::AbstractTuplePtr>()->elements();
    return std::all_of(abs_tuple.begin(), abs_tuple.end(),
                       [](const AbstractBasePtr &abs) { return IsAbstractOutputTensor(abs); });
  }
  return abs->isa<abstract::AbstractTensor>();
}

void ReorderParamsForReuseGraph(const FuncGraphPtr &graph, const PrimitivePyPtr &prim_py) {
  std::vector<AnfNodePtr> parameters;
  py::object cell_obj = prim_py->GetPyObj().attr("cell_obj");
  GenerateTopGraphParams(cell_obj, &parameters);
  auto old_params = graph->parameters();
  std::vector<AnfNodePtr> new_params{};
  for (auto &i : old_params) {
    auto found_in_fv_list = find_if(parameters.begin(), parameters.end(), [&i](const AnfNodePtr &fv_param) {
      return !i->ToString().empty() && i->ToString() == fv_param->ToString();
    });
    if (found_in_fv_list == parameters.end()) {
      new_params.push_back(i);
    }
  }
  for (auto &i : parameters) {
    auto found_in_fv_list = find_if(old_params.begin(), old_params.end(), [&i](const AnfNodePtr &fv_param) {
      return !i->ToString().empty() && i->ToString() == fv_param->ToString();
    });
    if (found_in_fv_list != old_params.end()) {
      new_params.push_back(*found_in_fv_list);
    }
  }
  graph->set_parameters(new_params);
  return;
}

FuncGraphPtr PostProcessForReuseGraph(const FuncGraphPtr &graph, const PrimitivePyPtr &prim_py) {
  ReorderParamsForReuseGraph(graph, prim_py);
  FuncGraphVector func_graphs = {graph};
  Cloner cloner(func_graphs, false, false, true, std::make_shared<TraceCopy>(), std::make_shared<TraceGraphReusing>());
  cloner.Run();
  auto cloned_fg_iter = cloner.cloned_func_graphs().find(graph);
  if (cloned_fg_iter == cloner.cloned_func_graphs().end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Clone func graph failed! " << graph->ToString();
  }
  auto fg = cloned_fg_iter->second;
  fg->set_flag(FUNC_GRAPH_FLAG_CELL_REUSE, true);
  fg->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, true);
  fg->set_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE, true);
  return fg;
}
}  // namespace
using PackGraphMap = std::unordered_map<abstract::AbstractBasePtrList, FuncGraphPtr,
                                        abstract::AbstractBasePtrListHasher, abstract::AbstractBasePtrListEqual>;

static std::unordered_map<std::string, PackGraphMap> pack_graph_cache;
void ClearAllCache() { pack_graph_cache.clear(); }

FuncGraphPtr ExpandPackFunc(const PrimitivePtr &prim, const abstract::AbstractBasePtrList &abs_list) {
  auto key = GetValue<std::string>(prim->GetAttr("unique_key"));
  PackExpander::is_pynative_mode = GetValue<bool>(prim->GetAttr("is_pynative_mode"));
  auto &graph_map = pack_graph_cache[key];
  auto it = graph_map.find(abs_list);
  if (it != graph_map.end()) {
    return it->second;
  }
  auto prim_py = prim->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(prim_py);
  auto expander = expander::PackExpander::Instance();
  bool reuse = prim->HasAttr("reuse");
  expander->SetReuse(reuse);
  abstract::AbstractBasePtrList new_abs_list;
  for (auto &i : abs_list) {
    if (reuse && i->cast_ptr<abstract::AbstractRefTensor>()) {
      continue;
    }
    new_abs_list.push_back(i);
  }
  FuncGraphPtr graph;
  {
    py::gil_scoped_acquire acquire;
    py::object expand_func = prim_py->GetPyObj().attr("__expand__");
    py::object inputs = expander->BeginGraph(new_abs_list);
    py::object output = expand_func(inputs);
    graph = expander->EndGraph(output);
    if (reuse) {
      graph = PostProcessForReuseGraph(graph, prim_py);
    }
    graph_map[abs_list] = graph;
  }
  static const bool dump_result = (common::GetEnv("MS_DEV_DUMP_PACK") == "on");
  if (dump_result) {
    DumpIR("pack_func_" + key + ".ir", graph, true);
  }
  expander->SetReuse(false);
  return graph;
}

class PackFuncInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto abs = InferShapeAndType(nullptr, primitive, input_args);
    return abs->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto abs = InferShapeAndType(nullptr, primitive, input_args);
    return abs->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    if (IsAbstractDynamicShape(input_args)) {
      MS_LOG(WARNING) << "Dynamic shape operator is not fully supported in trace graph capturing. Please check the "
                         "dump-ir to confirm its correctness.";
    }
    auto graph = ExpandPackFunc(primitive, input_args);
    MS_EXCEPTION_IF_NULL(graph);
    // the python primitive object may be used in different places with different inputs, so we
    // cannot save the graph in graph mode. But for pynative mode, this primitive is inferred
    // in forward thread sequentially and deep copied to backend runtime, so we can save graph
    // in attr to save performance.
    auto abs = graph->output()->abstract();
    if (PackExpander::is_pynative_mode) {
      primitive->set_attr("recent_graph", graph);
      if (!IsAbstractOutputTensor(abs)) {
        MS_EXCEPTION(ValueError)
          << "The output of trace captured graph should be one or more flattened Tensor, bug get "
          << abs->BuildType()->ToString() << ".";
      }
    }
    return abs;
  }
};
}  // namespace expander
namespace ops {
REGISTER_PRIMITIVE_OP_INFER_IMPL(PackFunc, prim::kPrimPackFunc, expander::PackFuncInfer, false);
}  // namespace ops
}  // namespace mindspore
