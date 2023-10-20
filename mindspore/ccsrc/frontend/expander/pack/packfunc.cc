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
#include <utility>
#include <set>
#include <algorithm>
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/structure_ops.h"
#include "include/common/debug/anf_ir_dump.h"
#include "frontend/expander/pack/pack_expander.h"
#include "utils/ms_context.h"
#include "ir/func_graph_cloner.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/core/ops/packfunc.h"
#include "pipeline/jit/ps/parse/parse.h"
#include "pipeline/pynative/predict_out_type_map.h"
#include "include/common/utils/anfalgo.h"
#include "frontend/expander/pack/packfunc_grad.h"
#include "mindspore/core/ops/array_ops.h"

namespace mindspore {
namespace expander {
using PackGraphMap = std::unordered_map<abstract::AbstractBasePtrList, FuncGraphPtr,
                                        abstract::AbstractBasePtrListHasher, abstract::AbstractBasePtrListEqual>;

bool IsPackGraph(const FuncGraphPtr &fg) { return fg->ToString().find("pack_wrap") != std::string::npos; }

PrimitivePyPtr GetPackFuncPrimitive(const FuncGraphPtr &fg) {
  auto return_node = fg->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  auto cnode = return_node->input(1)->cast_ptr<CNode>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  auto do_signature = dyn_cast<prim::DoSignaturePrimitive>(prim);
  MS_EXCEPTION_IF_NULL(do_signature);
  auto res = do_signature->function()->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(res);
  return res;
}

FuncGraphPtr UpdateReusingGraphForPack(const FuncGraphPtr &reusing_graph, const std::vector<AnfNodePtr> &parameters) {
  auto return_node = reusing_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  auto pack_node = return_node->input(1)->cast_ptr<CNode>();
  MS_EXCEPTION_IF_NULL(pack_node);
  for (size_t i = 0; i < parameters.size(); i++) {
    auto param = reusing_graph->add_parameter();
    pack_node->add_input(param);
  }
  reusing_graph->set_has_vararg(false);
  reusing_graph->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, true);
  reusing_graph->set_flag(FUNC_GRAPH_FLAG_CELL_REUSE, true);
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

void GetSubPackGraphParams(const FuncGraphPtr &fg, const FuncGraphPtr &g, std::vector<AnfNodePtr> *parameters,
                           std::set<const AnfNode *> *memo) {
  std::vector<AnfNodePtr> p;
  GetPackGraphParams(g, &p);
  for (auto &item : p) {
    if (item->cast<ParameterPtr>()->has_default()) {
      auto return_node = g->get_return();
      MS_EXCEPTION_IF_NULL(return_node);
      auto node = return_node->input(kIndex1);
      auto pack_node = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(pack_node);
      pack_node->add_input(item);
      if (memo->emplace(item.get()).second) {
        g->add_parameter_obj_node(item);
        auto &node_users = fg->manager()->node_users();
        auto &users_node = node_users[item];
        users_node.add({node, SizeToInt(pack_node->size() - 1)});
        (void)parameters->emplace_back(item);
      }
    }
  }
  auto prim_py = GetPackFuncPrimitive(g);
  prim_py->AddAttr("reuse", MakeValue(True));
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
      auto name = i->cast<ParameterPtr>()->name();
      return !name.empty() && name == fv_param->cast<ParameterPtr>()->name();
    });
    if (found_in_fv_list == parameters.end()) {
      new_params.push_back(i);
    }
  }
  for (auto &i : parameters) {
    auto found_in_fv_list = find_if(old_params.begin(), old_params.end(), [&i](const AnfNodePtr &fv_param) {
      auto name = i->cast<ParameterPtr>()->name();
      return !name.empty() && name == fv_param->cast<ParameterPtr>()->name();
    });
    if (found_in_fv_list != old_params.end()) {
      new_params.push_back(*found_in_fv_list);
    } else {
      new_params.push_back(i);
    }
  }
  graph->set_parameters(new_params);
  return;
}

void GetTrainableParameters(const FuncGraphPtr &fg, std::vector<AnfNodePtr> *parameters) {
  std::set<const AnfNode *> memo;
  for (auto &item : fg->parameter_obj_nodes()) {
    if (item->cast<ParameterPtr>()->has_default() && memo.emplace(item.get()).second) {
      parameters->push_back(item);
    }
  }
  auto used_fgs = fg->func_graphs_used_total();
  for (auto &g : used_fgs) {
    for (auto &item : g->parameter_obj_nodes()) {
      if (item->cast<ParameterPtr>()->has_default() && memo.emplace(item.get()).second) {
        parameters->push_back(item);
      }
    }
  }
}

FuncGraphPtr GenerateReusingGraph(const FuncGraphPtr &fg) {
  std::vector<AnfNodePtr> parameters;
  GetTrainableParameters(fg, &parameters);
  FuncGraphVector func_graphs = {fg};
  Cloner cloner(func_graphs, false, false, true, std::make_shared<TraceCopy>(), std::make_shared<TraceGraphReusing>());
  cloner.Run();
  auto cloned_fg_iter = cloner.cloned_func_graphs().find(fg);
  auto reusing_graph = cloned_fg_iter->second;
  auto &cloned_nodes = cloner.cloned_nodes();
  auto manager = fg->manager();
  for (auto &fv : parameters) {
    TraceGuard guard(std::make_shared<TraceGraphReusing>(fv->debug_info()));
    auto param = reusing_graph->add_parameter();
    param->set_name(fv->cast<ParameterPtr>()->name());
    param->set_abstract(fv->cast<ParameterPtr>()->abstract());
    auto &node_users = manager->node_users()[fv];
    for (auto &n : node_users) {
      auto iter = cloned_nodes.find(n.first);
      if (iter == cloned_nodes.end()) {
        continue;
      }
      auto repl_n = iter->second->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(repl_n);
      repl_n->set_input(IntToSize(n.second), param);
    }
  }
  return reusing_graph;
}

FuncGraphPtr PostProcessForReuseGraph(const FuncGraphPtr &graph, const PrimitivePyPtr &prim_py) {
  auto mng = graph->manager();
  if (mng == nullptr) {
    mng = Manage(graph, false);
    graph->set_manager(mng);
  }
  auto fg = GenerateReusingGraph(graph);
  ReorderParamsForReuseGraph(fg, prim_py);
  return fg;
}

size_t GetSizeByAbstract(const AbstractBasePtr &abs) {
  if (!abs->isa<abstract::AbstractSequence>()) {
    return 1;
  }
  auto tuple_abstract = abs->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abstract);
  return tuple_abstract->elements().size();
}

void SetOutputNum(const PrimitivePyPtr &prim_py, const AnfNodePtr &out_node) {
  if (prim_py->HasAttr("output_num")) {
    return;
  }
  // If the output node is a variable output such as IdentityN op, it cannot be set the 'output_num'.
  auto cnode = out_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> output_nodes;
  if (IsPrimitive(cnode, prim::kPrimMakeTuple)) {
    size_t tuple_input_num = cnode->size() - 1;
    for (size_t j = 0; j < tuple_input_num; ++j) {
      if (auto node = common::AnfAlgo::VisitKernel(cnode, j).first; node->isa<CNode>()) {
        output_nodes.push_back(node);
      }
    }
  } else {
    if (auto node = common::AnfAlgo::VisitKernel(cnode, 0).first; node->isa<CNode>()) {
      output_nodes.push_back(node);
    }
  }
  auto output_num_is_fixed = [](const TypePtr &type) -> bool {
    return type == kTensorType || (type->isa<Tuple>() && type != kTuple);
  };
  for (const auto &node : output_nodes) {
    const auto &node_prim = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(node_prim);
    auto op_name = node_prim->name();
    auto out_type = pynative::PredictOutTypeByName(op_name);
    if (!output_num_is_fixed(out_type)) {
      MS_LOG_DEBUG << "For " << op_name << ", the number of outputs is not fixed.";
      return;
    }
  }
  py::object add_prim_attr_func = prim_py->GetPyObj().attr("add_prim_attr");
  add_prim_attr_func("output_num", GetSizeByAbstract(out_node->abstract()));
}

void SetGradAttachNum(const PrimitivePyPtr &prim_py, size_t grad_attach_num) {
  constexpr auto kGradAttachNum = "grad_attach_num";
  if (auto attr = prim_py->GetAttr(kGradAttachNum); attr != nullptr) {
    auto attr_value = GetValue<int64_t>(attr);
    if (LongToSize(attr_value) != grad_attach_num) {
      MS_LOG_EXCEPTION << "The grad_attach_num should be the same, but changed from " << attr_value << " to "
                       << grad_attach_num;
    }
    return;
  }
  py::object add_prim_attr_func = prim_py->GetPyObj().attr("add_prim_attr");
  add_prim_attr_func(kGradAttachNum, grad_attach_num);
}

bool NeedEliminate(const CNodePtr &cast_cnode) {
  auto inputs = cast_cnode->inputs();
  constexpr size_t kInputNum = 2;
  if (inputs.size() < kInputNum) {
    MS_LOG_EXCEPTION << "input num of cast cnode should be greater than " << kInputNum << " but got " << inputs.size();
  }
  // srt type check
  auto src = inputs[1];
  auto src_type = src->Type();
  if (src_type == nullptr || !src_type->isa<TensorType>()) {
    return false;
  }
  src_type = src_type->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(src_type);
  // dst type check
  auto dst_type = cast_cnode->Type();
  if (dst_type == nullptr || !dst_type->isa<TensorType>()) {
    return false;
  }
  dst_type = dst_type->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(dst_type);
  if (src_type->type_id() == dst_type->type_id()) {
    return true;
  }
  return false;
}

void CastEliminate(const FuncGraphPtr &graph) {
  const auto &node_list = TopoSort(graph->output());
  std::vector<CNodePtr> eliminate_cast_nodes;
  for (const auto &node : node_list) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsPrimitiveCNode(cnode, prim::kPrimCast) && NeedEliminate(cnode)) {
      eliminate_cast_nodes.push_back(cnode);
    }
  }
  if (eliminate_cast_nodes.empty()) {
    return;
  }
  auto mng = graph->manager();
  if (mng == nullptr) {
    mng = Manage(graph, false);
    graph->set_manager(mng);
  }
  for (auto &node : eliminate_cast_nodes) {
    if (!mng->Replace(node, node->input(1))) {
      MS_LOG_EXCEPTION << "CastEliminate fail for" << node->fullname_with_scope();
    }
    graph->DropNode(node);
  }
}

inline void DynamicShapeCheck(const abstract::AbstractBasePtrList &abs_list) {
  if (IsAbstractDynamicShape(abs_list)) {
    MS_LOG(WARNING) << "Dynamic shape operator is not fully supported in trace graph capturing. Please check the "
                       "dump-ir to confirm its correctness.";
  }
}

inline FuncGraphPtr PackFuncConstructGraph(const PrimitivePyPtr &prim_py,
                                           const abstract::AbstractBasePtrList &abs_list) {
  auto expander = expander::PackExpander::Instance();
  py::object cell_obj = prim_py->GetPyObj().attr("cell_obj");
  py::object expand_func = prim_py->GetPyObj().attr("__expand__");
  py::object inputs = expander->BeginGraph(cell_obj, abs_list);
  py::object output = expand_func(inputs);
  auto graph = expander->EndGraph(cell_obj, output);
  return graph;
}

inline void CacheResGraph(PackGraphMap *graph_map, const abstract::AbstractBasePtrList &abs_list,
                          const FuncGraphPtr &graph, bool pynative_grad) {
  if (pynative_grad) {
    // During the grad processing, there will be a `abs->set_value` operation that will change abs,
    // so abs needs to be cloned.
    abstract::AbstractBasePtrList abs_list_clone;
    std::transform(abs_list.begin(), abs_list.end(), std::back_inserter(abs_list_clone),
                   [](const abstract::AbstractBasePtr &abs) { return abs->Clone(); });
    MS_EXCEPTION_IF_NULL(graph);
    (*graph_map)[abs_list_clone] = graph;
  } else {
    (*graph_map)[abs_list] = graph;
  }
}
}  // namespace

// GraphMode needs to clear the PackCache after compilation, but PyNativeMode needs to keep the PackCache.
static std::unordered_map<std::string, PackGraphMap> pynative_pack_cache;
static std::unordered_map<std::string, PackGraphMap> graph_pack_cache;

void ClearCompileAllCache() { graph_pack_cache.clear(); }

void ClearAllPackCache() {
  graph_pack_cache.clear();
  pynative_pack_cache.clear();
  ClearGraphGradInfoCache();
}

FuncGraphPtr ExpandPackFuncGraph(const PrimitivePtr &prim, const abstract::AbstractBasePtrList &abs_list) {
  DynamicShapeCheck(abs_list);
  auto &pack_graph_cache = graph_pack_cache;
  auto key = GetValue<std::string>(prim->GetAttr("unique_key"));
  auto &graph_map = pack_graph_cache[key];
  auto it = graph_map.find(abs_list);
  if (it != graph_map.end()) {
    return it->second;
  }
  auto prim_py = prim->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(prim_py);
  bool reuse = prim->HasAttr("reuse");
  abstract::AbstractBasePtrList new_abs_list;
  for (auto &i : abs_list) {
    if (reuse && i->cast_ptr<abstract::AbstractRefTensor>()) {
      continue;
    }
    new_abs_list.push_back(i);
  }
  FuncGraphPtr graph;
  PackExpander::is_pynative_mode = false;
  {
    py::gil_scoped_acquire acquire;
    graph = PackFuncConstructGraph(prim_py, new_abs_list);
    if (reuse) {
      graph = PostProcessForReuseGraph(graph, prim_py);
    }
    MS_EXCEPTION_IF_NULL(graph);
  }
  static const bool dump_result = (common::GetEnv("MS_DEV_DUMP_PACK") == "on");
  graph_map[abs_list] = graph;
  if (dump_result) {
    DumpIR("pack_func_" + key + ".ir", graph, true);
  }
  return graph;
}

FuncGraphPtr ExpandPackFuncPynative(const PrimitivePtr &prim, const abstract::AbstractBasePtrList &abs_list,
                                    bool pynative_grad) {
  DynamicShapeCheck(abs_list);
  auto &pack_graph_cache = pynative_pack_cache;
  auto key = GetValue<std::string>(prim->GetAttr("unique_key"));
  auto &graph_map = pack_graph_cache[key];
  auto it = graph_map.find(abs_list);
  if (it != graph_map.end()) {
    return it->second;
  }
  auto prim_py = prim->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(prim_py);
  FuncGraphPtr graph;
  PackExpander::is_pynative_mode = true;
  {
    py::gil_scoped_acquire acquire;
    graph = PackFuncConstructGraph(prim_py, abs_list);
    MS_EXCEPTION_IF_NULL(graph);
  }
  static const bool dump_result = (common::GetEnv("MS_DEV_DUMP_PACK") == "on");
  const auto abs = graph->output()->abstract();
  if (!IsAbstractOutputTensor(abs)) {
    MS_EXCEPTION(ValueError) << "The output of trace captured graph should be one or more flattened Tensor, bug get "
                             << abs->BuildType()->ToString() << ".";
  }
  if (dump_result) {
    DumpIR("pack_func_" + key + "_ori.ir", graph, true);
  }
  // The graph constructed by pack may generate redundant casts, but in the pynative mode,
  // there is no optimization pass to eliminate the cast.
  CastEliminate(graph);
  if (dump_result) {
    DumpIR("pack_func_" + key + "_cast_eliminate.ir", graph, true);
  }
  size_t grad_attach_num = 0;
  if (pynative_grad) {
    const auto &graph_grad_info = GenGraphGradInfo(graph);
    graph = graph_grad_info->graph;
    if (graph->modify_output()) {
      auto ori_output_num = GetSizeByAbstract(abs);
      grad_attach_num = GetSizeByAbstract(graph->output()->abstract()) - ori_output_num;
    }
  }
  CacheResGraph(&graph_map, abs_list, graph, pynative_grad);
  if (dump_result) {
    DumpIR("pack_func_" + key + ".ir", graph, true);
  }
  py::gil_scoped_acquire acquire;
  // SetOutputNum: In order to be able to get the specific output type in PredictOutputType.
  SetOutputNum(prim_py, graph->output());
  SetGradAttachNum(prim_py, grad_attach_num);
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
    // Infer under pynative directly calls the ExpandPackFunc
    if (GetValue<bool>(primitive->GetAttr("is_pynative_mode"))) {
      return nullptr;
    }
    auto graph = ExpandPackFuncGraph(primitive, input_args);
    return graph->output()->abstract();
  }
};
}  // namespace expander
namespace ops {
REGISTER_PRIMITIVE_OP_INFER_IMPL(PackFunc, prim::kPrimPackFunc, expander::PackFuncInfer, false);
}  // namespace ops
}  // namespace mindspore
