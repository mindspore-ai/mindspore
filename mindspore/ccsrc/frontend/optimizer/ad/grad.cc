/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/ad/grad.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "frontend/optimizer/irpass.h"
#include "ir/func_graph_cloner.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"

namespace mindspore {
namespace ad {
namespace {
FuncGraphPtr PartialEliminateOptPass(const pipeline::ResourcePtr &resource, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(resource);

  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassConfig partial_eliminate_opt_ = opt::OptPassConfig(
    {irpass.partial_eliminate_, irpass.switch_partial_eliminater_, irpass.switch_layer_partial_eliminater_});
  opt::OptPassGroupMap map({{"partial_eliminate_", partial_eliminate_opt_}});

  auto after_lift_opt = opt::Optimizer::MakeOptimizer("partial_eliminate", resource, map);

  FuncGraphPtr opt_fg = nullptr;
  ProfileExecute(MsProfile::GetProfile()->Step("partial_eliminate_before_grad"),
                 [&after_lift_opt, func_graph, &opt_fg]() { opt_fg = after_lift_opt->step(func_graph, true); });
  return opt_fg;
}

FuncGraphVector PartialEliminateMulti(const pipeline::ResourceBasePtr &resource, const FuncGraphVector &func_graphs) {
  auto new_res = std::dynamic_pointer_cast<pipeline::Resource>(resource);
  if (new_res == nullptr) {
    MS_LOG(EXCEPTION) << "Parameter resources is not a pipeline::Resource";
  }
  FuncGraphVector opt_fgs;
  for (const auto &func_graph : func_graphs) {
    auto opt_fg = PartialEliminateOptPass(new_res, func_graph);
#ifdef ENABLE_DUMP_IR
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->CanDump(kAdvanced)) {
      DumpIR("after_opt_" + opt_fg->ToString() + ".ir", opt_fg);
    }
#endif
    opt_fgs.push_back(opt_fg);
  }
  return opt_fgs;
}

FuncGraphPtr LiftFv(const pipeline::ResourceBasePtr &resource, const FuncGraphPtr &func_graph) {
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool enable_save_graphs = context->CanDump(kAdvanced);
  if (enable_save_graphs) {
    DumpIR("before_lift_" + func_graph->ToString() + ".ir", func_graph);
  }
#endif
  FuncGraphPtr new_fg = LiftingClone(func_graph);
#ifdef ENABLE_DUMP_IR
  if (enable_save_graphs) {
    DumpIR("after_lift_" + new_fg->ToString() + ".ir", new_fg);
  }
#endif
  auto new_res = std::dynamic_pointer_cast<pipeline::Resource>(resource);
  if (new_res == nullptr) {
    MS_LOG(EXCEPTION) << "Parameter resources is not a pipeline::Resource";
  }
  auto opt_fg = PartialEliminateOptPass(new_res, new_fg);
#ifdef ENABLE_DUMP_IR
  if (enable_save_graphs) {
    DumpIR("after_opt_" + opt_fg->ToString() + ".ir", opt_fg);
  }
#endif
  return opt_fg;
}

FuncGraphVector LiftFvMulti(const pipeline::ResourceBasePtr &resource, const FuncGraphVector &func_graphs) {
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kAdvanced)) {
    for (const auto &func_graph : func_graphs) {
      DumpIR("before_lift_" + func_graph->ToString() + ".ir", func_graph);
    }
  }
#endif
  bool has_used_fg = std::any_of(func_graphs.cbegin(), func_graphs.cend(), [](const FuncGraphPtr &func_graph) {
    return func_graph->func_graphs_used().size() != 0;
  });
  // All func_graphs being graded don't have used funcgraphs, no need to do lifting clone.
  if (!has_used_fg) {
    return func_graphs;
  }
  FuncGraphVector new_fgs = LiftingCloneMulti(func_graphs);
#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kAdvanced)) {
    for (const auto &new_fg : new_fgs) {
      DumpIR("after_lift_" + new_fg->ToString() + ".ir", new_fg);
    }
  }
#endif
  return PartialEliminateMulti(resource, new_fgs);
}
}  // namespace

FuncGraphPtr GradOneFuncGraph(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer, bool is_top) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto gradkv = func_graph->transforms().find("grad");
  if (gradkv != func_graph->transforms().end()) {
    return gradkv->second.func_graph();
  }

  const auto &resources = optimizer->resource();
  auto manager_ptr = resources->manager();
  MS_EXCEPTION_IF_NULL(manager_ptr);
  manager_ptr->AddFuncGraph(func_graph);

  auto multi_graph_sink = [&func_graph](const FuncGraphPtr &f) {
    if (MsContext::GetInstance()->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK)) {
      if (func_graph->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE)) {
        f->set_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE, true);
      }
    }
  };

  auto f = std::make_shared<DFunctor>(func_graph, resources, is_top);
  auto user_defined = f->KUserDefined(func_graph);
  if (user_defined != nullptr) {
    multi_graph_sink(user_defined);
    if (is_top) {
      DFunctor::Clear();
    }
    return user_defined;
  }
  f->Init(is_top);
  f->MapObject();
  f->MapMorphism();
  f->Finish();
  auto res = f->k_graph();
  auto tape = f->tape();
  tape->set_flag(mindspore::kFuncGraphFlagBackPropEntry, true);
  if (is_top) {
    DFunctor::Clear();
  }

  multi_graph_sink(res);
  (void)func_graph->transforms().emplace("grad", FuncGraphTransform(res));
  return res;
}

FuncGraphPtr Grad(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer, bool is_top) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto gradkv = func_graph->transforms().find("grad");
  if (gradkv != func_graph->transforms().end()) {
    return gradkv->second.func_graph();
  }

  const auto &resources = optimizer->resource();
  auto manager_ptr = resources->manager();
  MS_EXCEPTION_IF_NULL(manager_ptr);
  manager_ptr->AddFuncGraph(func_graph);

  FuncGraphPtr grad_fg = func_graph;
  if (func_graph->func_graphs_used().size() != 0 && optimizer->is_first_order_j()) {
    lift_fv_before_grad = true;
    grad_fg = LiftFv(resources, func_graph);
  } else {
    lift_fv_before_grad = false;
  }
  return GradOneFuncGraph(grad_fg, optimizer, is_top);
}

FuncGraphVector GradMultiFuncGraph(const FuncGraphVector &func_graphs, const opt::OptimizerPtr &optimizer,
                                   bool is_top) {
  FuncGraphVector grad_fgs;
  if (func_graphs.size() == 1) {
    auto grad_fg = Grad(func_graphs[0], optimizer, is_top);
    grad_fgs.push_back(grad_fg);
    return grad_fgs;
  }
  const auto &resources = optimizer->resource();
  auto manager_ptr = resources->manager();
  MS_EXCEPTION_IF_NULL(manager_ptr);
  for (const auto &func_graph : func_graphs) {
    manager_ptr->AddFuncGraph(func_graph);
  }

  FuncGraphVector before_grad_fgs;
  if (optimizer->is_first_order_j()) {
    lift_fv_before_grad = true;
    before_grad_fgs = LiftFvMulti(resources, func_graphs);
  } else {
    before_grad_fgs = func_graphs;
    lift_fv_before_grad = false;
  }

  for (const auto &func_graph : before_grad_fgs) {
    auto grad_fg = GradOneFuncGraph(func_graph, optimizer, is_top);
    grad_fgs.push_back(grad_fg);
  }
  return grad_fgs;
}

FuncGraphPtr Kprim(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources) {
  auto fg = g_k_prims.KPrimitive(nullptr, value_node, resources);
  if (fg == nullptr) {
    return nullptr;
  }
  return BasicClone(fg);
}

MetaFuncGraphPtr Kmeta(const PrimitivePtr &prim, const pipeline::ResourceBasePtr &) {
  MetaFuncGraphPtr fg = g_k_prims.KMetaFuncGraph(prim);
  return fg;
}

void CleanRes() { DFunctor::Clear(); }
}  // namespace ad
}  // namespace mindspore
