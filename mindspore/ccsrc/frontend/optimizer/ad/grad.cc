/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
FuncGraphPtr PartialEliminateOptPass(const ResourcePtr &resource, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(resource);

  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassConfig partial_eliminate_opt_ = opt::OptPassConfig(
    {irpass.partial_eliminate_, irpass.switch_partial_eliminater_, irpass.switch_layer_partial_eliminater_});
  opt::OptPassGroupMap map({{"partial_eliminate_", partial_eliminate_opt_}});

  auto after_lift_opt = opt::Optimizer::MakeOptimizer("partial_eliminate", resource, map);

  FuncGraphPtr opt_fg = nullptr;
  WITH(MsProfile::GetProfile()->Step("partial_eliminate_before_grad"))[&after_lift_opt, func_graph, &opt_fg]() {
    opt_fg = after_lift_opt->step(func_graph, true);
  };
  return opt_fg;
}

FuncGraphPtr LiftFv(const pipeline::ResourceBasePtr &resource, const FuncGraphPtr &func_graph) {
#ifdef ENABLE_DUMP_IR
  bool save_graphs_flag = MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs_flag) {
    DumpIR("before_lift_" + func_graph->ToString() + ".ir", func_graph);
  }
#endif
  FuncGraphPtr new_fg = LiftingClone(func_graph);
#ifdef ENABLE_DUMP_IR
  if (save_graphs_flag) {
    DumpIR("after_lift_" + new_fg->ToString() + ".ir", new_fg);
  }
#endif
  auto new_res = std::dynamic_pointer_cast<pipeline::Resource>(resource);
  if (new_res == nullptr) {
    MS_LOG(EXCEPTION) << "Parameter resources is not a pipeline::Resource";
  }
  auto opt_fg = PartialEliminateOptPass(new_res, new_fg);
#ifdef ENABLE_DUMP_IR
  if (save_graphs_flag) {
    DumpIR("after_opt_" + opt_fg->ToString() + ".ir", opt_fg);
  }
#endif
  return opt_fg;
}
}  // namespace

FuncGraphPtr Grad(const FuncGraphPtr &func_graph, const pipeline::ResourceBasePtr &resources, bool is_top) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto gradkv = func_graph->transforms().find("grad");
  if (gradkv != func_graph->transforms().end()) {
    return gradkv->second.func_graph();
  }

  auto manager_ptr = resources->manager();
  MS_EXCEPTION_IF_NULL(manager_ptr);
  manager_ptr->AddFuncGraph(func_graph);

  FuncGraphPtr grad_fg = func_graph;
  lift_fv_before_grad = (common::GetEnv("ENV_DONT_LIFT_FV_BEFORE_GRAD") != "1");
  if (lift_fv_before_grad && func_graph->func_graphs_used().size() != 0) {
    grad_fg = LiftFv(resources, func_graph);
  }
  auto multi_graph_sink = [&func_graph](const FuncGraphPtr &f) {
    if (MsContext::GetInstance()->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK)) {
      if (func_graph->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUES)) {
        f->set_flag(FUNC_GRAPH_FLAG_IGNORE_VALUES, true);
      }
    }
  };

  auto f = std::make_shared<DFunctor>(grad_fg, resources);
  auto user_defined = f->KUserDefined(grad_fg);
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
  if (func_graph != grad_fg) {
    (void)func_graph->transforms().insert(std::make_pair("grad", FuncGraphTransform(res)));
  }
  return res;
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
