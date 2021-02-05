/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "ir/func_graph_cloner.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"

namespace mindspore {
namespace ad {
FuncGraphPtr Grad(const FuncGraphPtr &func_graph, const pipeline::ResourceBasePtr &resources, bool is_top) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto gradkv = func_graph->transforms().find("grad");
  if (gradkv != func_graph->transforms().end()) {
    return gradkv->second.func_graph();
  }

  auto manager_ptr = resources->manager();
  MS_EXCEPTION_IF_NULL(manager_ptr);
  manager_ptr->AddFuncGraph(func_graph);

  auto multi_graph_sink = [&func_graph](const FuncGraphPtr &f) {
    if (MsContext::GetInstance()->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK)) {
      if (func_graph->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUES)) {
        f->set_flag(FUNC_GRAPH_FLAG_IGNORE_VALUES, true);
      }
    }
  };

  auto f = std::make_shared<DFunctor>(func_graph, resources);
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
