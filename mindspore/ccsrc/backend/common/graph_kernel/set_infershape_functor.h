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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SET_INFERSHAPE_FUNCTOR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SET_INFERSHAPE_FUNCTOR_H_
#include <string>

#include "ir/func_graph.h"
#include "ir/functor.h"
#include "include/backend/visible.h"
#include "include/backend/optimizer/pass.h"

namespace mindspore::graphkernel {
class SymbolEngineInfer : public InferShapeFunctor {
 public:
  explicit SymbolEngineInfer(const std::string &name) : InferShapeFunctor(name) {}
  ~SymbolEngineInfer() override = default;
  MS_DECLARE_PARENT(SymbolEngineInfer, InferShapeFunctor)
  BaseShapePtr InferShape(const CNodePtr &cnode, const AbstractBasePtrList &args) override;
};

class SetInferShapeFunctor : public opt::Pass {
 public:
  explicit SetInferShapeFunctor(const std::string &pass_name = "set_infershape_funtor") : Pass(pass_name) {}
  ~SetInferShapeFunctor() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};

}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_SET_INFER_FUNCTOR_H_
