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
#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_SET_INFER_FUNCTOR_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_SET_INFER_FUNCTOR_H_
#include <string>
#include <vector>
#include <memory>

#include "ir/func_graph.h"
#include "runtime/hardware_abstract/visible.h"
#include "include/backend/common/pass_manager/pass.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/infershape_functor.h"

namespace mindspore::graphkernel {
namespace symshape {
class CppVisitor;
using CppVisitorPtr = std::shared_ptr<CppVisitor>;
}  // namespace symshape

using opt::dynamic_shape::InferShapeFunctor;
using opt::dynamic_shape::kAttrInferShapeFunctor;
using DynFuncType = void (*)(const int64_t **, int64_t **);

class RUNTIME_HARDWARE_EXPORT SymbolEngineInfer : public InferShapeFunctor {
 public:
  SymbolEngineInfer(const std::string &name, const SymbolEnginePtr &engine, const AbstractBasePtr &out_abstract)
      : InferShapeFunctor(name), engine_(engine), out_abstract_(out_abstract) {}
  ~SymbolEngineInfer() override = default;
  MS_DECLARE_PARENT(SymbolEngineInfer, InferShapeFunctor)
  BaseShapePtr InferShape(const AbstractBasePtrList &args) override;

 protected:
  SymbolEnginePtr engine_;
  AbstractBasePtr out_abstract_;
};

class SymbolEngineJitInfer : public InferShapeFunctor {
 public:
  SymbolEngineJitInfer(const std::string &name, const std::string &func_name,
                       const symshape::CppVisitorPtr &cpp_visitor, const ListSymbolPtr &output_symbol)
      : InferShapeFunctor(name), func_name_(func_name), cpp_visitor_(cpp_visitor), output_symbol_(output_symbol) {
    Init();
  }
  MS_DECLARE_PARENT(SymbolEngineJitInfer, InferShapeFunctor)
  BaseShapePtr InferShape(const AbstractBasePtrList &args) override;

 protected:
  void Init();

 private:
  std::string func_name_;
  symshape::CppVisitorPtr cpp_visitor_;
  ListSymbolPtr output_symbol_;
  DynFuncType infer_func_ = nullptr;
  std::vector<int64_t *> output_parm_;
  ShapeArray out_shapes_;
};

class RUNTIME_HARDWARE_EXPORT SetInferShapeFunctor : public opt::Pass {
 public:
  explicit SetInferShapeFunctor(const std::string &pass_name = "set_infershape_funtor") : Pass(pass_name) {}
  ~SetInferShapeFunctor() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_SET_INFER_FUNCTOR_H_
