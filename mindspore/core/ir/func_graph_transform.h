/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CORE_IR_FUNC_GRAPH_TRANSFORM_H_
#define MINDSPORE_MINDSPORE_CORE_IR_FUNC_GRAPH_TRANSFORM_H_
#include "ir/anf.h"

namespace mindspore {
// ANF transform class.
// Either a primitive or a func_graph.
class MS_CORE_API FuncGraphTransform {
 public:
  enum Type { kGtPrimitive, kGtFuncGraph };

  explicit FuncGraphTransform(const PrimitivePtr prim, const FuncGraphPtr func_graph = nullptr)
      : prim_(prim), func_graph_(FuncGraphWeakPtr(func_graph)) {}

  explicit FuncGraphTransform(const FuncGraphPtr &func_graph, const PrimitivePtr &prim = func_graph_prim_)
      : prim_(prim), func_graph_(FuncGraphWeakPtr(func_graph)) {}

  FuncGraphTransform(const FuncGraphTransform &t) : prim_(t.prim_), func_graph_(t.func_graph_) {}

  ~FuncGraphTransform() = default;

  Type type() const {
    if (IsFuncGraph()) {
      return kGtFuncGraph;
    } else {
      return kGtPrimitive;
    }
  }

  bool IsPrimitive() const { return (func_graph_.lock() == nullptr); }
  bool IsFuncGraph() const { return (func_graph_.lock() != nullptr); }
  FuncGraphPtr func_graph() const { return func_graph_.lock(); }
  PrimitivePtr primitive() const { return prim_; }

  FuncGraphTransform &operator=(const FuncGraphTransform &t) {
    if (this != &t) {
      prim_ = t.prim_;
      func_graph_ = t.func_graph_;
    }
    return *this;
  }

 private:
  PrimitivePtr prim_;
  // FuncGraph will be hold by FuncGraphManager, so weak_ptr is enough here.
  // And use weak_ptr can break the reference cycle between "primal" and "grad" graph in
  // FPropRemapper::FinalizeGraph().
  FuncGraphWeakPtr func_graph_;
  static const PrimitivePtr func_graph_prim_;
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CORE_IR_FUNC_GRAPH_TRANSFORM_H_
