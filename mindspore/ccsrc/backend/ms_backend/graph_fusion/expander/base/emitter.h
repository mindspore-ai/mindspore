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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_BASE_EMITTER_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_BASE_EMITTER_H_

#include <memory>

#include "ir/anf.h"
#include "ir/tensor.h"
#include "backend/ms_backend/graph_fusion/expander/base/meta_op.h"
#include "backend/ms_backend/graph_fusion/expander/base/node.h"

namespace mindspore::graphkernel::expander {
class Emitter {
 public:
  Emitter() = default;
  virtual ~Emitter() = default;

  /// \brief emit an operator node
  inline NodePtr Emit(MetaOp op, const NodePtrList &args, const NodePtrDict &kargs) { return EmitOp(op, args, kargs); }
  /// \brief emit an operator node
  inline NodePtr Emit(MetaOp op, const NodePtrList &args) {
    static const NodePtrDict kargs{};
    return EmitOp(op, args, kargs);
  }
  /// \brief emit a value node
  virtual NodePtr EmitValue(const ValuePtr &value) = 0;

 protected:
  virtual NodePtr EmitOp(MetaOp op, const NodePtrList &args, const NodePtrDict &kargs) = 0;
};
using EmitterPtr = std::shared_ptr<Emitter>;
}  // namespace mindspore::graphkernel::expander
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_BASE_EMITTER_H_
