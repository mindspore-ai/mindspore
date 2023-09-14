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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_OPERATIONS_INFERVALUE_OP_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_OPERATIONS_INFERVALUE_OP_H_
#include <memory>
#include <vector>
#include <string>

#include "backend/common/graph_kernel/symbol_engine/symbol.h"
#include "backend/common/graph_kernel/symbol_engine/utils.h"
#include "backend/common/graph_kernel/symbol_engine/operations/operation.h"

namespace mindspore::graphkernel::symbol {
namespace ops::infervalue {
class RealValue : public Operation {
 public:
  explicit RealValue(const SymbolPtr &inp) : Operation({inp}) {}
  ~RealValue() override = default;
  MS_DECLARE_PARENT(RealValue, Operation)

 protected:
  SymbolPtr Eval() override;
};
}  // namespace ops::infervalue
}  // namespace mindspore::graphkernel::symbol
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_OPERATIONS_INFERVALUE_OP_H_
