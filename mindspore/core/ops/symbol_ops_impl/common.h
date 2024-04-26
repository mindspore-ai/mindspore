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
#ifndef MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_COMMON_H_
#define MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_COMMON_H_
#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <utility>

#include "mindspore/core/symbolic_shape/symbol.h"
#include "mindspore/core/symbolic_shape/utils.h"
#include "mindspore/core/symbolic_shape/operation.h"
#include "mindspore/core/symbolic_shape/operation_builder.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API InferShapeOp : public Operation {
 public:
  using Operation::Operation;
  ~InferShapeOp() override = default;
  MS_DECLARE_PARENT(InferShapeOp, Operation)
  static void SetPositive(const ListSymbol *list);

 protected:
  void UpdateMathInfo() override { SetPositive(output_as<ListSymbol>()); }
};

class MS_CORE_API InferValueOp : public Operation {
 public:
  using Operation::Operation;
  ~InferValueOp() override = default;
  MS_DECLARE_PARENT(InferValueOp, Operation)
};

class MS_CORE_API ScalarOp : public InferValueOp {
 public:
  using InferValueOp::InferValueOp;
  MS_DECLARE_PARENT(ScalarOp, InferValueOp)
};

/// \brief Use the input symbol as output directly.
///
/// \note When using this function, the `SetShapeDepend` or `SetValueDepend` should be set, and only one
/// "DependOn::kShape" (or "DependOn::kValue") exists. the depending symbol is used as output.
SymbolPtr TransparentInput(OperationBuilder *b);

/// \brief The default builder to create an `Operation`, input shapes or values are related to the depend list.
///
/// When using this function in `SetShapeFunc`, it generates inputs according to the depend list of `SetShapeDepend`.
/// if the shape depend list is not set, all inputs are treated as depending Shape.
/// When using this function in `SetValueFunc`, it generates inputs according to the depend list of `SetValueDepend`.
/// if the value depend list is not set, all inputs are treated as depending Value.
///
/// \tparam OP The class that inherit from `Operation`.
template <typename OP, typename = std::enable_if_t<std::is_base_of_v<Operation, OP>>>
SymbolPtr DefaultBuilder(OperationBuilder *b) {
  bool build_value = !b->is_building_shape();
  auto depends = b->symbol_builder_info().GetDepends(b->prim(), build_value);
  if (depends.empty()) {
    depends.resize(b->input_num(), (build_value ? DependOn::kValue : DependOn::kShape));
  }
  if (b->input_num() < depends.size()) {
    MS_LOG(WARNING) << "For " << b->prim()->name() << ", the input args num is less than the depends size. "
                    << b->input_num() << " vs " << depends.size();
    return nullptr;
  }
  SymbolPtrList inputs;
  inputs.reserve(depends.size());
  for (size_t i = 0; i < depends.size(); i++) {
    if (depends[i] == DependOn::kShape) {
      (void)inputs.emplace_back(b->GetInputShape(i));
    } else if (depends[i] == DependOn::kValue) {
      (void)inputs.emplace_back(b->GetInputValue(i));
    }
  }
  return b->Emit(std::make_shared<OP>(std::move(inputs)));
}
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_COMMON_H_
