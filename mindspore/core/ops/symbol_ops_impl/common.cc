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
#include "mindspore/core/ops/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
namespace ops {
void InferShapeOp::SetPositive(const ListSymbol *list) {
  for (auto &s : list->symbols()) {
    auto list_s = s->as<ListSymbol>();
    if (list_s != nullptr) {
      SetPositive(list_s);
    } else {
      auto int_s = s->as<IntSymbol>();
      MS_EXCEPTION_IF_NULL(int_s);
      if (!int_s->is_positive()) {
        int_s->SetRangeMin(1);
      }
    }
  }
}
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
