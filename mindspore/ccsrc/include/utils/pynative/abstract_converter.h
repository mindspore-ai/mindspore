/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_COMMON_PYNATIVE_ABSTRACT_CONVERTER_H
#define MINDSPORE_CCSRC_COMMON_PYNATIVE_ABSTRACT_CONVERTER_H

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <optional>
#include "include/utils/visible.h"
#include "include/utils/pynative/ring_buffer.h"
#include "ir/tensor.h"

namespace mindspore {
namespace pynative {
// For get abstract from value and cache abstract
constexpr size_t kAbstractCacheSize = 8192;
class COMMON_EXPORT AbstractConverter {
 public:
  using AbstractCache = kernel::pyboost::RingBuffer<AbstractBasePtr, kAbstractCacheSize>;
  void CacheAbstract(const AbstractBasePtr &abstract);
  AbstractBasePtr ConvertAbstract(const ValuePtr &t);
  // Tensor is held by Abstract, may lead to memory leak.
  AbstractBasePtr ConvertAbstract(const tensor::TensorPtr &t);
  AbstractBasePtr ConvertAbstract(const std::vector<tensor::TensorPtr> &t);
  AbstractBasePtr ConvertAbstract(const ValueTuplePtr &t);

  template <typename T>
  AbstractBasePtr ConvertAbstract(const std::optional<T> &t) {
    if (!t.has_value()) {
      return kNone->ToAbstract();
    }
    return ConvertAbstract(t.value());
  }
  void clear() {}
};
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_COMMON_PYNATIVE_ABSTRACT_CONVERTER_H
