/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_STRATEGY_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_STRATEGY_H
#include <cstddef>
#include "ir/anf.h"

namespace mindspore::lite::quant {
class QuantStrategy {
 public:
  QuantStrategy(size_t min_quant_weight_size, size_t min_quant_weight_channel)
      : min_quant_weight_size_(min_quant_weight_size), min_quant_weight_channel_(min_quant_weight_channel) {}

  ~QuantStrategy() = default;

  static bool CanOpFullQuantized(const AnfNodePtr &node);
  bool CanTensorQuantized(const AnfNodePtr &input_node, int preferred_dim);

 private:
  size_t min_quant_weight_size_;
  size_t min_quant_weight_channel_;
};
}  // namespace mindspore::lite::quant

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_STRATEGY_H
