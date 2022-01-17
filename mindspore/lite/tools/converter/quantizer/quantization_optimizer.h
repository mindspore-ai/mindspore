/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZATION_OPTIMIZER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZATION_OPTIMIZER_H
#include <utility>
#include <map>
#include <vector>
#include "backend/optimizer/common/pass.h"
#include "tools/converter/converter_flags.h"

namespace mindspore::lite::quant {
class QuantizationOptimizer {
 public:
  explicit QuantizationOptimizer(converter::Flags *flags) : flags_(flags) {}
  ~QuantizationOptimizer() = default;
  int Run(const FuncGraphPtr &func_graph);

 private:
  converter::Flags *flags_;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZATION_OPTIMIZER_H
