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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZATION_OPTIMIZER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZATION_OPTIMIZER_H_
#include <utility>
#include <map>
#include <memory>
#include <vector>
#include "include/backend/optimizer/pass.h"
#include "tools/converter/cxx_api/converter_para.h"

namespace mindspore::lite::quant {
class QuantizationOptimizer {
 public:
  explicit QuantizationOptimizer(const std::shared_ptr<ConverterPara> &param) : param_(param) {}
  ~QuantizationOptimizer() = default;
  int Run(const FuncGraphPtr &func_graph);

 private:
  const std::shared_ptr<ConverterPara> &param_;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANTIZATION_OPTIMIZER_H_
