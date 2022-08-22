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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_REMOVE_UNUSED_QUANT_PARAM_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_REMOVE_UNUSED_QUANT_PARAM_H

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore::lite::quant {
class RemoveQuantParam {
 public:
  explicit RemoveQuantParam(const FuncGraphPtr &funcGraph) : func_graph_(funcGraph) {}
  ~RemoveQuantParam() = default;

 public:
  int Remove();

 private:
  FuncGraphPtr func_graph_ = nullptr;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_REMOVE_UNUSED_QUANT_PARAM_H
