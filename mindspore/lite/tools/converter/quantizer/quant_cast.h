/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER__QUANT_CAST_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER__QUANT_CAST_H

#include "include/errorcode.h"
#include "ir/anf.h"
#include "ir/dtype/type_id.h"
#include "ir/func_graph.h"

namespace mindspore::lite::quant {
class QuantCast {
 public:
  QuantCast() = default;
  ~QuantCast() = default;
  STATUS Run(const FuncGraphPtr &graph);
  void SetInputDataDType(TypeId dataType) { this->inputDataDType = dataType; }

 private:
  TypeId inputDataDType = kNumberTypeFloat32;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER__QUANT_CAST_H
