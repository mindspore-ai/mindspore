/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef LITE_TFLITE_INPUTS_ORDER_EXCHANGE_PASS_H
#define LITE_TFLITE_INPUTS_ORDER_EXCHANGE_PASS_H

#include <string>
#include "schema/inner/model_generated.h"
#include "tools/converter/converter_flags.h"
#include "backend/optimizer/common/pass.h"
#include "src/param_value_lite.h"

namespace mindspore::opt {
class TfliteInputsOrderExchangePass : public Pass {
 public:
  TfliteInputsOrderExchangePass() : Pass("tflite_inputs_order_exchange_pass") {}
  ~TfliteInputsOrderExchangePass() override = default;
  bool Run(const FuncGraphPtr &graph) override;
};
}  // namespace mindspore::opt
#endif  // LITE_TFLITE_INPUTS_ORDER_EXCHANGE_PASS_H
