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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_DYNAMIC_QUANTIZER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_DYNAMIC_QUANTIZER_H_

#include <future>
#include <memory>
#include <unordered_map>
#include <map>
#include <list>
#include <string>
#include <utility>
#include <vector>
#include <set>
#include "tools/converter/quantizer/quantizer.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/quant_params.h"
#include "tools/converter/quantizer/quant_strategy.h"
#include "tools/converter/preprocess/preprocess_param.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/model.h"
#include "base/base.h"
#include "abstract/dshape.h"
#include "src/common/quant_utils.h"

namespace mindspore::lite::quant {
class DynamicQuantizer : public Quantizer {
 public:
  explicit DynamicQuantizer(const std::shared_ptr<ConverterPara> &param) : Quantizer(param) {
    bit_num_ = param->commonQuantParam.bit_num;
    activation_channel_weight_layer_ =
      (param->dynamicQuantParam.quant_strategy == quant::ACTIVATION_CHANNEL_WEIGHT_LAYER);
  }
  ~DynamicQuantizer() = default;

  int DoQuantize(FuncGraphPtr func_graph) override;

 private:
  size_t bit_num_{8};
  int quant_max_{127};
  int quant_min_{-128};
  TypeId type_id_{kNumberTypeInt8};
  bool activation_channel_weight_layer_ = false;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_WEIGHT_QUANTIZER_H_
