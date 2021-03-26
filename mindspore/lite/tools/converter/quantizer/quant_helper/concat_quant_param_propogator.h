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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_PREPROCESSOR_CONCAT_QUANT_PARAM_PROPOGATOR_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_PREPROCESSOR_CONCAT_QUANT_PARAM_PROPOGATOR_H

#include "tools/converter/quantizer/quant_helper/quant_node_helper.h"
namespace mindspore::lite {
class ConcatQuantParamPropogator : public QuantParamPropogator {
 public:
  STATUS PropogateQuantParams(schema::MetaGraphT *graph, const schema::CNodeT &node) override;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_PREPROCESSOR_CONCAT_QUANT_PARAM_PROPOGATOR_H
