/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_WEIGHT_QUANTIZER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_WEIGHT_QUANTIZER_H

#include <future>
#include <memory>
#include <map>
#include <list>
#include <string>
#include <vector>
#include "tools/converter/quantizer/quantizer.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/model.h"
#include "base/base.h"
#include "abstract/dshape.h"
#include "src/lite_session.h"

namespace mindspore::lite::quant {
class WeightQuantizer : public Quantizer {
 public:
  WeightQuantizer(FuncGraphPtr graph, const std::string &config_file, const std::string &weightSize,
                  const std::string &covWeightChannelThreshold, const std::string &bitNum);
  WeightQuantizer(FuncGraphPtr graph, const PostQuantConfig &config);
  ~WeightQuantizer();

  STATUS DoQuantize(FuncGraphPtr func_graph) override;
  STATUS DoConvQuantize(const std::list<CNodePtr> &nodes);
  STATUS DoMulQuantize(const std::list<CNodePtr> &nodes);
  static STATUS WeightQuantInputCheck(const converter::Flags *config);
  static bool IsPosNum(const std::string &str);

  int quant_max;
  int quant_min;
  TypeId type_id{kTypeUnknown};
  std::map<std::string, int> opname_bit_;

 private:
  std::unique_ptr<QuantStrategy> quant_strategy_;
  size_t bit_num_;
  std::string config_file_;
  PostQuantConfig config_param_;
  std::vector<std::vector<std::string>> images_;  // multi_input, [[mode_input_0], [model_input_1]...]
  session::LiteSession *fp32_session_ = nullptr;

  STATUS DoMiexedQuant(FuncGraphPtr);
  STATUS SetAbstract(ParamValueLitePtr param_value, ParameterPtr param_node, std::shared_ptr<PrimitiveC> primitive_c);
};
}  // namespace mindspore::lite::quant
#endif
