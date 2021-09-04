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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_WEIGHT_QUANTIZER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_WEIGHT_QUANTIZER_H_

#include <future>
#include <memory>
#include <unordered_map>
#include <map>
#include <list>
#include <string>
#include <vector>
#include "tools/converter/quantizer/quantizer.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/quant_params.h"
#include "tools/converter/preprocess/preprocess_param.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/model.h"
#include "base/base.h"
#include "abstract/dshape.h"
#include "src/lite_session.h"

namespace mindspore::lite::quant {
class WeightQuantizer : public Quantizer {
 public:
  WeightQuantizer(FuncGraphPtr graph, const converter::Flags &config);
  ~WeightQuantizer() override;

  STATUS DoQuantize(FuncGraphPtr func_graph) override;
  STATUS DoConvQuantize(const CNodePtr &cnode);
  STATUS DoMulQuantize(const CNodePtr &cnode);
  STATUS DoOptimizerQuantize(const CNodePtr &cnode);
  STATUS DoLstmQuantize(const CNodePtr &cnode);
  STATUS DoGatherQuantize(const CNodePtr &cnode);

  STATUS ProcessLstmWeightByIndex(const CNodePtr &cnode, const PrimitivePtr &primitive, const int &index);

  int quant_max_{127};
  int quant_min_{-128};
  TypeId type_id_{kNumberTypeInt8};

 private:
  std::unique_ptr<QuantStrategy> quant_strategy_;
  size_t bit_num_{8};
  std::map<tensor::TensorPtr, ParameterPtr> weight_quantized_tensors_;
  std::vector<std::unordered_map<std::string, mindspore::tensor::MSTensor *>> fp32_output_tensors_;
  bool is_mixed_bit_ = false;

  STATUS SetAbstract(const tensor::TensorPtr &tensor_info, const ParameterPtr &param_node,
                     const PrimitivePtr &primitive);
  STATUS MarkWeightQuantizationInNodes(const FuncGraphPtr &);
  STATUS DoMarkWeightQuantizeIfQuantized(const CNodePtr &);
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_WEIGHT_QUANTIZER_H_
