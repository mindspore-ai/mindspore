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

#ifndef WEIGHT_QUANTIZER_H
#define WEIGHT_QUANTIZER_H

#include <memory>
#include <list>
#include <string>
#include "tools/converter/quantizer/quantizer.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/model.h"
#include "base/base.h"
#include "abstract/dshape.h"

namespace mindspore {
namespace lite {
namespace quant {
class WeightQuantizer : public Quantizer {
 public:
  WeightQuantizer(FuncGraphPtr graph, const std::string& weightSize,
                  const std::string& covWeightChannelThreshold, const std::string& bitNum);

  ~WeightQuantizer() = default;

  STATUS DoQuantize(FuncGraphPtr funcGraph) override;
  STATUS DoConvQuantize(const std::list<CNodePtr> &nodes);
  STATUS DoMulQuantize(const std::list<CNodePtr> &nodes);

 private:
  std::unique_ptr<QuantStrategy> mStrategy;
  size_t bitNum;
};
}  // namespace quant
}  // namespace lite
}  // namespace mindspore
#endif

