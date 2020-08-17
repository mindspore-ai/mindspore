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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_WEIGHT_FORMAT_HARDCODE_PASS_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_WEIGHT_FORMAT_HARDCODE_PASS_H

#include <memory>
#include "tools/converter/converter_flags.h"
#include "tools/converter/optimizer.h"
#include "tools/common/graph_util.h"

namespace mindspore {
namespace lite {
class WeightFormatHardCodePass : public GraphPass {
 public:
  WeightFormatHardCodePass() = default;

  ~WeightFormatHardCodePass() override = default;

  void SetQuantType(QuantType quantType);

  void SetFmkType(converter::FmkType fmkType);

  STATUS Run(MetaGraphT *graph) override;

 private:
  STATUS HardCodeCAFFE(const std::unique_ptr<CNodeT> &node, const std::unique_ptr<TensorT> &weightTensor);
  STATUS HardCodeTFLITE(const std::unique_ptr<CNodeT> &node, const std::unique_ptr<TensorT> &weightTensor);
  STATUS HardCodeONNX(const std::unique_ptr<CNodeT> &node, const std::unique_ptr<TensorT> &weightTensor);
  STATUS HardCodeMS(const std::unique_ptr<CNodeT> &node, const std::unique_ptr<TensorT> &weightTensor);

 private:
  QuantType quantType = QuantType_QUANT_NONE;
  converter::FmkType fmkType = converter::FmkType_TF;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_LEGACY_OPTIMIZER_WEIGHT_FORMAT_HARDCODE_PASS_H
