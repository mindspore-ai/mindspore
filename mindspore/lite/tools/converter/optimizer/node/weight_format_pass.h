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

#ifndef MINDSPORE_PREDICT_WEIGHT_FORMAT_PASS_H
#define MINDSPORE_PREDICT_WEIGHT_FORMAT_PASS_H

#include "tools/converter/optimizer.h"
#include "tools/converter/converter_flags.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace lite {
class WeightFormatPass : public NodePass {
 public:
  WeightFormatPass() = default;

  ~WeightFormatPass() override = default;

  void SetQuantType(QuantType quantType);

  void SetFmkType(converter::FmkType fmkType);

  int Run(GraphNode *graphNode) override;

 private:
  // correct weightTensor->Format
  int ShapeFormatTrans(GraphNode *graphNode);

  // transform weightTensor data and format
  // if quant : conv transform dataFormat to NHWC, weight format to HWCK
  // if quant : depth transform dataFormat to NCHW, weight format to CKHW
  int QuantDataFormatTrans(GraphNode *graphNode);

  // if no quant : transform dataFormat to NCHW, weight format to KCHW/CKHW
  int NonQuantDataFormatTrans(GraphNode *graphNode);

 private:
  QuantType quantType = QuantType_QUANT_NONE;
  converter::FmkType fmkType = converter::FmkType_TF;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_PREDICT_WEIGHT_FORMAT_PASS_H

