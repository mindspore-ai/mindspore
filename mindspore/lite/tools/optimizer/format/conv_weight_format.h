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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FORMAT_CONV_WEIGHT_FORMAT_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FORMAT_CONV_WEIGHT_FORMAT_H_

#include <string>
#include "backend/optimizer/common/pass.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {
class ConvWeightFormatBase : public Pass {
 public:
  explicit ConvWeightFormatBase(const std::string &name = "ConvWeightFormatBase") : Pass(name) {}
  ~ConvWeightFormatBase() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  STATUS ConvWeightFormatTrans(const FuncGraphPtr &graph);
  STATUS TransferConvWeight(const AnfNodePtr &weight_node);

 protected:
  schema::Format src_format_{schema::Format_KHWC};
  schema::Format dst_format_{schema::Format_KHWC};
};

class ConvWeightToKHWC : public ConvWeightFormatBase {
 public:
  ConvWeightToKHWC() : ConvWeightFormatBase("ConvWeightToKHWC") { src_format_ = schema::Format_KCHW; }
  ~ConvWeightToKHWC() override = default;
};

class ConvWeightToKCHW : public ConvWeightFormatBase {
 public:
  ConvWeightToKCHW() : ConvWeightFormatBase("ConvWeightToKCHW") { dst_format_ = schema::Format_KCHW; }
  ~ConvWeightToKCHW() override = default;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FORMAT_CONV_WEIGHT_FORMAT_H_
