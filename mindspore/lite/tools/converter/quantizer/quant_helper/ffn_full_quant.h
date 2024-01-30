/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FFN_ANTIQUANT_FUSION_H
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_FFN_ANTIQUANT_FUSION_H

#include <memory>
#include <set>
#include <vector>
#include "base/base.h"
#include "tools/converter/cxx_api/converter_para.h"

namespace mindspore::lite::quant {
class FFNFullQuant {
 public:
  FFNFullQuant(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param)
      : func_graph_(func_graph), param_(param) {}

  ~FFNFullQuant() = default;

  int Transform();

 private:
  int PreProcess(const FuncGraphPtr &func_graph);
  int DoWeightQuantWithFakeQuantNode(const FuncGraphPtr &func_graph, const CNodePtr ffn_cnode, int index);
  int IsFullQuantNode(const CNodePtr &cnode);
  bool CheckFFNNeedFullQuant(const FuncGraphPtr &func_graph);
  int Process(const FuncGraphPtr &func_graph, const CNodePtr &cnode);
  int DoSingleGraphFFNFullQuantTransform(const FuncGraphPtr &func_graph);

 private:
  FuncGraphPtr func_graph_{nullptr};
  const std::shared_ptr<mindspore::ConverterPara> param_{nullptr};
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_ASCEND_DISTRIBUTE_FAKE_QUANT_TRANSFORM
