/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_QAT_TRANSFORM_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_QAT_TRANSFORM_H_

#include <memory>
#include <set>
#include "base/base.h"
#include "tools/converter/cxx_api/converter_para.h"
namespace mindspore::lite::quant {
class QATTransform {
 public:
  QATTransform(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param)
      : func_graph_(func_graph), param_(param) {}
  ~QATTransform() = default;
  int Transform();
  int StaticWeightQuantInfo(const FuncGraphPtr &func_graph,
                            const std::set<PrimitivePtr> &per_channel_primitive_types = {});

 private:
  int DoSingleGraphQATTransform(const FuncGraphPtr &func_graph);
  bool CheckWeightQuantExist(const CNodePtr &cnode);
  FuncGraphPtr func_graph_ = nullptr;
  std::shared_ptr<ConverterPara> param_ = nullptr;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_QAT_TRANSFORM_H_
