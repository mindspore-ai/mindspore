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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_ASCEND_DISTRIBUTE_FAKE_QUANT_TRANSFORM
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_ASCEND_DISTRIBUTE_FAKE_QUANT_TRANSFORM

#include <memory>
#include <set>
#include <vector>
#include "base/base.h"
#include "tools/converter/cxx_api/converter_para.h"
#include "tools/converter/quantizer/quant_helper/qat_transform.h"

namespace mindspore::lite::quant {
class AscendDistributeFakeQuantTransform {
 public:
  AscendDistributeFakeQuantTransform(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param)
      : func_graph_(func_graph), param_(param) {}

  ~AscendDistributeFakeQuantTransform() = default;

  int Transform();

 private:
  int DoSingleGraphAscendDistributeFakeQuantTransform(const FuncGraphPtr &func_graph);

  int SetInputQuantParam(const FuncGraphPtr &func_graph);

  int SetWeightQuantParam(const FuncGraphPtr &func_graph);

  std::vector<schema::QuantParamT> GetQuantParamWithFakeQuantNode(const CNodePtr &depend_node, bool symmetric = false);

  std::vector<schema::QuantParamT> CalQuantParam(const tensor::TensorPtr &min_value, const tensor::TensorPtr &max_value,
                                                 bool symmetric = false);

  int InsertAscendQuantDeQuantNode(const FuncGraphPtr &func_graph);

  int MatMulWeightTranspose(const FuncGraphPtr &func_graph);

  int RemoveWeightRedundantNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode);

  int NeedAscendDistributeFakeQuantTransform(const FuncGraphPtr &func_graph);

  int FetchWeightQuantParamFromFakeQuant(const FuncGraphPtr &func_graph);

  int PreProcess(const FuncGraphPtr &func_graph);

 private:
  FuncGraphPtr func_graph_{nullptr};
  const std::shared_ptr<mindspore::ConverterPara> param_{nullptr};
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_HELPER_ASCEND_DISTRIBUTE_FAKE_QUANT_TRANSFORM
