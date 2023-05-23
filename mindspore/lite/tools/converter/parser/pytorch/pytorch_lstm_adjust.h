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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TORCH_LSTM_ADJUST_PASS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_TORCH_LSTM_ADJUST_PASS_H_
#include <vector>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include "ops/lstm.h"
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/format_utils.h"

namespace mindspore {
namespace opt {
class PytorchLstmAdjustPass {
 public:
  PytorchLstmAdjustPass() {}
  ~PytorchLstmAdjustPass() = default;
  bool Run(const FuncGraphPtr &func_graph);

 private:
  bool AdjustDataFormat(const ParameterPtr &parameter);
  bool AdjustInputShape(const ParameterPtr &weight_input, const ParameterPtr &weight_hidden,
                        const int64_t &bidirectional_dim);
  ParameterPtr CombineTwoBiasInput(const ParameterPtr &bias_input, const ParameterPtr &bias_hidden,
                                   const FuncGraphPtr &func_graph, const CNodePtr &lstm,
                                   const int64_t &bidirectional_dim, const int64_t &hidden_size);
  bool GetAndSetHiddenSize(const ParameterPtr &weight_input, const api::SharedPtr<ops::LSTM> &lstm_prim,
                           int64_t *hidden_size);
};
}  // namespace opt
}  // namespace mindspore
#endif
