/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/fusion/quant_dtype_cast_fusion.h"
#include <memory>
#include "tools/optimizer/common/gllo_utils.h"
namespace mindspore::opt {
namespace {
constexpr size_t kActivationInputsLength = 2;
}
const BaseRef QuantDtypeCastFusion::DefinePattern() const {
  auto quant_var = std::make_shared<CondVar>(IsQuantNode);
  auto input_var = std::make_shared<Var>();
  return VectorRef({quant_var, input_var});
}

const AnfNodePtr QuantDtypeCastFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  MS_LOG(DEBUG) << "quant dtype cast fusion pass process";
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
    return nullptr;
  }
  auto act_node = node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(act_node) != lite::RET_OK ||
      CheckInputSize(act_node, kActivationInputsLength) != lite::RET_OK) {
    return nullptr;
  }
  AnfNodePtr pre_node = act_node->input(1);
  if (CheckIfAnfNodeIsNull(pre_node) != lite::RET_OK) {
    return nullptr;
  }
  return pre_node;
}
}  // namespace mindspore::opt
