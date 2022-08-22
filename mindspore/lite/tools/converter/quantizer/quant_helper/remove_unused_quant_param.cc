/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/quantizer/quant_helper/remove_unused_quant_param.h"

namespace mindspore::lite::quant {
int RemoveQuantParam::Remove() {
  CHECK_NULL_RETURN(func_graph_);
  auto nodes = func_graph_->GetOrderedCnodes();
  for (auto const &cnode : nodes) {
    auto quant_holder = GetCNodeQuantHolder(cnode);
    if (quant_holder == nullptr) {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << "quant holder is nullptr.";
      return RET_ERROR;
    }
    quant_holder->ClearQuantParams();
  }
  return RET_OK;
}
}  // namespace mindspore::lite::quant
