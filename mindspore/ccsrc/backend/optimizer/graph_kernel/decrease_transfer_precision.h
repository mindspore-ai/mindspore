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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_DECREASE_TRANSFER_PRECISION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_DECREASE_TRANSFER_PRECISION_H_

#include <string>
#include "backend/optimizer/common/optimizer.h"

namespace mindspore {
namespace opt {
class DecreaseTransferPrecision : public Pass {
 public:
  DecreaseTransferPrecision() : Pass("decrease_transfer_precision") {}
  ~DecreaseTransferPrecision() override = default;
  bool Run(const FuncGraphPtr &func_graph);

 private:
  bool Process_Father(const FuncGraphPtr &func_graph, const AnfNodePtr &node, bool is_tuple_out = false,
                      size_t index = 0);
  bool Process_Son(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t index);
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_DECREASE_TRANSFER_PRECISION_H_
