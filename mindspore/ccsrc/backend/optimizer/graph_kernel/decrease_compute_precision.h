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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_DECREASE_COMPUTE_PRECISION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_DECREASE_COMPUTE_PRECISION_H_

#include <vector>
#include <string>
#include "backend/optimizer/common/optimizer.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace opt {
class DecreaseComputePrecision : public Pass {
 public:
  explicit DecreaseComputePrecision(const std::vector<PrimitivePtr> &black_list = {})
      : Pass("decrease_compute_precision"), black_list_(black_list) {}
  ~DecreaseComputePrecision() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  bool Process(const FuncGraphPtr &func_graph);
  AnfNodePtr CreateCast(const AnfNodePtr &input, const TypePtr &dst_type, const std::string &format) const;
  std::vector<PrimitivePtr> black_list_;
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_DECREASE_COMPUTE_PRECISION_H_
