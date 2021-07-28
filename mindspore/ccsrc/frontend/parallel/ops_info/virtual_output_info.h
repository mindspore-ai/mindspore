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

#ifndef PARALLEL_OPS_INFO_OUTPUT_INFO_H_
#define PARALLEL_OPS_INFO_OUTPUT_INFO_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/ops_info/virtual_dataset_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class VirtualOutputInfo : public VirtualDatasetInfo {
 public:
  VirtualOutputInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                    const PrimitiveAttrs &attrs)
      : VirtualDatasetInfo(name, inputs_shape, outputs_shape, attrs) {}
  ~VirtualOutputInfo() override = default;
  Status GenerateStrategies(int64_t stage_id) override;

 protected:
  Status CheckStrategy(const StrategyPtr &strategy) override;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // PARALLEL_OPS_INFO_VIRTUAL_OUTPUT_INFO_H_
