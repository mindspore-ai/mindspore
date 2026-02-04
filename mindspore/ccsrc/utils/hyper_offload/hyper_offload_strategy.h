/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_STRATEGY_H_
#define MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_STRATEGY_H_

#include <memory>
#include <vector>

#include "ir/anf.h"
#include "utils/hyper_offload/hyper_offload_utils.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
class HyperOffloadStrategy {
 public:
  virtual ~HyperOffloadStrategy() = default;
  // Execute the strategy to determine offload information.
  virtual OffloadInfoList Run(const CNodePtrList &execution_order, const UserInfoList &user_info_list) = 0;
};
using HyperOffloadStrategyPtr = std::unique_ptr<HyperOffloadStrategy>;

class DistanceBaseHyperOffloadStrategy : public HyperOffloadStrategy {
 public:
  // Execute the distance-based strategy.
  OffloadInfoList Run(const CNodePtrList &execution_order, const UserInfoList &user_info_list) override;
};
using DistanceBaseHyperOffloadStrategyPtr = std::unique_ptr<DistanceBaseHyperOffloadStrategy>;

class OffloadInfoFilter {
 public:
  virtual ~OffloadInfoFilter() = default;
  // Filter the offload information list.
  virtual OffloadInfoList Filter(const OffloadInfoList &offload_info_list) = 0;
};

class OffloadInfoFilterByNumber : public OffloadInfoFilter {
 public:
  // Filter offload information based on the number limit.
  OffloadInfoList Filter(const OffloadInfoList &offload_info_list) override;
};
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_STRATEGY_H_
