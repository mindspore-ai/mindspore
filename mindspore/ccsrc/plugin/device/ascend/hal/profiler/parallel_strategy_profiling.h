/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_PARALLEL_STRATEGY_PROFILING_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_PARALLEL_STRATEGY_PROFILING_H

#include <string>
#include <memory>

#include "ir/func_graph.h"
#include "base/base.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace irpb {
class ProfilingParallel;
}
namespace profiler {
namespace ascend {
class ParallelStrategy {
 public:
  static std::shared_ptr<ParallelStrategy> &GetInstance();
  ParallelStrategy() = default;
  ~ParallelStrategy() {}
  void DumpProfileParallelStrategy(const FuncGraphPtr &func_graph);
  void SaveParallelStrategyToFile();
  std::string GetParallelStrategyForReport();

 private:
  std::shared_ptr<irpb::ProfilingParallel> GetProfilingParallel();
  bool IsProfilingParallelStrategyEnabled();
  bool StringToInt(std::string *str, int32_t *value);

  static std::shared_ptr<ParallelStrategy> parallel_strategy_inst_;
  bool has_save_parallel_strategy_ = false;
  bool has_got_parallel_strategy_data_ = false;
  std::shared_ptr<irpb::ProfilingParallel> cache_profiling_parallel_pb_;
  std::string graph_proto_str_;
};
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_PARALLEL_STRATEGY_PROFILING_H
