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
#ifndef MINDSPORE_PI_JIT_PERF_H
#define MINDSPORE_PI_JIT_PERF_H

#include <memory>
#include <map>
#include <chrono>
#include <functional>
#include <limits>

namespace mindspore {
namespace pijit {

/// \brief performance statistics
class PerfStatistics {
 public:
  virtual double GetAverageDuration() = 0;
  virtual double GetMaxDuration() = 0;
  virtual double GetMinDuration() = 0;
  virtual int GetTotalCount() = 0;
};
using PerfStatisticsPtr = std::shared_ptr<PerfStatistics>;

/// \brief performance for optimized code
class OptPerf {
 public:
  enum PerfKind {
    kPerfPyNative = 0,
    kPerfGraph,
    kPerfCount,
  };
  OptPerf();
  virtual ~OptPerf() = default;
  void AddDuration(double duration);
  PerfStatisticsPtr GetStatistics();

 protected:
  PerfStatisticsPtr stat_;
};
using OptPerfPtr = std::shared_ptr<OptPerf>;

template <typename Ret, typename... Args>
Ret CallFunction(OptPerfPtr perf, std::function<Ret(Args...)> func, Args... args) {
  constexpr auto kMicroExecUnit = 1000000;
  auto start_time = std::chrono::steady_clock::now();
  Ret res = func(args...);
  std::chrono::duration<double, std::ratio<1, kMicroExecUnit>> duration = std::chrono::steady_clock::now() - start_time;
  perf->AddDuration(duration.count());
  return res;
}
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_PERF_H
