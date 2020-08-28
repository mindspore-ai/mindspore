/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_ASCEND_PROFILING_H_
#define MINDSPORE_ASCEND_PROFILING_H_
#include <atomic>
#include <chrono>
#include <mutex>
#include <ostream>
#include <string>
#include <vector>
using std::string;

namespace mindspore {
namespace profiler {
namespace ascend {
enum EventType { kGeneral = 0, kCompiler, kExecution, kCallback };
struct Event {
  std::chrono::system_clock::time_point timestamp;
  EventType event_type;
  std::string desc;
};
class AscendProfiler {
 public:
  AscendProfiler();
  ~AscendProfiler() = default;

  static AscendProfiler &GetInstance() {
    static AscendProfiler instance;
    return instance;
  }

  void RecordEvent(EventType event_type, const char *fmt, ...);

  void Reset();

  void Dump(std::ostream &os);

 private:
  std::vector<Event> events_;
  std::atomic_int counter_;
};

}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_ASCEND_PROFILING_H_
