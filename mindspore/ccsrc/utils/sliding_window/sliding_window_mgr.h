/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_SLIDING_WINDOW_SLIDING_WINDOW_MGR_H_
#define MINDSPORE_CCSRC_UTILS_SLIDING_WINDOW_SLIDING_WINDOW_MGR_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include "ir/tensor.h"
#include "ir/value.h"
#include "include/utils/visible.h"
#include "mindspore/ccsrc/utils/sliding_window/device_space_manager.h"
#include "mindspore/ccsrc/utils/sliding_window/source_group_manager.h"

namespace mindspore {
namespace utils {
namespace sliding_window {

class COMMON_EXPORT SlidingWindowMgr {
 public:
  SlidingWindowMgr();
  ~SlidingWindowMgr() = default;
  SlidingWindowMgr(const SlidingWindowMgr &) = delete;
  SlidingWindowMgr &operator=(const SlidingWindowMgr &) = delete;

  void PreProcess(const std::string &op_name, const std::vector<ValuePtr> &inputs);
  void PostProcess(const std::string &op_name, const std::vector<ValuePtr> &inputs, const ValuePtr &output);
  void Finalize();

 private:
  void RegisterOutputs(const std::vector<tensor::TensorPtr> &outputs, uint64_t timestamp,
                       const std::unordered_set<TensorId> &view_output_ids);
  uint64_t GetCurrentTimestamp();

  std::unique_ptr<SourceGroupManager> source_group_mgr_;
  std::unique_ptr<DeviceSpaceManager> space_manager_;
};

}  // namespace sliding_window
}  // namespace utils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_SLIDING_WINDOW_SLIDING_WINDOW_MGR_H_
