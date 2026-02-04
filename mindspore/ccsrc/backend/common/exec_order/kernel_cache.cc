/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "include/backend/common/exec_order/kernel_cache.h"
#include <utility>
#include <memory>
#include <string>
#include <vector>

namespace mindspore {
namespace runtime {
void KernelCache::SwapBuffers(int step) {
  std::lock_guard<std::mutex> lock(mutex_);
  step_buffers_[step] = std::move(current_buffer_);
  current_buffer_.clear();
  current_buffer_.reserve(100000);
}

std::vector<std::any> KernelCache::GetBuffers(int step) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (step == -1) {
    return current_buffer_;
  }
  auto it = step_buffers_.find(step);
  if (it != step_buffers_.end()) {
    return it->second;
  }
  return {};
}

void KernelCache::AddPyboostKernel(const std::string &prim_name, const std::string &group,
                                   const std::string &input_shape, const std::string &output_shape, int64_t rank) {
  auto kernel = std::make_shared<CommPyboostKernel>(prim_name, group, input_shape, output_shape, rank);
  current_buffer_.emplace_back(kernel);
}
}  // namespace runtime
}  // namespace mindspore
