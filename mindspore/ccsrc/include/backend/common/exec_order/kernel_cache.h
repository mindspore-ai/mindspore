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

#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_EXEC_ORDER_KERNEL_CACHE_H_
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_EXEC_ORDER_KERNEL_CACHE_H_

#include <any>
#include <vector>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include "ir/anf.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace runtime {
struct BACKEND_COMMON_EXPORT CommPyboostKernel {
  std::string primitive;
  std::string group;
  std::string input_shape;
  std::string output_shape;
  int64_t rank;

  CommPyboostKernel(const std::string &prim_name, const std::string &group, const std::string &input_shape,
                    const std::string &output_shape, int64_t rank)
      : primitive(prim_name), group(group), input_shape(input_shape), output_shape(output_shape), rank(rank) {}
};
using CommPyboostKernelPtr = std::shared_ptr<CommPyboostKernel>;

class BACKEND_COMMON_EXPORT KernelCache {
 public:
  static KernelCache &GetInstance() {
    static KernelCache instance;
    return instance;
  }

  inline void Add(const CNodePtr &kernel) { current_buffer_.emplace_back(kernel); }

  void AddPyboostKernel(const std::string &prim_name, const std::string &group, const std::string &input_shape,
                        const std::string &output_shape, int64_t rank);

  void SwapBuffers(int step);

  void ClearBuffers() { current_buffer_.clear(); }

  std::vector<std::any> GetBuffers(int step);

  bool need_add{false};

 private:
  KernelCache() = default;
  ~KernelCache() = default;
  KernelCache(const KernelCache &) = delete;
  KernelCache &operator=(const KernelCache &) = delete;

  std::vector<std::any> current_buffer_;
  std::unordered_map<int, std::vector<std::any>> step_buffers_;
  std::mutex mutex_;
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_EXEC_ORDER_KERNEL_CACHE_H_
