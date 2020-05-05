/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_MKL_KERNEL_ENGINE_H_
#define MINDSPORE_MKL_KERNEL_ENGINE_H_

#include <unordered_map>
#include <vector>
#include <memory>
#include "dnnl.hpp"
#include "common/utils.h"

namespace mindspore {
namespace kernel {
class MKLKernelEngine {
 public:
  static MKLKernelEngine &Get() {
    static MKLKernelEngine instance;
    return instance;
  }
  DISABLE_COPY_AND_ASSIGN(MKLKernelEngine)

  const dnnl::engine &engine() const { return engine_; }

  dnnl::memory CreateMemory(const dnnl::memory::desc &mem_desc, bool alloc = false);

  void Execute(const std::shared_ptr<dnnl::primitive> &primitive,
               const std::unordered_map<int, dnnl::memory> &arguments);

 private:
  MKLKernelEngine() : engine_(dnnl::engine::kind::cpu, 0), stream_(engine_) {}
  ~MKLKernelEngine() = default;
  dnnl::engine engine_;
  dnnl::stream stream_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_MKL_KERNEL_ENGINE_H_
