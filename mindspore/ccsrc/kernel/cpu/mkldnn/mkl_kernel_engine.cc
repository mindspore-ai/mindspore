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
#include "kernel/cpu/mkldnn/mkl_kernel_engine.h"
#include "utils/log_adapter.h"
#include "dnnl.hpp"

namespace mindspore {
namespace kernel {
void MKLKernelEngine::Execute(const std::shared_ptr<dnnl::primitive> &primitive,
                              const std::unordered_map<int, dnnl::memory> &arguments) {
  MS_EXCEPTION_IF_NULL(primitive);
  primitive->execute(stream_, arguments);
  (void)stream_.wait();
}

dnnl::memory MKLKernelEngine::CreateMemory(const dnnl::memory::desc &mem_desc, bool alloc) {
  if (alloc) {
    return dnnl::memory(mem_desc, engine_);
  } else {
    return dnnl::memory(mem_desc, engine_, nullptr);
  }
}
void MKLKernelEngine::Reorder(dnnl::memory *src_mem, dnnl::memory *dst_mem) {
  dnnl::reorder(*src_mem, *dst_mem).execute(stream_, *src_mem, *dst_mem);
}
}  // namespace kernel
}  // namespace mindspore
