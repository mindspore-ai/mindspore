/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "utils/log_adapter.h"
#include "dnnl.hpp"

namespace mindspore {
namespace kernel {
void MKLKernelEngine::Execute(const std::shared_ptr<dnnl::primitive> &primitive,
                              const std::unordered_map<int, dnnl::memory> &arguments) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_LOG(DEBUG) << "begin to invoke primitive::execute";
  primitive->execute(stream_, arguments);
  MS_LOG(DEBUG) << "end to invoke primitive::execute";

  MS_LOG(DEBUG) << "begin to invoke dnnl::stream::wait";
  (void)stream_.wait();
  MS_LOG(DEBUG) << "end to invoke dnnl::stream::wait";
}

dnnl::memory MKLKernelEngine::CreateMemory(const dnnl::memory::desc &mem_desc, bool alloc) {
  if (alloc) {
    MS_LOG(DEBUG) << "begin to invoke constructor of dnnl::memory(const desc&, const engine&, void*)";
    auto res = dnnl::memory(mem_desc, engine_);
    MS_LOG(DEBUG) << "end to invoke constructor of dnnl::memory(const desc&, const engine&, void*)";
    return res;
  } else {
    MS_LOG(DEBUG) << "begin to invoke constructor of dnnl::memory(const desc&, const engine&)";
    auto res = dnnl::memory(mem_desc, engine_, nullptr);
    MS_LOG(DEBUG) << "end to invoke constructor of dnnl::memory(const desc&, const engine&)";
    return res;
  }
}

void MKLKernelEngine::Reorder(dnnl::memory *src_mem, dnnl::memory *dst_mem) {
  MS_LOG(DEBUG) << "begin to invoke constructor of dnnl::reorder";
  auto desc = dnnl::reorder(*src_mem, *dst_mem);
  MS_LOG(DEBUG) << "end to invoke constructor of dnnl::reorder";
  MS_LOG(DEBUG) << "end to invoke primitive::execute";
  desc.execute(stream_, *src_mem, *dst_mem);
  MS_LOG(DEBUG) << "end to invoke primitive::execute";
}
}  // namespace kernel
}  // namespace mindspore
