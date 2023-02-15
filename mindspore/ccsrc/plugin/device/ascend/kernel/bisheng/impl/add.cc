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

#include "plugin/device/ascend/kernel/bisheng/impl/add.h"
#include <type_traits>
#include <sycl/sycl.hpp>
#include "runtime/rt.h"

namespace mindspore::kernel::bisheng {
template <typename T>
void Add(void *x1, void *x2, void *y, uint64_t size, void *stream) {
  T *input0 = static_cast<T *>(x1);
  T *input1 = static_cast<T *>(x2);
  T *output = static_cast<T *>(y);
  CCEcontext rt_context = nullptr;
  rtCtxGetCurrent(&rt_context);
  sycl::context sycl_context = sycl::make_context<sycl::backend::cce>(rt_context);
  sycl::queue queue = sycl::make_queue<sycl::backend::cce>(stream, sycl_context);
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<T>(size / sizeof(T), [=](sycl::id<1> idx) { output[idx] = input0[idx] + input1[idx]; });
  });
}

template BISHENG_LIB_EXPORT void Add<uint8_t>(void *x1, void *x2, void *y, size_t size, void *stream);
template BISHENG_LIB_EXPORT void Add<int8_t>(void *x1, void *x2, void *y, size_t size, void *stream);
template BISHENG_LIB_EXPORT void Add<uint16_t>(void *x1, void *x2, void *y, size_t size, void *stream);
template BISHENG_LIB_EXPORT void Add<int16_t>(void *x1, void *x2, void *y, size_t size, void *stream);
template BISHENG_LIB_EXPORT void Add<uint32_t>(void *x1, void *x2, void *y, size_t size, void *stream);
template BISHENG_LIB_EXPORT void Add<int32_t>(void *x1, void *x2, void *y, size_t size, void *stream);
template BISHENG_LIB_EXPORT void Add<uint64_t>(void *x1, void *x2, void *y, size_t size, void *stream);
template BISHENG_LIB_EXPORT void Add<int64_t>(void *x1, void *x2, void *y, size_t size, void *stream);
template BISHENG_LIB_EXPORT void Add<sycl::half>(void *x1, void *x2, void *y, size_t size, void *stream);
template BISHENG_LIB_EXPORT void Add<float>(void *x1, void *x2, void *y, size_t size, void *stream);
}  // namespace mindspore::kernel::bisheng
