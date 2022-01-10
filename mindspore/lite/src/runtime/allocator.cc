/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifdef MSLITE_ENABLE_SERVER_INFERENCE
#include "src/runtime/dynamic_mem_allocator.h"
#else
#include "src/runtime/inner_allocator.h"
#endif

namespace mindspore {
std::shared_ptr<Allocator> Allocator::Create() {
#ifdef MSLITE_ENABLE_SERVER_INFERENCE
  return std::make_shared<DynamicMemAllocator>();
#else
  return std::make_shared<DefaultAllocator>();
#endif
}
}  // namespace mindspore
