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

#include "include/context.h"
#include "src/runtime/allocator.h"

namespace mindspore::lite {
Context::Context() { allocator = Allocator::Create(); }

Context::~Context() = default;

Context::Context(int thread_num, std::shared_ptr<Allocator> allocator, DeviceContext device_ctx) {
  this->allocator = std::move(allocator);
  this->thread_num_ = thread_num;
  this->device_ctx_ = device_ctx;
}
}  // namespace mindspore::lite

