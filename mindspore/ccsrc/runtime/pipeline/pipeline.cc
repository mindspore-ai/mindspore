/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "runtime/pipeline/pipeline.h"
#include <memory>
#include "runtime/pipeline/async_rqueue.h"

namespace mindspore {
namespace runtime {
Pipeline &Pipeline::Get() {
  static Pipeline instance;
  return instance;
}

Pipeline::Pipeline()
    : frontend_stage_(
        std::make_shared<runtime::AsyncRQueue>("frontend_queue", runtime::kThreadWaitLevel::kLevelFrontend)),
      backend_stage_(std::make_shared<runtime::AsyncRQueue>("backend_device", kThreadWaitLevel::kLevelDevice)) {}
}  // namespace runtime
}  // namespace mindspore
