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
#include "graph_kernel/common/graph_kernel_common_test_suite.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "backend/common/graph_kernel/adapter/callback_impl.h"

namespace mindspore {

void GraphKernelCommonTestSuite::SetUp() {
  if (graphkernel::Callback::Instance() == nullptr) {
    graphkernel::Callback::RegImpl(std::make_shared<graphkernel::CallbackImpl>());
  }
}
}  // namespace mindspore
