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

#include "pipeline/pynative/forward/forward_task.h"

namespace mindspore {
namespace pynative {
void ForwardTask::Run() { run_func_(op_run_info_); }

void ForwardTask::SetException(const std::exception_ptr &e) { op_run_info_->stub_output->SetException(e); }
}  // namespace pynative
}  // namespace mindspore
