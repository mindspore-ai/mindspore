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

#include "include/runtime/utils/dispatch/dispatch_env.h"
#include "utils/ms_utils.h"

namespace mindspore {
static auto dispatch_env = common::GetEnv("MS_DEV_DISABLE_AUTO_H2D");
static auto enable_with_check = dispatch_env == "1";
static auto enable_with_stack = dispatch_env == "2";
static auto enable_without_check = dispatch_env == "3";
static auto enable = enable_with_check || enable_with_stack || enable_without_check;

bool EnableDispatch() { return enable; }

bool EnableDispatchWithStack() { return enable_with_stack; }

bool EnableDispatchWithCheck() { return enable_with_check; }

bool EnableDispatchWithoutCheck() { return enable_without_check; }
}  // namespace mindspore
