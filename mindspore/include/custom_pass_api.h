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
#ifndef MINDSPORE_INCLUDE_CUSTOM_PASS_API_H_
#define MINDSPORE_INCLUDE_CUSTOM_PASS_API_H_

// MindSpore core headfiles
#include "mindspore/core/include/ir/anf.h"
#include "mindspore/core/include/ir/primitive.h"
#include "mindspore/core/include/ir/dtype/type_id.h"
#include "mindspore/core/include/utils/log_adapter.h"
#include "mindspore/core/include/utils/ms_context.h"

// pass headfiles
#include "mindspore/ccsrc/include/backend/common/pass_manager/pass.h"
#include "mindspore/ccsrc/include/backend/common/pass_manager/pattern_to_pattern.h"

// custom pass plugin headfile
#include "mindspore/ccsrc/include/backend/common/custom_pass/custom_pass_plugin.h"

// util headfiles
#include "mindspore/ccsrc/include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "mindspore/ccsrc/include/backend/common/pass_manager/helper.h"
#include "mindspore/ccsrc/include/utils/anfalgo.h"
#include "mindspore/ccsrc/include/utils/utils.h"

#endif  // MINDSPORE_INCLUDE_CUSTOM_PASS_API_H_
