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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_AUTO_MONAD_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_AUTO_MONAD_H_

#include <string>
#include <memory>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "base/effect_info.h"

namespace mindspore::pipeline {

// Run auto-monad, handle side-effects, called from frontend pipeline.
bool AutoMonad(const FuncGraphPtr &func_graph);

// Run auto-monad after grad or Renormalize, handle side-effects, called from frontend opt pass.
bool ReAutoMonad(const FuncGraphPtr &func_graph);
}  // namespace mindspore::pipeline

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PARSE_AUTO_MONAD_H_
