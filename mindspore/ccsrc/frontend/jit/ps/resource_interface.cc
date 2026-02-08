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

#include "include/frontend/jit/ps/resource_interface.h"

#include "frontend/jit/ps/resource.h"
#include "frontend/jit/ps/pass_config.h"
#include "frontend/expander/utils.h"
#include "frontend/jit/ps/action.h"
#include "frontend/jit/ps/static_analysis/prim.h"
#include "frontend/jit/ps/static_analysis/async_eval_result.h"

namespace mindspore {
namespace pipeline {
void ClearAttrAndMethodMap() {
  GetMethodMap().clear();
  GetAttrMap().clear();
}

void ClearPassConfigure() { opt::PassConfigure::Instance().Clear(); }

void ClearOpPrimPyRegister() { expander::OpPrimPyRegister::GetInstance().Clear(); }

void CleanParserResource() { parse::Parser::CleanParserResource(); }

void ClearAnalysisSchedule() { abstract::AnalysisSchedule::GetInstance().Stop(); }

void ClearAnalysisResultCacheMgr() { abstract::AnalysisResultCacheMgr::GetInstance().Clear(); }

void ClearPrimitiveEvaluatorMap() { abstract::ClearPrimEvaluatorMap(); }

void ClearActionConfigure() { ActionConfigure::Instance().Clear(); }
}  // namespace pipeline
}  // namespace mindspore
