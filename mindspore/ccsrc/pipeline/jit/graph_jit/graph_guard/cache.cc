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
#include "pipeline/jit/graph_jit/graph_guard/cache.h"
#include "pipeline/jit/ps/pipeline.h"

namespace mindspore {
namespace jit {
namespace graph {
bool OptOption::operator==(const OptOption &obj) const {
  if (obj.target_ == this->target_) {
    return true;
  } else {
    return false;
  }
}

OptOption::OptOption(PyCodeObject *code) { target_ = code; }

OptOption::OptOption(void *ptr) { target_ = ptr; }

std::shared_ptr<OptOption> OptOption::CreateOptionByCode(PyCodeObject *code) {
  OptOptionPtr ret(new OptOption(code));
  return ret;
}

std::shared_ptr<OptOption> OptOption::CreateOptionByPoint(void *ptr) {
  OptOptionPtr ret(new OptOption(ptr));
  return ret;
}

OptCode::OptCode() : phase_(""), cFunc_(nullptr), rFunc_(nullptr), pFunc_(NULL) {
  guard_ = std::make_shared<OptGuard>();
}

OptCode::~OptCode() {
  if (rFunc_ != nullptr) {
    rFunc_();
  }
  Py_XDECREF(pFunc_);
}

void OptCode::SetPhase(std::string phase) { phase_ = phase; }

void OptCode::SetNativeFunc(NativeFunc cFunc, ReleaseFunc rFunc) {
  cFunc_ = cFunc;
  rFunc_ = rFunc;
}

NativeFunc OptCode::GetNativeFunc() { return cFunc_; }

void OptCode::SetPythonCallable(PyObject *pFunc) {
  if (pFunc != NULL && (PyCallable_Check(pFunc) || PyCode_Check(pFunc))) {
    Py_INCREF(pFunc);
    Py_XSETREF(this->pFunc_, pFunc);
  } else {
    pFunc_ = NULL;
  }
}

PyObject *OptCode::GetPythonCallable() { return pFunc_; }

void OptCode::SetGuard(OptGuardPtr guard) { guard_ = guard; }

OptGuardPtr OptCode::GetGuard() { return guard_; }

void OptCode::SetOption(OptOptionPtr option) { option_ = option; }

OptOptionPtr OptCode::GetOption() { return option_; }

OptCodePtr OptCodeHub::AddOptTarget(OptOptionPtr option) {
  OptCodePtr ret;
  for (auto &item : codeMap_) {
    if (*(item.first.get()) == *(option.get())) {
      ret = std::make_shared<OptCode>();
      item.second.push_back(ret);
      return ret;
    }
  }
  ret = std::make_shared<OptCode>();
  codeMap_[option].push_back(ret);
  ret->SetOption(option);
  return ret;
}

OptCodeSet OptCodeHub::GetOptTarget(OptOptionPtr option) {
  for (auto &item : codeMap_) {
    if (*(item.first.get()) == *(option.get())) {
      return item.second;
    }
  }
  return {};
}

void OptCodeHub::DelOptTarget(OptOptionPtr option, OptCodePtr code) {
  for (auto &item : codeMap_) {
    if (*(item.first.get()) == *(option.get())) {
      auto it = std::find(item.second.begin(), item.second.end(), code);
      if (it != item.second.end()) {
        item.second.erase(it);
      }
      if (item.second.size() == 0) {
        codeMap_.erase(item.first);
      }
      break;
    }
  }
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
