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
#include "pipeline/jit/pi/graph_guard/cache.h"
#include <algorithm>
#include "pipeline/jit/ps/pipeline.h"

namespace mindspore {
namespace pijit {
OptFunc::OptFunc(NativeFunc cFunc, ReleaseFunc rFunc) : cFunc_(cFunc), rFunc_(rFunc) {}

OptFunc::~OptFunc() {
  if (rFunc_ != nullptr) {
    rFunc_();
  }
}

NativeFunc OptFunc::GetFunc() { return cFunc_; }

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

OptCode::OptCode() : phase_(""), compiled_code_(), call_count_(0) {
  guard_ = std::make_shared<OptGuard>();
  graph_perf_ = std::make_shared<OptPerf>();
  pynative_perf_ = std::make_shared<OptPerf>();
  compiled_func_ = nullptr;
}

OptCode::~OptCode() {}

void OptCode::SetNativeFunc(const std::string &phase, NativeFunc cFunc, ReleaseFunc rFunc) {
  phase_ = phase;
  compiled_func_ = std::make_shared<OptFunc>(cFunc, rFunc);
}

NativeFunc OptCode::GetNativeFunc() const {
  if (compiled_func_ != nullptr) {
    return compiled_func_->GetFunc();
  } else {
    return nullptr;
  }
}

std::string OptCode::GetPhase() const { return phase_; }

void OptCode::SetPythonCode(const py::object &code) {
  MS_EXCEPTION_IF_CHECK_FAIL(code.ptr() != nullptr && PyCode_Check(code.ptr()) && Py_REFCNT(code.ptr()) == 1,
                             "code handler must be only one");
  compiled_code_ = code;
}

PyCodeObject *OptCode::GetPythonCode() const { return reinterpret_cast<PyCodeObject *>(compiled_code_.ptr()); }

void OptCode::SetGuard(OptGuardPtr guard) { guard_ = guard; }

OptGuardPtr OptCode::GetGuard() { return guard_; }

void OptCode::SetOption(OptOptionPtr option) { option_ = option; }

OptOptionPtr OptCode::GetOption() { return option_; }

OptPerfPtr OptCode::GetPerf(OptPerf::PerfKind kind) {
  switch (kind) {
    case OptPerf::PerfKind::kPerfGraph:
      return graph_perf_;
    case OptPerf::PerfKind::kPerfPyNative:
      return pynative_perf_;
    default:
      return nullptr;
  }
}

void OptCode::Copy(OptCodePtr dst) {
  dst->graph_perf_ = graph_perf_;
  dst->pynative_perf_ = pynative_perf_;
  dst->phase_ = phase_;
  dst->compiled_func_ = compiled_func_;
}

void OptCode::Inc() { call_count_++; }

uint64_t OptCode::Count() { return call_count_; }

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

void OptCodeHub::UpdateOptTarget(OptOptionPtr option, OptCodePtr code) {
  for (auto &item : codeMap_) {
    if (*(item.first.get()) == *(option.get())) {
      auto it = std::find(item.second.begin(), item.second.end(), code);
      if (it != item.second.end()) {
        item.second.erase(it);
        item.second.push_back(code);
      }
      break;
    }
  }
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

void OptCodeHub::DelOptTarget(OptCodePtr code) {
  for (auto &item : codeMap_) {
    auto it = std::find(item.second.begin(), item.second.end(), code);
    if (it != item.second.end()) {
      item.second.erase(it);
      if (item.second.size() == 0) {
        codeMap_.erase(item.first);
      }
      break;
    }
  }
}

std::vector<OptCodeSet> OptCodeHub::GetAllOptTarget() {
  std::vector<OptCodeSet> ret;
  std::transform(codeMap_.begin(), codeMap_.end(), std::back_inserter(ret),
                 [](const auto &item) { return item.second; });
  return ret;
}

using OptCodeWPtr = std::weak_ptr<OptCode>;
using OptCodeWSet = std::vector<OptCodeWPtr>;
static std::map<std::string, OptCodeWSet> code_set;

void OptCodeHub::Register(std::string key, OptCodePtr code) { code_set[key].emplace_back(code); }
OptCodePtr OptCodeHub::Filter(std::string key, OptCodeFilterFunc filter) {
  if (code_set.find(key) != code_set.end()) {
    OptCodeWSet &codes = code_set[key];
    for (size_t idx = 0; idx < codes.size();) {
      OptCodePtr ptr = codes[idx].lock();
      if (ptr != nullptr) {
        if (filter(ptr)) {
          return ptr;
        }
        idx++;
      } else {
        codes.erase(codes.begin() + idx);
      }
    }
  }
  return nullptr;
}
}  // namespace pijit
}  // namespace mindspore
