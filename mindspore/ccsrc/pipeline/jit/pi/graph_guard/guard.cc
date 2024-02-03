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
#include "pipeline/jit/pi/graph_guard/guard.h"
#include <chrono>
#include <regex>
#include "pybind11/pybind11.h"
#include "pybind_api/ir/cell_py.h"
#include "pybind_api/ir/primitive_py.h"
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/pi/utils/utils.h"
#include "pipeline/jit/pi/graph_guard/strategy.h"

namespace mindspore {
namespace pijit {
const char kSpecializeScalar[] = "specialize_scalar";
const char kSpecializeContainer[] = "specialize_container";
const char kSpecializeTensor[] = "specialize_tensor";

static std::map<std::string, bool> g_mapDefaultConfig = {
  {kSpecializeScalar, false},
  {kSpecializeContainer, false},
  {kSpecializeTensor, false},
};

static GuardItemPtr GuardOnGDeduce(TracePtr var, PyObject *obj, const std::map<std::string, bool> &config);
static GuardItemPtr GuardOnScalar(TracePtr var, const std::map<std::string, bool> &config);
static GuardItemPtr GuardOnContainer(TracePtr var, const std::map<std::string, bool> &config);
static GuardItemPtr GuardOnLiteral(TracePtr var, const std::map<std::string, bool> &config);
static GuardItemPtr GuardOnTensor(TracePtr var, const std::map<std::string, bool> &config);
static GuardItemPtr GuardOnMutableOrConstObj(TracePtr var);
static GuardItemPtr GuardOnDynamicLenContainer(TracePtr var);

static bool CheckLiteral(PyObject *obj) {
  if (obj == nullptr) {
    return false;
  }

  ReprRecursionScope scope(obj);
  if (scope.ReEnterOrError()) {
    return scope.ReEnter();
  }
  if (CheckScalar(obj)) {
    return true;
  } else if (PyList_Check(obj)) {
    for (Py_ssize_t i = 0; i < PyList_Size(obj); ++i) {
      PyObject *item = PyList_GetItem(obj, i);
      if (!CheckLiteral(item)) {
        return false;
      }
    }
    return true;
  } else if (PyTuple_Check(obj)) {
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(obj); ++i) {
      PyObject *item = PyTuple_GET_ITEM(obj, i);
      if (!CheckLiteral(item)) {
        return false;
      }
    }
    return true;
  } else if (PySet_Check(obj) || PyFrozenSet_Check(obj)) {
    Py_ssize_t pos = 0;
    PyObject *item;
    Py_hash_t hash;
    while (_PySet_NextEntry(obj, &pos, &item, &hash)) {
      if (!CheckLiteral(item)) {
        return false;
      }
    }
    return true;
  } else if (PyDict_Check(obj)) {
    Py_ssize_t pos = 0;
    PyObject *key, *val;
    while (PyDict_Next(obj, &pos, &key, &val)) {
      if (!CheckLiteral(key) || !CheckLiteral(val)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool CheckOwnerIsCell(TracePtr var) {
  if (py::isinstance<mindspore::Cell>(var->GetObject())) {
    return true;
  } else if (var->GetOrigin() != NULL) {
    return CheckOwnerIsCell(var);
  } else {
    return false;
  }
}

class OptGuardPerfImpl : public OptGuardPerf {
 public:
  virtual void GetGuardPerfInfo(std::map<std::string, std::pair<size_t, size_t>> *guard_info,
                                std::map<std::string, std::pair<size_t, size_t>> *item_info) const;
  OptGuardPerfImpl() = default;
  virtual ~OptGuardPerfImpl() = default;
  virtual void LogGuardPerfStart();
  virtual void LogGuardPerfEnd(GuardItem *item);
  virtual void LogTracePerfStart();
  virtual void LogTracePerfEnd(Trace *trace);

 protected:
  std::chrono::steady_clock::time_point guard_start_;
  std::chrono::steady_clock::time_point item_start_;
  std::map<std::string, std::pair<size_t, size_t>> guard_info_;
  std::map<std::string, std::pair<size_t, size_t>> item_info_;
};

static OptGuardPerfImpl g_guard_perf;
OptGuardPerf *OptGuardPerf::GetGuardPerf() { return &g_guard_perf; }

void OptGuardPerfImpl::GetGuardPerfInfo(std::map<std::string, std::pair<size_t, size_t>> *guard_info,
                                        std::map<std::string, std::pair<size_t, size_t>> *item_info) const {
  if (guard_info != nullptr) {
    guard_info->clear();
    guard_info->insert(guard_info_.begin(), guard_info_.end());
  }
  if (item_info != nullptr) {
    item_info->clear();
    item_info->insert(item_info_.begin(), item_info_.end());
  }
}

void OptGuardPerfImpl::LogGuardPerfStart() { guard_start_ = std::chrono::steady_clock::now(); }

void OptGuardPerfImpl::LogGuardPerfEnd(GuardItem *item) {
  auto duration =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - guard_start_);
  size_t dur = (size_t)(duration.count());
  size_t inc = 1;
  auto info = item->ToString();
  info = std::regex_replace(info, std::regex("(\n)"), "");
  if (guard_info_.find(info) != guard_info_.end()) {
    guard_info_[info].first += inc;
    guard_info_[info].second += dur;
  } else {
    guard_info_[info] = std::make_pair(inc, dur);
  }
}

void OptGuardPerfImpl::LogTracePerfStart() { item_start_ = std::chrono::steady_clock::now(); }

void OptGuardPerfImpl::LogTracePerfEnd(Trace *trace) {
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - item_start_);
  size_t dur = (size_t)(duration.count());
  size_t inc = 1;
  auto info = trace->ToString(false);
  info = std::regex_replace(info, std::regex("(\n)"), "");
  if (item_info_.find(info) != item_info_.end()) {
    item_info_[info].first += inc;
    item_info_[info].second += dur;
  } else {
    item_info_[info] = std::make_pair(inc, dur);
  }
}

OptGuard::OptGuard() { config_ = g_mapDefaultConfig; }

OptGuard::OptGuard(const std::map<std::string, bool> &cfg) { UpdateConfig(cfg); }

void OptGuard::UpdateGuardList(GuardItemPtr item) {
  // reorder list to speed up check on next run
  for (size_t i = 0; i < guardList_.size(); ++i) {
    if (guardList_[i] == item) {
      guardList_.erase(guardList_.begin() + i);
      guardList_.insert(guardList_.begin(), item);
    }
  }
}

bool OptGuard::Check(const PyFrameObject *frame, bool print, std::map<size_t, PyObject *> *cache,
                     std::map<size_t, bool> *success, std::map<size_t, bool> *fail, bool perf) {
  // filter failure case
  if (fail != nullptr) {
    for (auto item : guardMap_) {
      if (fail->find(item.first) != fail->end()) {
        return false;
      }
    }
  }
  std::vector<GuardItemPtr> list;
  list.reserve(guardList_.size());
  // filter success case
  if (success != nullptr) {
    for (auto item : guardList_) {
      if (success->find(item->Info().Id()) == success->end()) {
        list.push_back(item);
      }
    }
  }
  list = OptStrategy::MakeGuardItemListStrategyByFrame(frame, list);
  for (size_t i = 0; i < list.size(); ++i) {
    GuardItemPtr item = list[i];
    if (perf) {
      g_guard_perf.LogGuardPerfStart();
    }
    bool result = item->Check(frame, cache, perf);
    if (perf) {
      g_guard_perf.LogGuardPerfEnd(item.get());
    }
    if (!result) {
      UpdateGuardList(item);
      if (fail != nullptr) {
        fail->operator[](item->Info().Id()) = false;
      }
      if (print) {
        auto trace = item->GetTrace();
        auto obj = GetObjectFromTrace(frame, trace);
        GRAPH_JIT_LOG_F("Guard check fail: %s v.s. %s\n", item->ToString().c_str(),
                        std::string(py::str(py::cast<py::object>(obj))).c_str());
        Py_XDECREF(obj);
      } else {
        MS_LOG(DEBUG) << "Guard check fail:" << item->ToString();
      }
      return false;
    } else if (success != nullptr) {
      success->operator[](item->Info().Id()) = true;
    }
  }
  return true;
}

bool OptGuard::GuardOn(TracePtr var, GuardLevel tp, bool needSpecialize, int recurseDepth) {
  // Now we have TypeGuard IdGuard NameGuard AttrGuard EqGuard, let's add guard to guardlist based on type
  PyObject *obj = var->GetObject();
  GuardItemPtr item = nullptr;
  if (obj != nullptr) {
    py::object py_obj = py::reinterpret_borrow<py::object>(obj);
    if (IsStubTensor(py_obj)) {
      py_obj = python_adapter::CallPyObjMethod(py_obj, "stub_sync");
      obj = py_obj.ptr();
    }
    if (tp == GuardLevel::GDeduce) {
      item = GuardOnGDeduce(var, obj, config_);
    } else if (tp == GuardLevel::GId) {
      item = GuardId(var);
    } else if (tp == GuardLevel::GType) {
      item = GuardType(var);
    } else if (tp == GuardLevel::GAttr) {
      item = GuardAttr(var);
    } else if (tp == GuardLevel::GEqual) {
      item = GuardEqual(var, needSpecialize, recurseDepth);
    }
  } else {
    // Check obj == None
    item = GuardEqual(var, 0);
  }
  if (item != nullptr) {
    size_t szItem = item->Info().Id();
    if (guardMap_.find(szItem) == guardMap_.end()) {
      guardList_.push_back(item);
      guardMap_[szItem] = item;
    }
    return true;
  } else {
    return false;
  }
}

const InfoPack &OptGuard::Info() {
  if (info_ == nullptr) {
    InfoPack info;
    info.Begin();
    for (auto &item : guardList_) {
      info << item->Info();
    }
    info.End();
    info_ = std::make_shared<InfoPack>(info);
    info_->Update();
  }
  return *info_;
}

static GuardItemPtr GuardOnGDeduce(TracePtr var, PyObject *obj, const std::map<std::string, bool> &config) {
  GuardItemPtr item = nullptr;
  if (CheckLiteral(obj)) {
    item = GuardOnLiteral(var, config);
  } else if (PyFrozenSet_Check(obj)) {
    item = GuardId(var);
  } else if (PyFunction_Check(obj) || PyMethod_Check(obj) || PyInstanceMethod_Check(obj)) {
    item = GuardEqual(var, false, 0);
  } else if (PyType_Check(obj)) {
    item = GuardEqual(var, false, 0);
  } else if (CheckContainer(obj)) {
    // due to the failure of CheckLiteral, it need check size and element type
    item = GuardOnContainer(var, config);
  } else if (PySlice_Check(obj)) {
    item = GuardType(var);
  } else if (py::isinstance<py::array>(obj)) {
    item = GuardId(var);
  } else if (py::isinstance<mindspore::Type>(obj)) {
    item = GuardEqual(var, true, INT_MAX);
  } else if (IsTensorPyObject(obj)) {
    item = GuardOnTensor(var, config);
  } else if (py::isinstance<mindspore::PrimitivePyAdapter>(obj)) {
    if (CheckOwnerIsCell(var)) {
      item = GuardEqual(var, true, INT_MAX);
    } else {
      item = GuardRepr(var);
    }
  } else if (py::isinstance<mindspore::Cell>(obj)) {
    item = GuardRepr(var);
  } else if (py::isinstance<mindspore::ParamInfo>(obj)) {
    item = GuardEqual(var, true, INT_MAX);
  } else {
    item = GuardType(var);
  }
  return item;
}

static GuardItemPtr GuardOnScalar(TracePtr var, const std::map<std::string, bool> &config) {
  GuardItemPtr item = GuardOnMutableOrConstObj(var);
  if (item != nullptr) {
    return item;
  }
  bool need_specialize = false;
  auto cfg = config.find(kSpecializeScalar);
  if (cfg != config.end()) {
    need_specialize = cfg->second;
  }
  // need take dynamic symbolic into account
  if (need_specialize) {
    if ((var->GetOriginType() == TraceType::Global || var->GetOriginType() == TraceType::BuiltIn) ||
        var->GetOriginType() == TraceType::Param || var->GetTraceType() == TraceType::Item ||
        var->GetTraceType() == TraceType::Attr) {
      item = GuardEqual(var, true, INT_MAX);
    } else {
      item = GuardType(var);
    }
  } else {
    item = GuardEqual(var, false, 0);
  }
  return item;
}

static GuardItemPtr GuardOnContainer(TracePtr var, const std::map<std::string, bool> &config) {
  GuardItemPtr item = GuardOnDynamicLenContainer(var);
  if (item != nullptr) {
    return item;
  } else {
    item = GuardOnMutableOrConstObj(var);
  }
  if (item != nullptr) {
    return item;
  }
  bool need_specialize = false;
  auto cfg = config.find(kSpecializeContainer);
  if (cfg != config.end()) {
    need_specialize = cfg->second;
  }
  if (need_specialize) {
    item = GuardEqual(var, true, INT_MAX);
  } else {
    item = GuardEqual(var, false, 0);
  }
  return item;
}

static GuardItemPtr GuardOnLiteral(TracePtr var, const std::map<std::string, bool> &config) {
  GuardItemPtr item = nullptr;
  PyObject *obj = var->GetObject();
  if (CheckScalar(obj)) {
    return GuardOnScalar(var, config);
  } else if (CheckContainer(obj)) {
    return GuardOnContainer(var, config);
  } else {
    item = GuardOnMutableOrConstObj(var);
    if (item == nullptr) {
      item = GuardEqual(var, false, 0);
    }
  }
  return item;
}

static GuardItemPtr GuardOnTensor(TracePtr var, const std::map<std::string, bool> &config) {
  GuardItemPtr item = nullptr;
  bool need_specialize = false;
  auto cfg = config.find(kSpecializeTensor);
  if (cfg != config.end()) {
    need_specialize = cfg->second;
  }
  item = GuardOnMutableOrConstObj(var);
  if (item != nullptr) {
    return item;
  }
  if (CheckOwnerIsCell(var)) {
    if (var->GetOriginType() == TraceType::Const) {
      item = GuardId(var);
    } else {
      item = GuardEqual(var, false, INT_MAX);
    }
  } else if (var->GetOriginType() == TraceType::Const) {
    item = GuardId(var);
  } else if (need_specialize) {
    item = GuardEqual(var, true, INT_MAX);
  } else {
    item = GuardEqual(var, false, INT_MAX);
  }
  return item;
}

static GuardItemPtr GuardOnMutableOrConstObj(TracePtr var) {
  PyObject *obj = var->GetObject();
  GuardItemPtr item = nullptr;
  if (HasMutableOrConstAttr(obj)) {
    if (CheckMutableOrNonConstAttr(obj)) {
      item = GuardEqual(var, false, INT_MAX);
    } else {
      item = GuardEqual(var, true, INT_MAX);
    }
  }
  return item;
}

static GuardItemPtr GuardOnDynamicLenContainer(TracePtr var) {
  PyObject *obj = var->GetObject();
  GuardItemPtr item = nullptr;
  if (HasDynamicLength(obj)) {
    if (CheckDynamicLength(obj)) {
      item = GuardType(var);
    } else {
      item = GuardEqual(var, false, 0);
    }
  }
  return item;
}

void OptGuard::AddTraceFromGuard(const std::vector<TracePtr> &traces, OptGuardPtr other) {
  for (size_t i = 0; i < traces.size(); ++i) {
    auto dst = traces[i];
    auto src = std::make_shared<RootTrace>(dst->GetObject(), TraceType::Param, i);
    for (auto item : other->guardList_) {
      item->Replace(dst, src);
    }
  }
  for (auto item : other->guardList_) {
    guardList_.push_back(item);
  }
}

std::string OptGuard::GetDescript() {
  std::string ret;
  for (auto item : guardList_) {
    ret += ";" + item->ToString();
  }
  if (ret.size() > 0) {
    ret = ret.substr(1);
  }
  return ret;
}

void OptGuard::UpdateConfig(const std::map<std::string, bool> &config) {
  for (auto item : config) {
    if (g_mapDefaultConfig.find(item.first) != g_mapDefaultConfig.end()) {
      config_[item.first] = item.second;
    }
  }
}

void OptGuard::Backup() { guardStack_.push(std::make_pair(guardList_, guardMap_)); }

void OptGuard::Rollback() {
  GuardCheckPoint point = guardStack_.top();
  guardList_.swap(point.first);
  guardMap_.swap(point.second);
  guardStack_.pop();
}

void OptGuard::Pop() { guardStack_.pop(); }

static bool MatchDynamicShape(GuardItemPtr item, const std::vector<GuardItemPtr> &list) {
  auto trace_type = item->GetTrace()->GetTraceType();
  auto guard_type = item->GetType();
  if ((trace_type != TraceType::Deref && trace_type != TraceType::Param) || guard_type != GIType::GTEqual) {
    return false;
  }
  for (auto other : list) {
    if (item->MatchDynamicShape(other)) {
      return true;
    }
  }
  return false;
}

bool OptGuard::MatchShape(OptGuardPtr other) {
  if (std::any_of(guardList_.begin(), guardList_.end(), [other](auto &item) {
        return (!std::any_of(other->guardList_.begin(), other->guardList_.end(), [item](GuardItemPtr oi) {
          return *item == *oi;
        }) && !MatchDynamicShape(item, other->guardList_));
      })) {
    return false;
  }
  if (std::any_of(other->guardList_.begin(), other->guardList_.end(), [this](auto &item) {
        return (!std::any_of(guardList_.begin(), guardList_.end(), [item](GuardItemPtr oi) { return *item == *oi; }));
      })) {
    return false;
  }
  return true;
}

static PyObject *FindItem(const std::vector<GuardItemPtr> &guardList, int idx, TraceType type, PyObject *obj) {
  auto iter = std::find_if(guardList.begin(), guardList.end(), [idx, type](GuardItemPtr item) {
    if (item->GetTrace()->GetTraceType() == type) {
      int index;
      std::string name, module_name;
      (reinterpret_cast<RootTrace *>(item->GetTrace().get()))->GetParam(&index, &name, &module_name);
      return (idx == index);
    } else {
      return false;
    }
  });
  if (iter != guardList.end()) {
    GuardItemPtr item = *iter;
    return item->ApplyDynamicShape(obj);
  } else {
    return nullptr;
  }
}

std::vector<PyObject *> OptGuard::ApplyDynamicShape(PyFrameObject *f) {
  std::vector<PyObject *> ret;
  int argc = f->f_code->co_argcount + f->f_code->co_kwonlyargcount;
  PyTupleObject *vargs = NULL;
  PyDictObject *kwargs = NULL;
  if (f->f_code->co_flags & CO_VARARGS) {
    vargs = _PyTuple_CAST(f->f_localsplus[argc]);
  }
  if (f->f_code->co_flags & CO_VARKEYWORDS) {
    kwargs = reinterpret_cast<PyDictObject *>(f->f_localsplus[argc + (vargs ? 1 : 0)]);
  }
  for (int i = 0; i < argc; ++i) {
    auto new_obj = FindItem(guardList_, i, TraceType::Param, f->f_localsplus[i]);
    if (new_obj == nullptr) {
      ret.push_back(nullptr);
    } else {
      ret.push_back(f->f_localsplus[i]);
      f->f_localsplus[i] = new_obj;
    }
  }
  if (vargs != NULL) {
    ret.push_back(nullptr);
  }
  if (kwargs != NULL) {
    ret.push_back(nullptr);
  }
  ret.resize(f->f_code->co_nlocals, nullptr);
  for (int i = 0; f->f_code->co_cell2arg && i < PyTuple_GET_SIZE(f->f_code->co_cellvars); ++i) {
    Py_ssize_t arg = f->f_code->co_cell2arg[i];
    if (arg != CO_CELL_NOT_AN_ARG) {
      auto cell = f->f_localsplus[f->f_code->co_nlocals + i];
      auto new_obj = FindItem(guardList_, i, TraceType::Deref, PyCell_GET(cell));
      if (new_obj == nullptr) {
        ret.push_back(nullptr);
      } else {
        ret.push_back(PyCell_GET(cell));
        PyCell_SET(cell, new_obj);
      }
    }
  }
  ret.resize(f->f_code->co_nlocals + PyTuple_GET_SIZE(f->f_code->co_cellvars), nullptr);
  for (int i = 0; i < PyTuple_GET_SIZE(f->f_code->co_freevars); ++i) {
    Py_ssize_t arg = PyTuple_GET_SIZE(f->f_code->co_cellvars) + i;
    auto cell = f->f_localsplus[f->f_code->co_nlocals + arg];
    auto new_obj = FindItem(guardList_, arg, TraceType::Deref, PyCell_GET(cell));
    if (new_obj == nullptr) {
      ret.push_back(nullptr);
    } else {
      ret.push_back(PyCell_GET(cell));
      PyCell_SET(cell, new_obj);
    }
  }
  return ret;
}

void OptGuard::RevertDynamicShape(PyFrameObject *f, const std::vector<PyObject *> &backup) {
  int argc = f->f_code->co_argcount + f->f_code->co_kwonlyargcount;
  for (int i = 0; i < argc; ++i) {
    if (backup[i] != nullptr) {
      Py_XDECREF(f->f_localsplus[i]);
      f->f_localsplus[i] = backup[i];
    }
  }
  for (int i = 0; f->f_code->co_cell2arg && i < PyTuple_GET_SIZE(f->f_code->co_cellvars); ++i) {
    Py_ssize_t arg = f->f_code->co_cell2arg[i];
    if (arg != CO_CELL_NOT_AN_ARG) {
      auto cell = f->f_localsplus[f->f_code->co_nlocals + i];
      if (backup[f->f_code->co_nlocals + i] != nullptr) {
        Py_XDECREF(PyCell_GET(cell));
        PyCell_SET(cell, backup[f->f_code->co_nlocals + i]);
      }
    }
  }
  for (int i = 0; i < PyTuple_GET_SIZE(f->f_code->co_freevars); ++i) {
    Py_ssize_t arg = PyTuple_GET_SIZE(f->f_code->co_cellvars) + i;
    auto cell = f->f_localsplus[f->f_code->co_nlocals + arg];
    if (backup[f->f_code->co_nlocals + arg] != nullptr) {
      Py_XDECREF(PyCell_GET(cell));
      PyCell_SET(cell, backup[f->f_code->co_nlocals + arg]);
    }
  }
}

std::string OptGuard::ToString() const {
  std::stringstream s;
  for (const auto &i : guardMap_) {
    s << "  guard [ " << i.first << " ] at [" << i.second.get() << "]\n";
  }
  return s.str();
}

}  // namespace pijit
}  // namespace mindspore
