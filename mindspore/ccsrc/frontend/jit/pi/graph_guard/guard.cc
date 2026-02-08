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
#include "frontend/jit/pi/graph_guard/guard.h"
#include <chrono>
#include <regex>
#include "pybind11/pybind11.h"
#include "include/frontend/operator/primitive_py.h"
#include "include/utils/convert_utils_py.h"
#include "frontend/jit/pi/utils/utils.h"
#include "frontend/jit/pi/graph_guard/strategy.h"
#include "frontend/jit/pi/python_adapter/pydef.h"

namespace mindspore {
namespace pijit {
const char kSpecializeScalar[] = "specialize_scalar";
const char kSpecializeContainer[] = "specialize_container";
const char kSpecializeTensor[] = "specialize_tensor";
const char kGuardRelaxCnt[] = "relax_guard_count";

static std::map<std::string, bool> g_mapBoolDefaultConfig = {
  {kSpecializeScalar, false},
  {kSpecializeContainer, false},
  {kSpecializeTensor, false},
};

static std::map<std::string, int> g_mapIntDefaultConfig = {
  {kGuardRelaxCnt, 0},
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
    PyObject *key;
    PyObject *val;
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
                                std::map<std::string, std::pair<size_t, std::vector<size_t>>> *item_info,
                                std::map<std::string, std::pair<size_t, size_t>> *trace_info,
                                std::map<std::string, std::pair<size_t, size_t>> *guard_freq_info) const;
  OptGuardPerfImpl() = default;
  virtual ~OptGuardPerfImpl() = default;
  virtual void LogGuardPerfStart(OptGuard *tag2, GuardItem *item);
  virtual void LogGuardPerfEnd(GuardItem *item, bool res);
  virtual void LogItemPerfStart(int total_stage);
  virtual void LogItemPerfEnd(GuardItem *item, int stage);
  virtual void LogTracePerfStart();
  virtual void LogTracePerfEnd(Trace *trace, bool cache);

 protected:
  OptGuard *cur_tag2_ = nullptr;
  GuardItem *cur_guard_ = nullptr;
  std::chrono::steady_clock::time_point guard_start_;
  std::chrono::steady_clock::time_point trace_start_;
  std::vector<std::chrono::steady_clock::time_point> item_stage_;
  std::map<std::string, std::pair<size_t, size_t>> guard_info_;
  std::map<std::string, std::pair<size_t, std::vector<size_t>>> item_info_;
  std::map<std::string, std::pair<size_t, size_t>> trace_info_;
  std::map<std::string, std::pair<size_t, size_t>> guard_freq_info_;
};

static OptGuardPerfImpl g_guard_perf;
OptGuardPerf *OptGuardPerf::GetGuardPerf() { return &g_guard_perf; }

void OptGuardPerfImpl::GetGuardPerfInfo(std::map<std::string, std::pair<size_t, size_t>> *guard_info,
                                        std::map<std::string, std::pair<size_t, std::vector<size_t>>> *item_info,
                                        std::map<std::string, std::pair<size_t, size_t>> *trace_info,
                                        std::map<std::string, std::pair<size_t, size_t>> *guard_freq_info) const {
  if (guard_info != nullptr) {
    guard_info->clear();
    guard_info->insert(guard_info_.begin(), guard_info_.end());
  }
  if (trace_info != nullptr) {
    trace_info->clear();
    trace_info->insert(trace_info_.begin(), trace_info_.end());
  }
  if (guard_freq_info != nullptr) {
    guard_freq_info->clear();
    guard_freq_info->insert(guard_freq_info_.begin(), guard_freq_info_.end());
  }
  if (item_info != nullptr) {
    item_info->clear();
    item_info->insert(item_info_.begin(), item_info_.end());
  }
}

void OptGuardPerfImpl::LogGuardPerfStart(OptGuard *tag2, GuardItem *item) {
  cur_guard_ = item;
  cur_tag2_ = tag2;
  guard_start_ = std::chrono::steady_clock::now();
}

void OptGuardPerfImpl::LogGuardPerfEnd(GuardItem *item, bool res) {
  auto duration =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - guard_start_);
  size_t dur = (size_t)(duration.count());
  size_t inc = 1;
  auto info = item->ToString();
  std::stringstream s;
  s << reinterpret_cast<void *>(cur_tag2_) << "=>" << reinterpret_cast<void *>(cur_guard_) << "=>";
  info = s.str() + info;
  auto iter = guard_info_.find(info);
  if (iter != guard_info_.end()) {
    iter->second.first += inc;
    iter->second.second += dur;
  } else {
    guard_info_[info] = std::make_pair(inc, dur);
  }
  iter = guard_freq_info_.find(info);
  if (iter != guard_freq_info_.end()) {
    if (res) {
      iter->second.first += 1;
    } else {
      iter->second.second += 1;
    }
  } else {
    if (res) {
      guard_freq_info_[info] = std::make_pair(1, 0);
    } else {
      guard_freq_info_[info] = std::make_pair(0, 1);
    }
  }
}

void OptGuardPerfImpl::LogItemPerfStart(int total_stage) {
  item_stage_.clear();
  item_stage_.resize(total_stage + 1);
  item_stage_[0] = std::chrono::steady_clock::now();
}

void OptGuardPerfImpl::LogItemPerfEnd(GuardItem *item, int stage) {
  size_t cur_stage = static_cast<size_t>(stage + 1);
  if (item_stage_.size() > cur_stage) {
    item_stage_[cur_stage] = std::chrono::steady_clock::now();
  }
  if (item_stage_.size() == (cur_stage + 1)) {
    auto info = item->ToString();
    std::stringstream s;
    s << reinterpret_cast<void *>(cur_tag2_) << "=>" << reinterpret_cast<void *>(cur_guard_) << "=>";
    info = s.str() + info;
    std::vector<size_t> vecDur;
    for (int idx = 0; idx <= stage; ++idx) {
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(item_stage_[idx + 1] - item_stage_[idx]);
      vecDur.push_back((size_t)(duration.count()));
    }
    auto iter = item_info_.find(info);
    if (iter != item_info_.end()) {
      iter->second.first += 1;
      for (size_t i = 0; i < vecDur.size(); ++i) {
        iter->second.second[i] += vecDur[i];
      }
    } else {
      item_info_[info] = std::make_pair(1, vecDur);
    }
  }
}

void OptGuardPerfImpl::LogTracePerfStart() { trace_start_ = std::chrono::steady_clock::now(); }

void OptGuardPerfImpl::LogTracePerfEnd(Trace *trace, bool cache) {
  auto duration =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - trace_start_);
  size_t dur = (size_t)(duration.count());
  size_t inc = 1;
  auto info = trace->ToString(true);
  std::stringstream s;
  s << reinterpret_cast<void *>(cur_guard_) << "=>";
  if (cache) {
    s << "cache:";
  }
  info = s.str() + info;
  auto iter = trace_info_.find(info);
  if (iter != trace_info_.end()) {
    iter->second.first += inc;
    iter->second.second += dur;
  } else {
    trace_info_[info] = std::make_pair(inc, dur);
  }
}

OptGuard::OptGuard() {
  bool_config_ = g_mapBoolDefaultConfig;
  int_config_ = g_mapIntDefaultConfig;
}

void OptGuard::UpdateGuardList(GuardItemPtr item) {
  // reorder list to speed up check on next run
  for (size_t i = 0; i < guardList_.size(); ++i) {
    if (guardList_[i] == item) {
      guardList_.erase(guardList_.begin() + i);
      guardList_.insert(guardList_.begin(), item);
    }
  }
}

static std::string PrintTraceObj(const py::handle &object) {
  std::stringstream s;
  auto print_tensor = [&s](const py::handle &tensor) {
    constexpr int limit_print_size = 256;
    if (tensor.attr("_size").cast<int>() <= limit_print_size) {
      s << py::str(tensor);
    } else {
      s << Py_TYPE(tensor.ptr())->tp_name << "(shape=" << py::str(tensor.attr("shape").ptr())
        << ", dtype=" << py::str(tensor.attr("dtype").ptr()) << ")";
    }
  };

  if (object.ptr() == nullptr) {
    s << "<nullptr>";
  } else if (IsTensorPyObject(object.ptr())) {
    print_tensor(object);
  } else {
    s << py::str(object);
  }

  return s.str();
}

std::string GuardCheckFailInfo(const GuardItemPtr &item, const py::handle &object) {
  std::stringstream s;

  const char *type = object.ptr() != nullptr ? Py_TYPE(object.ptr())->tp_name : "";
  s << std::endl
    << "Guard check fail: " << item->GetFailInfo() << std::endl
    << item->ToString() << std::endl
    << "v.s." << std::endl
    << type << "(" << object.ptr() << "): " << PrintTraceObj(object) << GetItemDataStr(item, object.ptr());

  return s.str();
}

bool OptGuard::Check(PyFrameWrapper frame, bool perf) {
  // see `OptGuard::Record`, no duplicate item
  const auto &list = guardList_;
  for (size_t i = 0, size = list.size(); i < size; ++i) {
    const auto &item = list[i];
    if (item->fail_count()) {
      return false;
    }
    if (item->checked()) {
      continue;
    }
    if (perf) {
      g_guard_perf.LogGuardPerfStart(this, item.get());
      item->set_perf(perf);
    }
    bool result = item->Check(frame);
    item->Cache(result);
    if (perf) {
      g_guard_perf.LogGuardPerfEnd(item.get(), result);
      item->set_perf(false);
    }
    if (result) {
      continue;
    }
    UpdateGuardList(item);
    return false;
  }
  return true;
}

bool OptGuard::GuardOn(TracePtr var, GuardLevel tp, bool needSpecialize, int recurseDepth) {
  if (tp == GuardLevel::kGuardMatchIDS) {
    return GuardIDS(var);
  }
  // Now we have TypeGuard IdGuard NameGuard AttrGuard EqGuard, let's add guard to guardlist based on type
  PyObject *obj = var->GetObject();
  if (int_config_.find(kGuardRelaxCnt) != int_config_.end() && int_config_[kGuardRelaxCnt] != 0) {
    var->SetRelaxCount(int_config_[kGuardRelaxCnt]);
  }
  GuardItemPtr item = nullptr;
  if (obj != nullptr) {
    if (tp == GuardLevel::GDeduce) {
      item = GuardOnGDeduce(var, obj, bool_config_);
    } else if (tp == GuardLevel::GId) {
      item = GuardId(var);
    } else if (tp == GuardLevel::GType) {
      item = GuardType(var);
    } else if (tp == GuardLevel::GEqual) {
      item = GuardEqual(var, needSpecialize, recurseDepth);
    }
  } else {
    // Check obj == None
    item = GuardEqual(var, 0);
  }
  return Record(item);
}

bool OptGuard::Record(const GuardItemPtr &new_item) {
  GuardItemPtr item = new_item;
  if (item == nullptr) {
    MS_LOG(DEBUG) << "Guard item is null";
    return false;
  }
  const auto &trace_ctx_stack = TraceManager::CurrentContextInfo();
  if (trace_ctx_stack && trace_ctx_stack->location()) {
    item->location_ = trace_ctx_stack->location()->ToString();
  }
  MS_EXCEPTION_IF_NULL(code_hub());
  auto &guard_map = code_hub()->guard_map();

  size_t hash = item->Info().Id();
  auto cur_item = &guard_map[hash];
  if (*cur_item == nullptr) {
    *cur_item = item;
    item->UpdateTrace(&code_hub()->trace_map());
  } else if (*cur_item != item) {
    bool is_match = item->operator==(**cur_item);
    MS_LOG(DEBUG) << "find duplicate guard item in the global compile cache, current == reused: "
                  << (is_match ? "true" : "false, id conflict, not reuse") << std::endl
                  << "current: " << item.get() << ": [ " << item->ToString() << " ]" << std::endl
                  << "reused : " << cur_item->get() << ": [ " << (*cur_item)->ToString() << " ]";
    if (!is_match) {
      cur_item = &item;
    }
  }
  auto &list = guardList_;
  auto iter = std::find_if(list.begin(), list.end(), [hash](const auto &p) { return p->Info().Id() == hash; });
  if (iter == list.end()) {
    list.push_back(*cur_item);
    MS_LOG(INFO) << "New guard: " << (*cur_item)->ToString();
  } else if (*cur_item != *iter) {
    bool is_match = (*iter)->operator==(**cur_item);
    MS_LOG(DEBUG) << "find duplicate guard item for the function, current == reused: "
                  << (is_match ? "true" : "false, id conflict, not reuse") << std::endl
                  << "current: " << iter->get() << ": [ " << (*iter)->ToString() << " ]" << std::endl
                  << "reused : " << cur_item->get() << ": [ " << (*cur_item)->ToString() << " ]";
    if (!is_match) {
      list.push_back(*cur_item);
    }
  }
  return true;
}

bool OptGuard::GuardIDS(const TracePtr &tr) {
  if (tr == nullptr) {
    return false;
  }
  GuardItemPtr item;
  for (const auto &i : guardList_) {
    bool is_match_ids = i->GetType() == GIType::kMatchIDS;
    bool is_same_id = i->GetTrace()->GetObject() == tr->GetObject();
    if (is_match_ids && is_same_id) {
      item = i;
      break;
    }
  }
  auto new_item = pijit::GuardIDS(tr, item);
  if (new_item != item) {
    Record(new_item);
  }
  return true;
}

bool OptGuard::Erase(const GuardItemPtr &last) {
  auto iter = std::find(guardList_.rbegin(), guardList_.rend(), last);
  if (iter == guardList_.rend()) {
    return false;
  }
  guardList_.erase(guardList_.begin() + std::distance(iter, guardList_.rend()) - 1);
  return true;
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
    item = GuardEqual(var, false, 0);
  } else if (py::isinstance<mindspore::ParamInfo>(obj)) {
    item = GuardEqual(var, true, INT_MAX);
  } else {
    // CheckLiteral use exactly type match, so mindspore.mutable object will come to this case.
    item = GuardOnMutableOrConstObj(var);
    if (item == nullptr) {
      item = GuardType(var);
    }
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

void OptGuard::UpdateConfig(const std::map<std::string, bool> &bool_config,
                            const std::map<std::string, int> &int_config) {
  for (auto item : bool_config) {
    if (g_mapBoolDefaultConfig.find(item.first) != g_mapBoolDefaultConfig.end()) {
      bool_config_[item.first] = item.second;
    }
  }
  for (auto item : int_config) {
    if (g_mapIntDefaultConfig.find(item.first) != g_mapIntDefaultConfig.end()) {
      int_config_[item.first] = item.second;
    }
  }
}

void OptGuard::Backup() { guardStack_.push(std::make_tuple(guardList_)); }

void OptGuard::Rollback() {
  GuardCheckPoint point = guardStack_.top();
  guardList_.swap(std::get<0>(point));
  guardStack_.pop();
}

void OptGuard::Pop() { guardStack_.pop(); }

std::string OptGuard::ToString() const {
  std::stringstream s;
  for (const auto &i : guardList_) {
    s << "  guard [" << i.get() << "] " << i->Info().Id() << " [" << i->ToString() << " ]" << std::endl;
  }
  return s.str();
}

OptGuardPtr OptGuard::Optimize() {
  bool need_update = false;
  for (size_t i = 0; i < guardList_.size(); ++i) {
    auto old_item = guardList_[i];
    auto new_item = old_item->Optimize();
    if (new_item != nullptr) {
      guardList_[i] = new_item;
      need_update = true;
    }
  }
  if (need_update) {
    info_ = nullptr;
    Info();
    return shared_from_this();
  } else {
    return nullptr;
  }
}

void OptGuard::FilterConstItem() {
  for (size_t i = 0; i < guardList_.size();) {
    auto item = guardList_[i];
    if (item->GetTrace()->IsConst()) {
      guardList_.erase(guardList_.begin() + i);
    } else {
      i++;
    }
  }
}

}  // namespace pijit
}  // namespace mindspore
