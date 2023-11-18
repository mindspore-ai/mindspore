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
#include "pybind11/pybind11.h"
#include "pybind_api/ir/cell_py.h"
#include "pybind_api/ir/primitive_py.h"
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/pi/utils/utils.h"

namespace mindspore {
namespace jit {
namespace graph {
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

OptGuard::OptGuard() { config_ = g_mapDefaultConfig; }

OptGuard::OptGuard(const std::map<std::string, bool> &cfg) { UpdateConfig(cfg); }

bool OptGuard::Check(const PyFrameObject *frame, bool print, std::map<std::string, PyObject *> *cache) {
  for (size_t i = 0; i < guardList_.size(); ++i) {
    GuardItemPtr item = guardList_[i];
    if (!item->Check(frame, cache)) {
      // reorder list to speed up check on next run
      GuardItemPtr tmp = item;
      guardList_.erase(guardList_.begin() + i);
      guardList_.insert(guardList_.begin(), tmp);
      if (print) {
        GRAPH_JIT_LOG_F("Guard check fail: %s\n", item->ToString().c_str());
      } else {
        MS_LOG(DEBUG) << "Guard check fail:" << item->ToString();
      }
      return false;
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
    std::string strItem = item->ToString();
    if (guardMap_.find(strItem) == guardMap_.end()) {
      guardList_.push_back(item);
      guardMap_[strItem] = item;
    }
    return true;
  } else {
    return false;
  }
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
      item = GuardType(var);
    }
  } else if (py::isinstance<mindspore::Cell>(obj)) {
    item = GuardId(var);
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
      item = GuardEqual(var, true, INT_MAX);
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

void OptGuard::Backup() {
  backupList_.clear();
  backupList_.insert(backupList_.begin(), guardList_.begin(), guardList_.end());
  backupMap_.clear();
  backupMap_.insert(guardMap_.begin(), guardMap_.end());
}

void OptGuard::Rollback() {
  backupList_.swap(guardList_);
  backupList_.clear();
  backupMap_.swap(guardMap_);
  backupMap_.clear();
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
