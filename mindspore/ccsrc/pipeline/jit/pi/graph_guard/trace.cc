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
#include "pipeline/jit/pi/graph_guard/trace.h"
#include <map>
#include <vector>
#include <set>
#include <unordered_map>
#include <functional>
#include <utility>
#include <regex>
#include "pipeline/jit/pi/graph_guard/guard.h"
#include "pipeline/jit/pi/graph_guard/guard_utils.h"
#include "pybind11/pybind11.h"
#include "pybind_api/ir/primitive_py.h"
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/pi/graph_guard/infer.h"
#include "pipeline/jit/pi/graph_guard/strategy.h"
#include "pipeline/jit/pi/utils/utils.h"
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/pi/graph_capture/abstract_object.h"
#include "pipeline/jit/pi/pi_jit_config.h"
#include "pipeline/jit/pi/external.h"

namespace mindspore {
namespace pijit {

static constexpr size_t kParamCountOne = 1;
static constexpr size_t kParamCountTwo = 2;
static constexpr size_t kParamCountThree = 3;
static constexpr size_t kParamIndexOne = 0;
static constexpr size_t kParamIndexTwo = 1;
static constexpr size_t kParamIndexThree = 2;
static const char kCastPrimName[] = "Cast";
static const char kLayerNormPrimName[] = "LayerNorm";
static const char kReshapePrimName[] = "Reshape";
static const char kShapePrimName[] = "Shape";
static const char kShape_Name[] = "shape_";
static const char kShapeName[] = "shape";
static const char kRankPrimName[] = "Rank";
static const char kCastToMSTensor[] = "cast_to_ms_tensor";
static const char kCastToAdapterTensor[] = "cast_to_adapter_tensor";
static const std::vector<std::string> kCastFunc = {kCastToMSTensor, kCastToAdapterTensor};
static const char kIsInstance[] = "isinstance";
static const char kTensorName[] = "Tensor";
static const char kDTypeAttrName[] = "dtype";
static const char kDType_AttrName[] = "dtype_";
static const char kDTypePrimName[] = "DType";
static const char kCodeName[] = "__code__";
static const char kFuncName[] = "__func__";
static const char kIsSeqValUnknown[] = "is_sequence_value_unknown";
static const char kIsSeqShapeUnknown[] = "is_sequence_shape_unknown";
static const char kMindTorchFlag[] = "mindtorch";
static const char kTrainingFlag[] = "training";
static const char kMindSporePackPrefix[] = "mindspore.";
static const char kMindtorchPackPrefix[] = "mindtorch.";

constexpr const char *kFuncWhiteListModuleName = "mindspore._extends.pijit.pijit_func_white_list";
constexpr const char *kGuardFuncMapName = "_guard_func_map";

static PyObject *RichCompare(PyObject *left, PyObject *right, int oparg);

static bool IsCastFunc(std::string name) {
  return std::find(kCastFunc.begin(), kCastFunc.end(), name) != kCastFunc.end();
}

static TracePtr OptimizeTrace(TracePtr trace, bool *update) {
  if (trace != nullptr) {
    auto new_trace = trace->Optimize();
    if (new_trace != nullptr) {
      if (update != nullptr) {
        *update = true;
      }
      return new_trace;
    }
  }
  return trace;
}

template <typename T>
std::shared_ptr<T> CastTrace(TracePtr trace) {
  if (trace != nullptr && T::Support(trace->GetTraceType())) {
    return std::static_pointer_cast<T>(trace);
  }
  return nullptr;
}

static ConstTracePtr CastConstTrace(TracePtr trace) {
  ConstTracePtr ret = CastTrace<ConstTrace>(trace);
  if (ret != nullptr && ret->GetIndex() == -1) {
    return ret;
  }
  return nullptr;
}

static OpTracePtr CastOpTrace(TracePtr trace, int opcode) {
  OpTracePtr ret = CastTrace<OpTrace>(trace);
  if (ret != nullptr && ret->GetOpCode() == opcode) {
    return ret;
  }
  return nullptr;
}

static OpTracePtr CastOpTrace(TracePtr trace, const std::string &name) {
  OpTracePtr ret = CastTrace<OpTrace>(trace);
  if (ret != nullptr && ret->GetName() == name) {
    return ret;
  }
  return nullptr;
}

class TracePerf {
 public:
  TracePerf(Trace *trace, bool enable, bool cache)
      : trace_(trace), enable_(enable), cache_(cache), perf_(OptGuardPerf::GetGuardPerf()) {
    if (enable_) {
      perf_->LogTracePerfStart();
    }
  }
  ~TracePerf() {
    if (enable_) {
      perf_->LogTracePerfEnd(trace_, cache_);
    }
  }

 protected:
  Trace *trace_;
  bool enable_;
  bool cache_;
  OptGuardPerf *perf_;
};

Trace::Trace(PyObject *pObj, std::shared_ptr<Trace> pOrigin)
    : obj_(pObj),
      origin_(pOrigin),
      info_(nullptr),
      is_const_(false),
      relax_count_(-1),
      relax_limit_(0),
      is_specialized_(false),
      depth_(0) {
  if (pOrigin != nullptr) {
    originType_ = pOrigin->GetOriginType();
    curType_ = pOrigin->GetTraceType();
  } else {
    originType_ = Unknown;
    curType_ = Unknown;
  }
  if (obj_ != Py_None && obj_ != NULL) {
    Py_INCREF(obj_);
  }
}

Trace::~Trace() {
  if (obj_ != Py_None && obj_ != NULL) {
    Py_DECREF(obj_);
  }
}

TracePtr Trace::GetOrigin() {
  if (origin_ != nullptr) {
    return origin_;
  } else {
    return nullptr;
  }
}

PyObject *Trace::GetObject() { return obj_; }

TraceType Trace::GetTraceType() { return curType_; }

TraceType Trace::GetOriginType() { return originType_; }

void Trace::Replace(std::shared_ptr<Trace> dst, std::shared_ptr<Trace> src) {
  if (origin_ != nullptr) {
    if (*origin_ == *src) {
      origin_ = dst;
    } else {
      origin_->Replace(dst, src);
    }
  }
}

bool Trace::operator==(const Trace &trace) {
  if (curType_ == trace.curType_ && obj_ == trace.obj_) {
    return true;
  } else {
    return false;
  }
}

void Trace::Detach() {
  if (obj_ != Py_None && obj_ != nullptr && !is_const_ && !PyLong_Check(obj_) && !PyType_Check(obj_)) {
    Py_DECREF(obj_);
    obj_ = nullptr;
  }
  if (origin_ != nullptr) {
    origin_->Detach();
  }
}

PyObject *Trace::Retrieve(PTraceContext context, bool perf) {
  TracePerf tp(this, perf, true);
  if (is_const_) {
    Py_XINCREF(obj_);
    return obj_;
  }
  if (context->cache != nullptr) {
    size_t szTrace = this->Info().Id();
    auto cache = context->cache;
    auto iter = cache->find(szTrace);
    if (iter != cache->end()) {
      auto item = iter->second;
      Py_XINCREF(item);
      return item;
    }
  }
  return nullptr;
}

void Trace::Cache(PTraceContext context, PyObject *obj) {
  if (context->cache != nullptr && obj != nullptr) {
    size_t szTrace = this->Info().Id();
    Py_XINCREF(obj);
    auto iter = context->cache->find(szTrace);
    if (iter != context->cache->end()) {
      Py_XDECREF(iter->second);
      iter->second = obj;
    } else {
      (*(context->cache))[szTrace] = obj;
    }
  }
  if (RelaxEnabled() && obj != nullptr) {
    if (obj_ != nullptr) {
      auto cmp = RichCompare(obj_, obj, Py_EQ);
      if (cmp != Py_True) {
        relax_count_ = -1;
      } else if (relax_count_ < relax_limit_) {
        relax_count_++;
      } else {
        is_const_ = true;
      }
      Py_XDECREF(cmp);
    }
    if (obj_ == nullptr) {
      Py_XINCREF(obj);
      obj_ = obj;
    }
  }
}

bool Trace::IsConst() const { return is_const_; }

TracePtr Trace::This() { return shared_from_this(); }

void Trace::SetRelaxCount(int cnt) {
  relax_count_ = -1;
  relax_limit_ = cnt;
}

int Trace::GetRelaxCount() const { return relax_limit_; }

void Trace::EnableRelax() { relax_count_ = 0; }

bool Trace::RelaxEnabled() const { return relax_count_ >= 0; }

bool Trace::IsSpecialized() const { return is_specialized_; }

int Trace::GetDepth() const { return depth_; }

TracePtr Trace::Optimize() { return nullptr; }

std::string Trace::FormatString(std::map<Trace *, size_t> *cache) {
  cache->insert(std::make_pair(this, cache->size()));
  return "%" + std::to_string(cache->find(this)->second) + " = " + this->ToString();
}

RootTrace::RootTrace(PyObject *pObj, TraceType tt, int index, std::string name, std::string module_name)
    : Trace(pObj, nullptr), idx_(index), name_(name), module_name_(module_name) {
  depth_ = 1;
  originType_ = tt;
  curType_ = tt;
  for (auto n : kPIJitConfigDefault.allowed_inline_modules()) {
    if (module_name.find(n) == 0) {
      is_const_ = true;
      break;
    }
  }
  if (!is_const_ && (module_name.find(kMindSporePackPrefix) == 0 || module_name.find(kMindtorchPackPrefix) == 0 ||
                     name.find(kCastToAdapterTensor) == 0 || name.find(kCastToMSTensor) == 0)) {
    is_const_ = true;
  }
  if (curType_ == TraceType::Deref) {
    is_const_ = false;
  }
  if (pObj == nullptr) {
    return;
  }
  if (py::isinstance<mindspore::tensor::MetaTensor>(pObj) || py::isinstance<mindspore::tensor::Tensor>(pObj) ||
      IsStubTensor(py::cast<py::object>(pObj))) {
    is_specialized_ = false;
  }
}

void RootTrace::GetParam(int *index, std::string *name, std::string *module_name) {
  *index = idx_;
  *name = name_;
  *module_name = module_name_;
}

PyObject *RootTrace::Retrieve(PTraceContext context, bool perf) {
  PyObject *ret = Trace::Retrieve(context, perf);
  if (ret != nullptr) {
    return ret;
  }
  TracePerf tp(this, perf, false);
  switch (curType_) {
    case TraceType::Global: {
      ret = RetrieveGlobal(context);
      break;
    }
    case TraceType::Deref: {
      ret = RetrieveDeref(context);
      break;
    }
    case TraceType::Closure: {
      ret = RetrieveClosure(context);
      break;
    }
    case TraceType::BuiltIn: {
      ret = RetrieveBuiltin(context);
      break;
    }
    case TraceType::Local:
      ret = RetrieveLocal(context);
      Py_XINCREF(ret);
      break;
    case TraceType::Param:
      ret = RetrieveParam(context);
      Py_XINCREF(ret);
      break;
    case TraceType::Name: {
      ret = RetrieveName(context);
      break;
    }
    case TraceType::ClassDeref: {
      ret = RetrieveClassDeref(context);
      break;
    }
    default:
      break;
  }
  if (ret != Py_None && ret != NULL) {
    Cache(context, ret);
  }
  return ret;
}

PyObject *RootTrace::RetrieveGlobal(PTraceContext context) {
  MS_EXCEPTION_IF_CHECK_FAIL(name_.size() > 0, "check trace");
  PyObject *globals = context->f_globals;
  if (!module_name_.empty()) {
    PyObject *mn = PyUnicode_FromString(module_name_.c_str());
    PyObject *mm = PyImport_GetModule(mn);  // ensure module is initialized
    if (mn != nullptr && mm != nullptr) {
      globals = PyModule_GetDict(mm);
    }
    PyErr_Clear();
    Py_XDECREF(mn);
    Py_XDECREF(mm);
  }
  PyObject *key = PyUnicode_FromString(name_.c_str());
  PyObject *ret = PyObject_GetItem(globals, key);
  if (ret == nullptr) {
    PyErr_Clear();
    ret = PyObject_GetItem(context->f_builtins, key);
    if (ret == nullptr) {
      PyErr_Clear();
    }
  }
  Py_DECREF(key);
  return ret;
}

PyObject *RootTrace::RetrieveDeref(PTraceContext context) {
  PyObject *ret = nullptr;
  PyObject *cell = context->f_localsplus[context->f_code->co_nlocals + idx_];
  if (cell != nullptr && cell != Py_None) {
    ret = reinterpret_cast<PyObject *>(PyCell_GET(cell));
    Py_XINCREF(ret);
  }
  return ret;
}

PyObject *RootTrace::RetrieveClosure(PTraceContext context) {
  PyObject *ret = context->f_localsplus[context->f_code->co_nlocals + idx_];
  Py_XINCREF(ret);
  return ret;
}

PyObject *RootTrace::RetrieveBuiltin(PTraceContext context) {
  MS_EXCEPTION_IF_CHECK_FAIL(name_.size() > 0, "check trace");
  PyObject *key = PyUnicode_FromString(name_.c_str());
  PyObject *ret = PyObject_GetItem(context->f_builtins, key);
  if (ret == nullptr) {
    PyErr_Clear();
    ret = PyObject_GetItem(context->f_globals, key);
    if (ret == nullptr) {
      PyErr_Clear();
    }
  }
  Py_DECREF(key);
  return ret;
}

PyObject *RootTrace::RetrieveLocal(PTraceContext context) { return context->f_locals; }

PyObject *RootTrace::RetrieveParam(PTraceContext context) { return context->f_localsplus[idx_]; }

PyObject *RootTrace::RetrieveName(PTraceContext context) {
  PyObject *ret = nullptr;
  PyObject *name = PyTuple_GetItem(context->f_code->co_names, idx_);
  PyObject *locals = context->f_locals;
  if (PyDict_CheckExact(locals)) {
    ret = PyDict_GetItem(locals, name);
    Py_XINCREF(ret);
  } else {
    ret = PyObject_GetItem(locals, name);
  }
  if (ret == nullptr) {
    ret = PyDict_GetItem(context->f_globals, name);
    Py_XINCREF(ret);
  }
  if (ret == nullptr) {
    if (PyDict_CheckExact(context->f_builtins)) {
      ret = PyDict_GetItem(context->f_builtins, name);
      Py_XINCREF(ret);
    } else {
      ret = PyObject_GetItem(context->f_builtins, name);
    }
  }
  return ret;
}

PyObject *RootTrace::RetrieveClassDeref(PTraceContext context) {
  PyObject *ret = nullptr;
  Py_ssize_t idx = idx_ - PyTuple_GET_SIZE(context->f_code->co_cellvars);
  if (idx >= 0 && idx < PyTuple_GET_SIZE(context->f_code->co_freevars)) {
    PyObject *name = PyTuple_GET_ITEM(context->f_code->co_freevars, idx);
    if (PyDict_CheckExact(context->f_locals)) {
      ret = PyDict_GetItem(context->f_locals, name);
      Py_XINCREF(ret);
    } else {
      ret = PyObject_GetItem(context->f_locals, name);
    }
    if (!ret) {
      PyObject *cell = context->f_localsplus[context->f_code->co_nlocals + idx_];
      ret = PyCell_GET(cell);
      Py_XINCREF(ret);
    }
  }
  return ret;
}

std::string RootTrace::ToString(bool include_param) {
  if (strTrace_.size() > 0) {
    return strTrace_;
  }
  std::string ret;
  switch (curType_) {
    case TraceType::Global:
      if (!module_name_.empty()) {
        ret = "(global " + module_name_ + ".__dict__[" + name_ + "])";
      } else {
        ret = "f_globals[" + name_ + "]";
      }
      break;
    case TraceType::Deref:
      ret = "f_freevars[" + std::to_string(idx_) + "]";
      break;
    case TraceType::Closure:
      ret = "f_closure[" + std::to_string(idx_) + "]";
      break;
    case TraceType::BuiltIn:
      ret = "f_builtins[" + name_ + "]";
      break;
    case TraceType::Local:
      ret = "f_locals";
      break;
    case TraceType::Param:
      ret = "f_localsplus[";
      ret += std::to_string(idx_);
      ret += "]";
      break;
    case TraceType::Name:
      ret = "f->f_code->co_names[";
      ret += std::to_string(idx_);
      ret += "]";
      break;
    case TraceType::ClassDeref:
      ret = "f->f_classdef[";
      ret += std::to_string(idx_);
      ret += "]";
      break;
    default:
      ret = "unknown_root";
      break;
  }
  ret = (is_const_ ? std::string("const:") : std::string("var:")) + ret;
  ret = std::regex_replace(ret, std::regex("(\n)"), "");
  strTrace_ = ret;
  return ret;
}

const InfoPack &RootTrace::Info() {
  if (info_ == nullptr) {
    InfoPack info;
    info << uint8_t(curType_);
    info.Begin();
    switch (curType_) {
      case TraceType::Global:
        info << (!module_name_.empty());
        if (!module_name_.empty()) {
          info << module_name_ << name_;
        } else {
          info << name_;
        }
        break;
      case TraceType::Deref:
      case TraceType::Closure:
      case TraceType::Param:
      case TraceType::Name:
      case TraceType::ClassDeref:
        info << idx_;
        break;
      case TraceType::BuiltIn:
        info << name_;
        break;
      case TraceType::Local:
      default:
        break;
    }
    info.End();
    info_ = std::make_shared<InfoPack>(info);
    info_->Update();
  }
  return *info_;
}

bool RootTrace::operator==(const Trace &trace) {
  bool ret = false;
  if (Trace::operator==(trace)) {
    const RootTrace &t = (const RootTrace &)trace;
    ret = idx_ == t.idx_;
    if (ret && idx_ == -1) {
      ret = name_ == t.name_ && module_name_ == t.module_name_;
    }
  }
  return ret;
}

bool RootTrace::Support(TraceType tt) {
  switch (tt) {
    case TraceType::Global:
    case TraceType::Deref:
    case TraceType::Closure:
    case TraceType::BuiltIn:
    case TraceType::Local:
    case TraceType::Param:
    case TraceType::Name:
    case TraceType::ClassDeref:
      return true;
    default:
      return false;
  }
}

ItemTrace::ItemTrace(PyObject *pObj, TracePtr pOrigin, TracePtr pItem) : Trace(pObj, pOrigin), item_(pItem) {
  curType_ = TraceType::Item;
  if (origin_ != nullptr && item_ != nullptr) {
    if (origin_->IsConst() && item_->IsConst()) {
      is_const_ = true;
    }
    if (!origin_->IsSpecialized() && !item_->IsSpecialized()) {
      is_specialized_ = false;
    }
  }
  if (origin_ != nullptr) {
    depth_ = origin_->GetDepth() + 1;
  } else {
    depth_ = 1;
  }
  if (item_ != nullptr) {
    auto d = item_->GetDepth() + 1;
    if (d > depth_) {
      depth_ = d;
    }
  }
}

TracePtr ItemTrace::GetItem() { return item_; }

void ItemTrace::Replace(std::shared_ptr<Trace> dst, std::shared_ptr<Trace> src) {
  Trace::Replace(dst, src);
  if (item_ != nullptr) {
    if (*item_ == *src) {
      item_ = dst;
    } else {
      item_->Replace(dst, src);
    }
  }
}

PyObject *ItemTrace::Retrieve(PTraceContext context, bool perf) {
  PyObject *ret = Trace::Retrieve(context, perf);
  if (ret != nullptr) {
    return ret;
  }
  if (origin_ != nullptr && item_ != nullptr) {
    PyObject *pSet = origin_->Retrieve(context, perf);
    PyObject *pItem = item_->Retrieve(context, perf);
    if (pSet != NULL && pItem != NULL) {
      TracePerf tp(this, perf, false);
      if (PyDict_CheckExact(pSet)) {
        ret = PyDict_GetItem(pSet, pItem);
        Py_INCREF(ret);
      } else {
        ret = PyObject_GetItem(pSet, pItem);
      }
      Cache(context, ret);
    }
    Py_XDECREF(pSet);
    Py_XDECREF(pItem);
  }
  return ret;
}

std::string ItemTrace::ToString(bool include_param) {
  if (strTrace_.size() > 0) {
    return strTrace_;
  }
  std::string ret;
  if (origin_ != nullptr && item_ != nullptr) {
    std::string ori = origin_->ToString(include_param);
    std::string itm = item_->ToString(include_param);
    ret = ori + "[" + itm + "]";
  }
  ret = (is_const_ ? std::string("const:") : std::string("var:")) + ret;
  ret = std::regex_replace(ret, std::regex("(\n)"), "");
  strTrace_ = ret;
  return ret;
}

const InfoPack &ItemTrace::Info() {
  if (info_ == nullptr) {
    InfoPack info;
    info << uint8_t(curType_);
    info.Begin();
    info << (origin_ != nullptr && item_ != nullptr);
    if (origin_ != nullptr && item_ != nullptr) {
      auto ori = origin_->Info();
      auto itm = item_->Info();
      info << ori << itm;
    }
    info.End();
    info_ = std::make_shared<InfoPack>(info);
    info_->Update();
  }
  return *info_;
}

TracePtr ItemTrace::Optimize() {
  bool need_update = false;
  origin_ = OptimizeTrace(origin_, &need_update);
  item_ = OptimizeTrace(item_, &need_update);
  if (need_update) {
    if (origin_ != nullptr && item_ != nullptr && origin_->IsConst() && item_->IsConst()) {
      is_const_ = true;
    }
    info_ = nullptr;
    strTrace_ = "";
    Info();
    return shared_from_this();
  } else {
    return nullptr;
  }
}

void ItemTrace::SetRelaxCount(int cnt) {
  Trace::SetRelaxCount(cnt);
  if (origin_ != nullptr) {
    origin_->SetRelaxCount(cnt);
  }
  if (item_ != nullptr) {
    item_->SetRelaxCount(cnt);
  }
}

bool ItemTrace::operator==(const Trace &trace) {
  if (Trace::operator==(trace)) {
    const ItemTrace &t = (const ItemTrace &)trace;
    if (!item_ && !(t.item_)) {
      return true;
    } else if (item_ != nullptr && t.item_ != nullptr) {
      return *item_ == *(t.item_);
    }
  }
  return false;
}

void ItemTrace::Detach() {
  Trace::Detach();
  if (item_ != nullptr) {
    item_->Detach();
  }
}

bool ItemTrace::Support(TraceType tt) { return tt == TraceType::Item; }

AttrTrace::AttrTrace(PyObject *pObj, TracePtr pOrigin, std::string strAttr) : Trace(pObj, pOrigin), attr_(strAttr) {
  curType_ = TraceType::Attr;
  if (origin_ != nullptr && origin_->IsConst()) {
    is_const_ = true;
  }
  if (origin_ != nullptr) {
    depth_ = origin_->GetDepth() + 1;
  } else {
    depth_ = 1;
  }
}

std::string AttrTrace::GetAttribute() { return attr_; }

PyObject *AttrTrace::Retrieve(PTraceContext context, bool perf) {
  PyObject *ret = Trace::Retrieve(context, perf);
  if (ret != nullptr) {
    return ret;
  }
  if (origin_ != nullptr) {
    PyObject *pOrigin = origin_->Retrieve(context, perf);
    if (pOrigin != NULL) {
      TracePerf tp(this, perf, false);
      PyObject *itemName = PyUnicode_FromString(attr_.c_str());
      if (PyDict_CheckExact(pOrigin)) {
        ret = PyDict_GetItem(pOrigin, itemName);
        Py_INCREF(ret);
      } else {
        ret = PyObject_GetItem(pOrigin, itemName);
      }
      Py_DECREF(itemName);
      Py_DECREF(pOrigin);
      Cache(context, ret);
    }
  }
  return ret;
}

std::string AttrTrace::ToString(bool include_param) {
  if (strTrace_.size() > 0) {
    return strTrace_;
  }
  std::string ret;
  if (origin_ != nullptr) {
    std::string ori = origin_->ToString(include_param);
    ret = ori + "." + attr_;
  }
  ret = (is_const_ ? std::string("const:") : std::string("var:")) + ret;
  ret = std::regex_replace(ret, std::regex("(\n)"), "");
  strTrace_ = ret;
  return ret;
}

const InfoPack &AttrTrace::Info() {
  if (info_ == nullptr) {
    InfoPack info;
    info << uint8_t(curType_);
    info.Begin();
    info << (origin_ != nullptr);
    if (origin_ != nullptr) {
      auto ori = origin_->Info();
      info << ori;
    }
    info << attr_;
    info.End();
    info_ = std::make_shared<InfoPack>(info);
    info_->Update();
  }
  return *info_;
}

TracePtr AttrTrace::Optimize() {
  bool need_update = false;
  origin_ = OptimizeTrace(origin_, &need_update);
  if (need_update) {
    if (origin_ != nullptr && origin_->IsConst()) {
      is_const_ = true;
    }
    info_ = nullptr;
    strTrace_ = "";
    Info();
    return shared_from_this();
  } else {
    return nullptr;
  }
}

void AttrTrace::SetRelaxCount(int cnt) {
  Trace::SetRelaxCount(cnt);
  if (origin_ != nullptr) {
    origin_->SetRelaxCount(cnt);
  }
}

bool AttrTrace::operator==(const Trace &trace) {
  if (Trace::operator==(trace)) {
    const AttrTrace &t = (const AttrTrace &)trace;
    return attr_ == t.attr_;
  }
  return false;
}

bool AttrTrace::Support(TraceType tt) { return tt == TraceType::Attr; }

ConstTrace::ConstTrace(PyObject *pObj, int iIndex) : Trace(pObj, nullptr), index_(iIndex) {
  curType_ = TraceType::Const;
  originType_ = TraceType::Const;
  if (index_ == -1) {
    is_const_ = true;
  }
  depth_ = 1;
}

int ConstTrace::GetIndex() { return index_; }

PyObject *ConstTrace::Retrieve(PTraceContext context, bool perf) {
  PyObject *ret = Trace::Retrieve(context, perf);
  if (ret != nullptr) {
    return ret;
  }
  if (obj_ != NULL) {
    Py_INCREF(obj_);
    return obj_;
  }
  if (index_ >= 0 && index_ < PyTuple_GET_SIZE(context->f_code->co_consts)) {
    TracePerf tp(this, perf, false);
    ret = PyTuple_GET_ITEM(context->f_code->co_consts, index_);
    Py_INCREF(ret);
    Cache(context, ret);
  } else {
    ret = obj_;
    Py_INCREF(ret);
  }
  return ret;
}

std::string ConstTrace::ToString(bool include_param) {
  if (strTrace_.size() > 0) {
    return strTrace_;
  }
  std::string ret = "co_consts";
  if (index_ != -1) {
    ret = ret + "[" + std::to_string(index_) + "]";
  } else {
    ret = ret + "[-1](" + std::string(py::str(obj_)) + ")";
  }
  ret = (is_const_ ? std::string("const:") : std::string("var:")) + ret;
  ret = std::regex_replace(ret, std::regex("(\n)"), "");
  strTrace_ = ret;
  return ret;
}

const InfoPack &ConstTrace::Info() {
  if (info_ == nullptr) {
    InfoPack info;
    info << uint8_t(curType_);
    info.Begin();
    if (index_ != -1) {
      info << index_;
    } else {
      info << index_ << obj_;
    }
    info.End();
    info_ = std::make_shared<InfoPack>(info);
    info_->Update();
  }
  return *info_;
}

bool ConstTrace::operator==(const Trace &trace) {
  if (Trace::operator==(trace)) {
    const ConstTrace &t = (const ConstTrace &)trace;
    return index_ == t.index_;
  }
  return false;
}

void ConstTrace::Detach() {}

bool ConstTrace::Support(TraceType tt) { return tt == TraceType::Const; }

TypeTrace::TypeTrace(PyObject *pObj, TracePtr pOrigin) : Trace(pObj, pOrigin) {
  pType_ = Py_TYPE(pObj);
  curType_ = TraceType::Type;
  if (origin_ != nullptr && origin_->IsConst()) {
    is_const_ = true;
  }
  if (origin_ != nullptr) {
    depth_ = origin_->GetDepth() + 1;
  } else {
    depth_ = 1;
  }
}

PyTypeObject *TypeTrace::GetType() { return pType_; }

PyObject *TypeTrace::Retrieve(PTraceContext context, bool perf) {
  if (is_const_) {
    auto rt = reinterpret_cast<PyObject *>(pType_);
    Py_INCREF(rt);
    return rt;
  }
  PyObject *ret = Trace::Retrieve(context, perf);
  if (ret != nullptr) {
    return ret;
  }
  if (origin_ != NULL) {
    PyObject *pOrigin = origin_->Retrieve(context, perf);
    if (pOrigin != NULL) {
      TracePerf tp(this, perf, false);
      ret = reinterpret_cast<PyObject *>(Py_TYPE(pOrigin));
      Py_INCREF(ret);
      Py_DECREF(pOrigin);
      Cache(context, ret);
      return ret;
    }
  }
  return ret;
}

std::string TypeTrace::ToString(bool include_param) {
  if (strTrace_.size() > 0) {
    return strTrace_;
  }
  std::string ret = "type(type:";
  ret += std::string(py::str(reinterpret_cast<PyObject *>(pType_)));
  if (origin_ != NULL) {
    ret += ", origin:" + origin_->ToString(include_param);
  }
  ret += ")";
  ret = (is_const_ ? std::string("const:") : std::string("var:")) + ret;
  ret += std::regex_replace(ret, std::regex("(\n)"), "");
  strTrace_ = ret;
  return ret;
}

const InfoPack &TypeTrace::Info() {
  if (info_ == nullptr) {
    InfoPack info;
    info << uint8_t(curType_);
    info.Begin();
    info << reinterpret_cast<PyObject *>(pType_);
    info << (origin_ != nullptr);
    if (origin_ != nullptr) {
      info << origin_->Info();
    }
    info.End();
    info_ = std::make_shared<InfoPack>(info);
    info_->Update();
  }
  return *info_;
}

TracePtr TypeTrace::Optimize() {
  bool need_update = false;
  origin_ = OptimizeTrace(origin_, &need_update);
  if (need_update) {
    if (origin_ != nullptr && origin_->IsConst()) {
      is_const_ = true;
    }
    info_ = nullptr;
    strTrace_ = "";
    Info();
    return shared_from_this();
  } else {
    return nullptr;
  }
}

void TypeTrace::SetRelaxCount(int cnt) {
  Trace::SetRelaxCount(cnt);
  if (origin_ != nullptr) {
    origin_->SetRelaxCount(cnt);
  }
}

bool TypeTrace::operator==(const Trace &trace) {
  if (Trace::operator==(trace)) {
    const TypeTrace &t = (const TypeTrace &)trace;
    return pType_ == t.pType_;
  }
  return false;
}

void TypeTrace::Detach() {
  if (is_const_) {
    is_const_ = false;
    Trace::Detach();
    is_const_ = true;
  } else {
    Trace::Detach();
  }
}

bool TypeTrace::Support(TraceType tt) { return tt == TraceType::Type; }

static PyObject *RichCompare(PyObject *left, PyObject *right, int oparg) {
  bool invert;
  if (oparg >= Py_LT && oparg <= Py_GE) {
    return PyObject_RichCompare(left, right, oparg);
  } else if (Opcode(COMPARE_OP).CheckIsOp(oparg, &invert)) {
    auto ret = ((left == right) ^ invert) ? Py_True : Py_False;
    Py_INCREF(ret);
    return ret;
  } else if (Opcode(COMPARE_OP).CheckContainsOp(oparg, &invert)) {
    auto stat = PySequence_Contains(right, left);
    if (stat < 0) {
      return nullptr;
    }
    auto ret = (stat ^ invert) ? Py_True : Py_False;
    Py_INCREF(ret);
    return ret;
  }
  return nullptr;
}

static bool support_infer_primitive(PyObject *obj) {
  if (py::isinstance<mindspore::PrimitivePyAdapter>(obj) || py::isinstance<mindspore::PrimitiveFunctionAdapter>(obj)) {
    auto inst = mindspore::pijit::InferEngine::GetInstance();
    return inst->SupportInfer(obj);
  } else {
    return false;
  }
}

static bool support_create_primitive(PyObject *obj) {
  if (!obj || !PyType_Check(obj)) {
    return false;
  }
  py::object m = py::reinterpret_steal<py::object>(PyImport_GetModule(py::str("mindspore.ops").ptr()));
  if (!m.ptr()) {
    PyErr_Clear();
    return false;
  }
  py::object t = py::reinterpret_steal<py::object>(PyObject_GetAttrString(m.ptr(), "Primitive"));
  if (PyType_IsSubtype(reinterpret_cast<PyTypeObject *>(obj), reinterpret_cast<PyTypeObject *>((t.ptr())))) {
    return true;
  } else {
    return false;
  }
}

extern bool CheckJitConstexpr(const py::object &func);
extern bool CheckMSConstexpr(const py::object &func);
extern bool CheckBuiltinFuncOrMethod(const py::object &func);
static bool SupportCall(PyObject *func, const std::string &name) {
  /**
   * NOTE: exclude method type, it shouldn't be guard
   */
  static const std::set<PyTypeObject *> support_create_instance_type = {
    &PyComplex_Type, &PyMap_Type,       &PyBaseObject_Type, &PyRange_Type,   &PyZip_Type,  &PySlice_Type,
    &PyBool_Type,    &PyFloat_Type,     &PyLong_Type,       &PyType_Type,    &PyList_Type, &PyTuple_Type,
    &PySet_Type,     &PyFrozenSet_Type, &PyDict_Type,       &PyUnicode_Type, &PyEnum_Type, &PyMethod_Type,
  };
  if (PyType_CheckExact(func)) {
    if (IsMsClass(func)) {
      return true;
    }
    return support_create_instance_type.find(reinterpret_cast<PyTypeObject *>(func)) !=
           support_create_instance_type.end();
  }

  py::object handle = py::cast<py::object>(func);
  if (CheckJitConstexpr(handle)) {
    return true;
  }
  if (CheckMSConstexpr(handle)) {
    return true;
  }
  if (CheckBuiltinFuncOrMethod(handle)) {
    return true;
  }
  return support_infer_primitive(func) || support_create_primitive(func) || IsMsClass(func) ||
         (name.size() != 0 && PyDict_GetItemString(PyEval_GetBuiltins(), name.c_str()) == func);
}

static PyObject *DoCall(const std::vector<PyObject *> &params, int op, const std::string &name) {
  if (!Opcode(op).IsCall() || params.size() < 1) {
    return nullptr;
  }
  if (support_infer_primitive(params[0])) {
    std::vector<PyObject *> list;
    auto inst = mindspore::pijit::InferEngine::GetInstance();
    list.insert(list.begin(), params.begin() + 1, params.end());
    bool is_abstract = false;
    try {
      return inst->InferPrimitive(params[0], list, &is_abstract);
    } catch (py::error_already_set &e) {
      MS_LOG(ERROR) << "InferPrimitive failed " << std::endl << e.what();
    } catch (py::builtin_exception &e) {
      MS_LOG(ERROR) << "InferPrimitive failed " << std::endl << e.what();
    }
    return nullptr;
  }

  size_t nargs = (params.size() - 1);
  size_t kw_cnt;
  if (op == CALL_FUNCTION) {
    return PyObject_Vectorcall(params[0], params.data() + 1, nargs, NULL);
  } else if (op == CALL_FUNCTION_KW) {
    kw_cnt = PyTuple_GET_SIZE(params.back());
    return PyObject_Vectorcall(params[0], params.data() + 1, nargs - 1 - kw_cnt, params.back());
  } else if (op == CALL_FUNCTION_EX) {
    return PyObject_Call(params[0], params[1], params.size() > 2 ? params[2] : nullptr);
  }
  return nullptr;
}

using PyObjectArray = std::vector<PyObject *>;

static PyObject *CheckAndDoBinary(int op, const PyObjectArray &objs, binaryfunc pyfunc) {
  if (py::isinstance<mindspore::tensor::Tensor>(objs[0])) {
    auto arg0 = py::reinterpret_borrow<py::object>(objs[0]);
    auto arg1 = py::reinterpret_borrow<py::object>(objs[1]);
    auto res = pijit::AbstractTensor::Binary(op, arg0, arg1);
    return res.inc_ref().ptr();
  } else {
    return pyfunc(objs[0], objs[1]);
  }
}

using PythonBytecodeSupportCheckFunc = std::function<bool(int opargs, const PyObjectArray &objs)>;
using PythonBytecodeExecuteFunc = std::function<PyObject *(int opargs, const PyObjectArray &objs, PTraceContext ctx)>;
using PythonBytecodeFuncSet = std::pair<PythonBytecodeSupportCheckFunc, PythonBytecodeExecuteFunc>;
static bool ByteCodeUnsupported(int opargs, const PyObjectArray &objs) { return false; }
static bool ByteCodeSupported(int opargs, const PyObjectArray &objs) { return true; }
#define ByteCodeTest(bytecode)                                                                                       \
  [](int opargs, const PyObjectArray &objs) {                                                                        \
    return OptStrategy::MakeCalcStrategyByInputs(bytecode, opargs, objs) != OptStrategy::CalcKind::kCalcUnsupported; \
  }
#define ByteCodeCheck(bytecode, opargs, objs) \
  (OptStrategy::MakeCalcStrategyByInputs(bytecode, opargs, objs) == OptStrategy::CalcKind::kCalcValue)
static std::unordered_map<int, PythonBytecodeFuncSet> kBytecodeExecuter = {
  {POP_TOP, {ByteCodeUnsupported, nullptr}},
  {ROT_TWO, {ByteCodeUnsupported, nullptr}},
  {ROT_THREE, {ByteCodeUnsupported, nullptr}},
  {DUP_TOP, {ByteCodeUnsupported, nullptr}},
  {DUP_TOP_TWO, {ByteCodeUnsupported, nullptr}},
  {NOP, {ByteCodeUnsupported, nullptr}},
  {UNARY_POSITIVE,
   {ByteCodeTest(UNARY_POSITIVE),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(UNARY_POSITIVE, opargs, objs)) {
        return PyNumber_Positive(objs[0]);
      } else {
        Py_XINCREF(objs[0]);
        return objs[0];
      }
    }}},
  {UNARY_NEGATIVE,
   {ByteCodeTest(UNARY_NEGATIVE),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(UNARY_NEGATIVE, opargs, objs)) {
        return PyNumber_Negative(objs[0]);
      } else {
        Py_XINCREF(objs[0]);
        return objs[0];
      }
    }}},
  {UNARY_NOT,
   {ByteCodeTest(UNARY_NOT),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(UNARY_NOT, opargs, objs)) {
        auto ret = PyObject_IsTrue(objs[0]) ? Py_False : Py_True;
        Py_INCREF(ret);
        return ret;
      } else {
        Py_INCREF(Py_True);
        return Py_True;
      }
    }}},
  {UNARY_INVERT,
   {ByteCodeTest(UNARY_INVERT),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(UNARY_INVERT, opargs, objs)) {
        return PyNumber_Invert(objs[0]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {BINARY_MATRIX_MULTIPLY,
   {ByteCodeTest(BINARY_MATRIX_MULTIPLY),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(BINARY_MATRIX_MULTIPLY, opargs, objs)) {
        return PyNumber_MatrixMultiply(objs[0], objs[1]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {INPLACE_MATRIX_MULTIPLY,
   {ByteCodeTest(INPLACE_MATRIX_MULTIPLY),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(INPLACE_MATRIX_MULTIPLY, opargs, objs)) {
        return PyNumber_InPlaceMatrixMultiply(objs[0], objs[1]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {BINARY_POWER,
   {ByteCodeTest(BINARY_POWER),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(BINARY_POWER, opargs, objs)) {
        return PyNumber_Power(objs[0], objs[1], Py_None);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {BINARY_MULTIPLY,
   {ByteCodeTest(BINARY_MULTIPLY),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(BINARY_MULTIPLY, opargs, objs)) {
        return CheckAndDoBinary(BINARY_MULTIPLY, objs, PyNumber_Multiply);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {BINARY_MODULO,
   {ByteCodeTest(BINARY_MODULO),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(BINARY_MODULO, opargs, objs)) {
        if (PyUnicode_CheckExact(objs[0]) && (!PyUnicode_Check(objs[1]) || PyUnicode_CheckExact(objs[1]))) {
          return PyUnicode_Format(objs[0], objs[1]);
        } else {
          return PyNumber_Remainder(objs[0], objs[1]);
        }
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {BINARY_ADD,
   {[](int opargs, const PyObjectArray &objs) -> bool {
      return (!PyUnicode_CheckExact(objs[0]) || !PyUnicode_CheckExact(objs[1])) &&
             OptStrategy::MakeCalcStrategyByInputs(BINARY_ADD, opargs, objs) != OptStrategy::CalcKind::kCalcUnsupported;
    },
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(BINARY_ADD, opargs, objs)) {
        return CheckAndDoBinary(BINARY_ADD, objs, PyNumber_Add);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {BINARY_SUBTRACT,
   {ByteCodeTest(BINARY_SUBTRACT),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(BINARY_SUBTRACT, opargs, objs)) {
        return CheckAndDoBinary(BINARY_SUBTRACT, objs, PyNumber_Subtract);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {BINARY_SUBSCR,
   {ByteCodeTest(BINARY_SUBSCR),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      return PyObject_GetItem(objs[0], objs[1]);
    }}},
  {BINARY_FLOOR_DIVIDE,
   {ByteCodeTest(BINARY_FLOOR_DIVIDE),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(BINARY_FLOOR_DIVIDE, opargs, objs)) {
        return CheckAndDoBinary(BINARY_FLOOR_DIVIDE, objs, PyNumber_FloorDivide);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {BINARY_TRUE_DIVIDE,
   {ByteCodeTest(BINARY_TRUE_DIVIDE),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(BINARY_TRUE_DIVIDE, opargs, objs)) {
        return CheckAndDoBinary(BINARY_TRUE_DIVIDE, objs, PyNumber_TrueDivide);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {INPLACE_FLOOR_DIVIDE,
   {ByteCodeTest(INPLACE_FLOOR_DIVIDE),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(INPLACE_FLOOR_DIVIDE, opargs, objs)) {
        return PyNumber_InPlaceFloorDivide(objs[0], objs[1]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {INPLACE_TRUE_DIVIDE,
   {ByteCodeTest(INPLACE_TRUE_DIVIDE),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(INPLACE_TRUE_DIVIDE, opargs, objs)) {
        return PyNumber_InPlaceTrueDivide(objs[0], objs[1]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {GET_AITER, {ByteCodeUnsupported, nullptr}},
  {GET_ANEXT, {ByteCodeUnsupported, nullptr}},
  {BEFORE_ASYNC_WITH, {ByteCodeUnsupported, nullptr}},
  {INPLACE_ADD,
   {[](int opargs, const PyObjectArray &objs) -> bool {
      return (!PyUnicode_CheckExact(objs[0]) || !PyUnicode_CheckExact(objs[1])) &&
             OptStrategy::MakeCalcStrategyByInputs(INPLACE_ADD, opargs, objs) !=
               OptStrategy::CalcKind::kCalcUnsupported;
    },
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(INPLACE_ADD, opargs, objs)) {
        return PyNumber_InPlaceAdd(objs[0], objs[1]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {INPLACE_SUBTRACT,
   {ByteCodeTest(INPLACE_SUBTRACT),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(INPLACE_SUBTRACT, opargs, objs)) {
        return PyNumber_InPlaceSubtract(objs[0], objs[1]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {INPLACE_MULTIPLY,
   {ByteCodeTest(INPLACE_MULTIPLY),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(INPLACE_MULTIPLY, opargs, objs)) {
        return PyNumber_InPlaceMultiply(objs[0], objs[1]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {INPLACE_MODULO,
   {ByteCodeTest(INPLACE_MODULO),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(INPLACE_MODULO, opargs, objs)) {
        return PyNumber_InPlaceRemainder(objs[0], objs[1]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {STORE_SUBSCR, {ByteCodeUnsupported, nullptr}},
  {DELETE_SUBSCR, {ByteCodeUnsupported, nullptr}},
  {BINARY_LSHIFT,
   {ByteCodeTest(BINARY_LSHIFT),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(BINARY_LSHIFT, opargs, objs)) {
        return PyNumber_Lshift(objs[0], objs[1]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {BINARY_RSHIFT,
   {ByteCodeTest(BINARY_RSHIFT),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(BINARY_RSHIFT, opargs, objs)) {
        return PyNumber_Rshift(objs[0], objs[1]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {BINARY_AND,
   {ByteCodeTest(BINARY_AND),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(BINARY_AND, opargs, objs)) {
        return PyNumber_And(objs[0], objs[1]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {BINARY_XOR,
   {ByteCodeTest(BINARY_XOR),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(BINARY_XOR, opargs, objs)) {
        return PyNumber_Xor(objs[0], objs[1]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {BINARY_OR,
   {ByteCodeTest(BINARY_OR),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(BINARY_OR, opargs, objs)) {
        return PyNumber_Or(objs[0], objs[1]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {INPLACE_POWER,
   {ByteCodeTest(INPLACE_POWER),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(INPLACE_POWER, opargs, objs)) {
        return PyNumber_InPlacePower(objs[0], objs[1], Py_None);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {GET_ITER,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * { return PyObject_GetIter(objs[0]); }}},
  {GET_YIELD_FROM_ITER,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      PyObject *iterable = objs[0];
      if (PyCoro_CheckExact(iterable)) {
        if (!(ctx->f_code->co_flags & (CO_COROUTINE | CO_ITERABLE_COROUTINE))) {
          return nullptr;
        }
      } else if (!PyGen_CheckExact(iterable)) {
        return PyObject_GetIter(iterable);
      }
      Py_INCREF(iterable);
      return iterable;
    }}},
  {PRINT_EXPR, {ByteCodeUnsupported, nullptr}},
  {LOAD_BUILD_CLASS, {ByteCodeUnsupported, nullptr}},
  {YIELD_FROM, {ByteCodeUnsupported, nullptr}},
  {GET_AWAITABLE, {ByteCodeUnsupported, nullptr}},
  {INPLACE_LSHIFT,
   {ByteCodeTest(INPLACE_LSHIFT),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(INPLACE_LSHIFT, opargs, objs)) {
        return PyNumber_InPlaceLshift(objs[0], objs[1]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {INPLACE_RSHIFT,
   {ByteCodeTest(INPLACE_RSHIFT),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(INPLACE_RSHIFT, opargs, objs)) {
        return PyNumber_InPlaceRshift(objs[0], objs[1]);
      } else {
        Py_INCREF(objs[0]);
        return objs[0];
      }
    }}},
  {INPLACE_AND,
   {ByteCodeTest(INPLACE_AND),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     if (ByteCodeCheck(INPLACE_AND, opargs, objs)) {
                                                                       return PyNumber_InPlaceAnd(objs[0], objs[1]);
                                                                     } else {
                                                                       Py_INCREF(objs[0]);
                                                                       return objs[0];
                                                                     }
                                                                   }}},
  {INPLACE_XOR,
   {ByteCodeTest(INPLACE_XOR),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     if (ByteCodeCheck(INPLACE_XOR, opargs, objs)) {
                                                                       return PyNumber_InPlaceXor(objs[0], objs[1]);
                                                                     } else {
                                                                       Py_INCREF(objs[0]);
                                                                       return objs[0];
                                                                     }
                                                                   }}},
  {INPLACE_OR,
   {ByteCodeTest(INPLACE_OR),
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     if (ByteCodeCheck(INPLACE_OR, opargs, objs)) {
                                                                       return PyNumber_InPlaceOr(objs[0], objs[1]);
                                                                     } else {
                                                                       Py_INCREF(objs[0]);
                                                                       return objs[0];
                                                                     }
                                                                   }}},
  {RETURN_VALUE, {ByteCodeUnsupported, nullptr}},
  {IMPORT_STAR, {ByteCodeUnsupported, nullptr}},
  {SETUP_ANNOTATIONS, {ByteCodeUnsupported, nullptr}},
  {YIELD_VALUE, {ByteCodeUnsupported, nullptr}},
  {POP_BLOCK, {ByteCodeUnsupported, nullptr}},
  {POP_EXCEPT, {ByteCodeUnsupported, nullptr}},
  {STORE_NAME, {ByteCodeUnsupported, nullptr}},
  {DELETE_NAME, {ByteCodeUnsupported, nullptr}},
  {UNPACK_SEQUENCE, {ByteCodeUnsupported, nullptr}},
  {FOR_ITER, {ByteCodeUnsupported, nullptr}},
  {UNPACK_EX, {ByteCodeUnsupported, nullptr}},
  {STORE_ATTR, {ByteCodeUnsupported, nullptr}},
  {DELETE_ATTR, {ByteCodeUnsupported, nullptr}},
  {STORE_GLOBAL, {ByteCodeUnsupported, nullptr}},
  {DELETE_GLOBAL, {ByteCodeUnsupported, nullptr}},
  {LOAD_CONST, {ByteCodeSupported, nullptr}},
  {LOAD_NAME, {ByteCodeSupported, nullptr}},
  {BUILD_TUPLE,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PyObject *tup = PyTuple_New(opargs);
                                                                     while (--opargs >= 0) {
                                                                       Py_INCREF(objs[opargs]);
                                                                       PyTuple_SET_ITEM(tup, opargs, objs[opargs]);
                                                                     }
                                                                     return tup;
                                                                   }}},
  {BUILD_LIST,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PyObject *list = PyList_New(opargs);
                                                                     while (--opargs >= 0) {
                                                                       Py_INCREF(objs[opargs]);
                                                                       PyList_SET_ITEM(list, opargs, objs[opargs]);
                                                                     }
                                                                     return list;
                                                                   }}},
  {BUILD_SET,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PyObject *set = PySet_New(NULL);
                                                                     for (int i = opargs; i > 0; i--) {
                                                                       PySet_Add(set, objs[opargs - i]);
                                                                     }
                                                                     return set;
                                                                   }}},
  {BUILD_MAP,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PyObject *map =
                                                                       _PyDict_NewPresized((Py_ssize_t)opargs);
                                                                     for (Py_ssize_t i = opargs; i > 0; i--) {
                                                                       PyObject *key = objs[2 * (opargs - i)];
                                                                       PyObject *value = objs[2 * (opargs - i) + 1];
                                                                       PyDict_SetItem(map, key, value);
                                                                     }
                                                                     return map;
                                                                   }}},
  {LOAD_ATTR, {ByteCodeSupported, nullptr}},
  {COMPARE_OP,
   {[](int opargs, const PyObjectArray &objs) {
      return OptStrategy::MakeCalcStrategyByInputs(COMPARE_OP, opargs, objs) != OptStrategy::CalcKind::kCalcUnsupported;
    },
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      if (ByteCodeCheck(COMPARE_OP, opargs, objs)) {
        return RichCompare(objs[0], objs[1], opargs);
      } else {
        Py_INCREF(Py_True);
        return Py_True;
      }
    }}},
  {IMPORT_NAME, {ByteCodeUnsupported, nullptr}},
  {IMPORT_FROM, {ByteCodeUnsupported, nullptr}},
  {JUMP_FORWARD, {ByteCodeUnsupported, nullptr}},
  {JUMP_IF_FALSE_OR_POP, {ByteCodeUnsupported, nullptr}},
  {JUMP_IF_TRUE_OR_POP, {ByteCodeUnsupported, nullptr}},
  {JUMP_ABSOLUTE, {ByteCodeUnsupported, nullptr}},
  {POP_JUMP_IF_FALSE, {ByteCodeUnsupported, nullptr}},
  {POP_JUMP_IF_TRUE, {ByteCodeUnsupported, nullptr}},
  {LOAD_GLOBAL, {ByteCodeSupported, nullptr}},
  {SETUP_FINALLY, {ByteCodeUnsupported, nullptr}},
  {LOAD_FAST, {ByteCodeUnsupported, nullptr}},
  {STORE_FAST, {ByteCodeUnsupported, nullptr}},
  {DELETE_FAST, {ByteCodeUnsupported, nullptr}},
  {RAISE_VARARGS, {ByteCodeUnsupported, nullptr}},
  {CALL_FUNCTION, {ByteCodeSupported, nullptr}},
  {MAKE_FUNCTION, {ByteCodeUnsupported, nullptr}},
  {BUILD_SLICE,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PyObject *start;
                                                                     PyObject *stop;
                                                                     PyObject *step;
                                                                     if (opargs == 3)
                                                                       step = objs[2];
                                                                     else
                                                                       step = nullptr;
                                                                     stop = objs[1];
                                                                     start = objs[0];
                                                                     return PySlice_New(start, stop, step);
                                                                   }}},
  {LOAD_CLOSURE, {ByteCodeSupported, nullptr}},
  {LOAD_DEREF, {ByteCodeSupported, nullptr}},
  {STORE_DEREF, {ByteCodeUnsupported, nullptr}},
  {DELETE_DEREF, {ByteCodeUnsupported, nullptr}},
  {CALL_FUNCTION_KW, {ByteCodeSupported, nullptr}},
  {CALL_FUNCTION_EX, {ByteCodeSupported, nullptr}},
  {SETUP_WITH, {ByteCodeUnsupported, nullptr}},
  {EXTENDED_ARG, {ByteCodeUnsupported, nullptr}},
  {LIST_APPEND,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PyList_Append(objs[0], objs[1]);
                                                                     Py_INCREF(objs[0]);
                                                                     return objs[0];
                                                                   }}},
  {SET_ADD,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PySet_Add(objs[0], objs[1]);
                                                                     Py_INCREF(objs[0]);
                                                                     return objs[0];
                                                                   }}},
  {MAP_ADD,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PyDict_SetItem(objs[0], objs[2], objs[1]);
                                                                     Py_INCREF(objs[0]);
                                                                     return objs[0];
                                                                   }}},
  {LOAD_CLASSDEREF, {ByteCodeSupported, nullptr}},
  {SETUP_ASYNC_WITH, {ByteCodeUnsupported, nullptr}},
  {FORMAT_VALUE, {ByteCodeUnsupported, nullptr}},
  {BUILD_CONST_KEY_MAP,
   {[](int opargs, const PyObjectArray &objs) -> bool {
      return PyTuple_CheckExact(objs[opargs]) && PyTuple_GET_SIZE(objs[opargs]) == (Py_ssize_t)opargs;
    },
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * {
      PyObject *keys = objs[opargs];
      PyObject *map = _PyDict_NewPresized((Py_ssize_t)opargs);
      for (Py_ssize_t i = opargs; i > 0; i--) {
        PyObject *key = PyTuple_GET_ITEM(keys, opargs - i);
        PyObject *value = objs[opargs - i];
        PyDict_SetItem(map, key, value);
      }
      return map;
    }}},
  {BUILD_STRING,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PyObject *empty = PyUnicode_New(0, 0);
                                                                     PyObject *str =
                                                                       _PyUnicode_JoinArray(empty, objs.data(), opargs);
                                                                     Py_DECREF(empty);
                                                                     return str;
                                                                   }}},
  {LOAD_METHOD, {ByteCodeUnsupported, nullptr}},
  {CALL_METHOD, {ByteCodeUnsupported, nullptr}},
  {ROT_FOUR, {ByteCodeUnsupported, nullptr}},
  {RERAISE, {ByteCodeUnsupported, nullptr}},
  {WITH_EXCEPT_START, {ByteCodeUnsupported, nullptr}},
  {END_ASYNC_FOR, {ByteCodeUnsupported, nullptr}},
  {LOAD_ASSERTION_ERROR, {ByteCodeUnsupported, nullptr}},
  {LIST_TO_TUPLE,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject * { return PyList_AsTuple(objs[0]); }}},
  {IS_OP,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     auto ret = (objs[0] == objs[1]) ^ opargs
                                                                                  ? Py_True
                                                                                  : Py_False;
                                                                     Py_INCREF(ret);
                                                                     return ret;
                                                                   }}},
  {CONTAINS_OP,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     auto ret =
                                                                       (PySequence_Contains(objs[1], objs[0]) ^ opargs)
                                                                         ? Py_True
                                                                         : Py_False;
                                                                     Py_INCREF(ret);
                                                                     return ret;
                                                                   }}},
  {JUMP_IF_NOT_EXC_MATCH, {ByteCodeUnsupported, nullptr}},
  {LIST_EXTEND,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     _PyList_Extend(
                                                                       reinterpret_cast<PyListObject *>(objs[0]),
                                                                       objs[1]);
                                                                     Py_INCREF(objs[0]);
                                                                     return objs[0];
                                                                   }}},
  {SET_UPDATE,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     _PySet_Update(objs[0], objs[1]);
                                                                     Py_INCREF(objs[0]);
                                                                     return objs[0];
                                                                   }}},
  {DICT_MERGE,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     _PyDict_MergeEx(objs[0], objs[1], 2);
                                                                     return objs[0];
                                                                   }}},
  {DICT_UPDATE,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PyDict_Update(objs[0], objs[1]);
                                                                     return objs[0];
                                                                   }}},
  {BREAK_LOOP, {ByteCodeUnsupported, nullptr}},
  {WITH_CLEANUP_START, {ByteCodeUnsupported, nullptr}},
  {WITH_CLEANUP_FINISH, {ByteCodeUnsupported, nullptr}},
  {END_FINALLY, {ByteCodeUnsupported, nullptr}},
  {CONTINUE_LOOP, {ByteCodeUnsupported, nullptr}},
  {SETUP_LOOP, {ByteCodeUnsupported, nullptr}},
  {SETUP_EXCEPT, {ByteCodeUnsupported, nullptr}},
  {BUILD_LIST_UNPACK,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PyObject *sum = PyList_New(0);
                                                                     for (int i = opargs; i > 0; i--) {
                                                                       auto none_val = _PyList_Extend(
                                                                         reinterpret_cast<PyListObject *>(sum),
                                                                         objs[opargs - i]);
                                                                       Py_DECREF(none_val);
                                                                     }
                                                                     return sum;
                                                                   }}},
  {BUILD_MAP_UNPACK,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PyObject *sum = PyDict_New();
                                                                     for (int i = opargs; i > 0; i--) {
                                                                       PyDict_Update(sum, objs[opargs - i]);
                                                                     }
                                                                     return sum;
                                                                   }}},
  {BUILD_MAP_UNPACK_WITH_CALL,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PyObject *sum = PyDict_New();
                                                                     for (int i = opargs; i > 0; i--) {
                                                                       _PyDict_MergeEx(sum, objs[opargs - i], 2);
                                                                     }
                                                                     return sum;
                                                                   }}},
  {BUILD_TUPLE_UNPACK,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PyObject *sum = PyList_New(0);
                                                                     for (int i = opargs; i > 0; i--) {
                                                                       auto none_val = _PyList_Extend(
                                                                         reinterpret_cast<PyListObject *>(sum),
                                                                         objs[opargs - i]);
                                                                       Py_DECREF(none_val);
                                                                     }
                                                                     auto ret = PyList_AsTuple(sum);
                                                                     Py_DECREF(sum);
                                                                     return ret;
                                                                   }}},
  {BUILD_SET_UNPACK,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PyObject *sum = PySet_New(NULL);
                                                                     for (int i = opargs; i > 0; i--) {
                                                                       _PySet_Update(sum, objs[opargs - i]);
                                                                     }
                                                                     return sum;
                                                                   }}},
  {BUILD_TUPLE_UNPACK_WITH_CALL,
   {ByteCodeSupported,
    [](int opargs, const PyObjectArray &objs, PTraceContext ctx) -> PyObject
                                                                   * {
                                                                     PyObject *sum = PyList_New(0);
                                                                     for (int i = opargs; i > 0; i--) {
                                                                       auto none_val = _PyList_Extend(
                                                                         reinterpret_cast<PyListObject *>(sum),
                                                                         objs[opargs - i]);
                                                                       Py_DECREF(none_val);
                                                                     }
                                                                     auto ret = PyList_AsTuple(sum);
                                                                     Py_DECREF(sum);
                                                                     return ret;
                                                                   }}},
};

OpTrace::OpTrace(PyObject *obj, int opcode, int opargs, TraceVector params, std::string name)
    : Trace(obj, nullptr), opcode_(opcode), opargs_(opargs), params_(params), name_(name) {
  curType_ = TraceType::Operation;
  if (!std::any_of(params.begin(), params.end(), [](const TracePtr &item) { return !item->IsConst(); })) {
    is_const_ = true;
  } else if (name.find(kIsSeqValUnknown) != std::string::npos || name.find(kIsSeqShapeUnknown) != std::string::npos) {
    is_const_ = true;
  } else if (kPIJitConfigDefault.getIntConfig(GraphJitConfig::kGuardRelaxCount) > 0 && opcode == LOAD_ATTR &&
             name_ == kFuncName) {
    is_const_ = true;
  }
  depth_ = std::accumulate(params.begin(), params.end(), 1, [](int depth, const TracePtr &i) {
    int d = i->GetDepth() + 1;
    if (d > depth) {
      return d;
    } else {
      return depth;
    }
  });
  CheckSpecialize();
}

int OpTrace::GetOpCode() { return opcode_; }

int OpTrace::GetOpArgs() { return opargs_; }

TracePtr OpTrace::GetParam(size_t idx) {
  if (params_.size() > idx) {
    return params_[idx];
  } else {
    return nullptr;
  }
}

size_t OpTrace::GetParamCount() { return params_.size(); }

std::string OpTrace::GetName() { return name_; }

PyObject *OpTrace::Retrieve(PTraceContext context, bool perf) {
  PyObject *ret = Trace::Retrieve(context, perf);
  if (ret != nullptr) {
    return ret;
  }
  std::vector<PyObject *> params;
  auto iter = std::find_if(params_.begin(), params_.end(), [&params, &context, perf](const TracePtr &p) {
    auto param = p->Retrieve(context, perf);
    if (param == nullptr) {
      return true;
    }
    if (py::isinstance<mindspore::tensor::Tensor>(param)) {
      mindspore::tensor::TensorPtr tensor_ptr = py::cast<mindspore::tensor::TensorPtr>(param);
      if (OptStrategy::MakeCalcStrategyByShape(tensor_ptr->shape()) == OptStrategy::CalcKind::kCalcValue) {
        tensor_ptr->data_sync(true);
      }
    }
    params.push_back(param);
    return params.back() == nullptr;
  });
  if (iter != params_.end()) {
    MS_LOG(DEBUG) << "Guard Check Retrieve fail for " + (*iter)->ToString();
    std::for_each(params.begin(), params.end(), [](PyObject *p) { Py_XDECREF(p); });
    return nullptr;
  }
  TracePerf tp(this, perf, false);
  if (kBytecodeExecuter.find(opcode_) != kBytecodeExecuter.end() && kBytecodeExecuter[opcode_].first(opargs_, params) &&
      kBytecodeExecuter[opcode_].second != nullptr) {
    ret = kBytecodeExecuter[opcode_].second(opargs_, params, context);
  } else if (opcode_ == LOAD_ATTR) {
    MS_EXCEPTION_IF_CHECK_FAIL(name_.size(), "check trace");
    ret = PyObject_GetAttrString(params[0], name_.c_str());
  } else if (opcode_ == CALL_FUNCTION || opcode_ == CALL_FUNCTION_EX || opcode_ == CALL_FUNCTION_KW) {
    ret = DoCall(params, opcode_, name_);
  }
  for (auto p : params) {
    Py_DECREF(p);
  }
  if (PyErr_Occurred()) {
    PyErr_Clear();
  }
  Cache(context, ret);
  return ret;
}

std::string OpTrace::ToString(bool include_param) {
  if (strTrace_.size() > 0) {
    return strTrace_;
  }
  std::string ret = "operation ";
  ret += Opcode(opcode_).name();
  ret += "(arg:";
  ret += std::to_string(opargs_);
  if (name_.size() != 0 || params_.size() > 0) {
    ret += ",";
  }
  if (name_.size() != 0) {
    ret += std::string("name:") + name_;
    if (params_.size() > 0) {
      ret += ",";
    }
  }
  if (include_param && params_.size() > 0) {
    for (auto t : params_) {
      ret += t->ToString(include_param) + ",";
    }
    ret = ret.substr(0, ret.size() - 1);
  }
  ret = ret + ")";
  ret = (is_const_ ? std::string("const:") : std::string("var:")) + ret;
  ret = std::regex_replace(ret, std::regex("(\n)"), "");
  strTrace_ = ret;
  return ret;
}

std::string OpTrace::FormatString(std::map<Trace *, size_t> *cache) {
  std::stringstream s;
  std::stringstream params_str;
  params_str << "(";
  for (auto i : params_) {
    if (cache->find(i.get()) == cache->end()) {
      s << i->FormatString(cache) << std::endl;
    }
    params_str << "%" << (cache->find(i.get())->second) << ", ";
  }
  params_str << ")";

  cache->insert(std::make_pair(this, cache->size()));
  s << "%" << cache->find(this)->second << " = operation " << Opcode(opcode_).name() << " " << opargs_;
  if (!name_.empty()) {
    s << ", name: " << name_;
  }
  s << ": " << params_str.str();
  return s.str();
}

const InfoPack &OpTrace::Info() {
  if (info_ == nullptr) {
    InfoPack info;
    info << uint8_t(curType_);
    info.Begin();
    info << opcode_;
    info << opargs_;
    info << (name_.size() != 0);
    if (name_.size() != 0) {
      info << name_;
    }
    info << (params_.size() != 0);
    if (params_.size() > 0) {
      for (auto t : params_) {
        info << t->Info();
      }
    }
    info.End();
    info_ = std::make_shared<InfoPack>(info);
    info_->Update();
  }
  return *info_;
}

TracePtr OpTrace::RemoveCastDuplicatePatternPass() {
  OpTracePtr cast_op;
  TracePtr next_op;
  TracePtr this_op;
  TracePtr ret_op;
  if (opcode_ != CALL_FUNCTION || !IsCastFunc(name_) ||
      (cast_op = CastTrace<OpTrace>(GetParam(kParamIndexTwo))) == nullptr || !IsCastFunc(cast_op->GetName()) ||
      (next_op = cast_op->GetParam(kParamIndexTwo)) == nullptr) {
    return nullptr;
  }
  // remove duplicate cast or contrary cast
  if (name_ == cast_op->GetName()) {
    this_op = cast_op;
    ret_op = cast_op->Optimize();
  } else {
    this_op = next_op;
    ret_op = next_op->Optimize();
  }
  if (ret_op != nullptr) {
    return ret_op;
  } else {
    return this_op;
  }
}

TracePtr OpTrace::RemovePrimOutIsTensorPass() {
  RootTracePtr global_op;
  OpTracePtr call_op;
  TracePtr param_op;
  if (opcode_ != CALL_FUNCTION || !(name_ == kIsInstance) ||
      (global_op = CastTrace<RootTrace>(GetParam(kParamIndexThree))) == nullptr ||
      global_op->GetTraceType() != TraceType::Global ||
      (call_op = CastOpTrace(GetParam(kParamIndexTwo), CALL_FUNCTION)) == nullptr ||
      (param_op = call_op->GetParam(kParamIndexOne)) == nullptr) {
    return nullptr;
  }
  int idx;
  std::string name;
  std::string module_name;
  global_op->GetParam(&idx, &name, &module_name);
  // isinstance(cast_to_mstensor(...) or Primitive) should be Tensor
  if ((name == kTensorName) && ((call_op->GetName() == kCastToMSTensor) ||
                                (CastTrace<ConstTrace>(param_op) != nullptr &&
                                 py::isinstance<mindspore::PrimitivePyAdapter>(param_op->GetObject())))) {
    is_const_ = true;
    if (obj_ == nullptr) {
      obj_ = Py_True;
      Py_INCREF(obj_);
    }
    return shared_from_this();
  }
  return nullptr;
}

TracePtr OpTrace::RemoveCastPass() {
  TracePtr next_op;
  if (opcode_ == CALL_FUNCTION && name_ == kCastToMSTensor && (next_op = GetParam(kParamIndexTwo)) != nullptr) {
    auto new_ret = next_op->Optimize();
    if (new_ret != nullptr) {
      return new_ret;
    } else {
      return next_op;
    }
  }
  return nullptr;
}

TracePtr OpTrace::RemoveEmptyTensorPass() {
  OpTracePtr subscr_op;
  OpTracePtr loadattr_op;
  ConstTracePtr const_op;
  ConstTracePtr const2_op;
  if (opcode_ != COMPARE_OP || params_.size() < kParamCountTwo) {
    return nullptr;
  }
  for (size_t idx = 0; idx < kParamCountTwo; ++idx) {
    TracePtr tmp = GetParam(idx);
    if (subscr_op == nullptr) {
      subscr_op = CastOpTrace(tmp, BINARY_SUBSCR);
    }
    if (const_op == nullptr) {
      const_op = CastConstTrace(tmp);
    }
  }
  if (subscr_op == nullptr || const_op == nullptr ||
      (const2_op = CastConstTrace(subscr_op->GetParam(kParamIndexTwo))) == nullptr ||
      (loadattr_op = CastOpTrace(subscr_op->GetParam(kParamIndexOne), kShapeName)) == nullptr ||
      CastOpTrace(loadattr_op->GetParam(kParamIndexOne), kCastToAdapterTensor) == nullptr) {
    return nullptr;
  }
  // make judgement shape[0] == 0 as const
  auto c1 = const_op->GetObject();
  auto c2 = const2_op->GetObject();
  if (!PyLong_CheckExact(c1) || !PyLong_CheckExact(c2)) {
    return nullptr;
  }
  auto v1 = _PyLong_AsInt(c1);
  auto v2 = _PyLong_AsInt(c2);
  if (v1 == 0 && v2 == 0) {
    is_const_ = true;
    return shared_from_this();
  }
  return nullptr;
}

void OpTrace::JudgeDTypeChangePass() {
  if (opcode_ != COMPARE_OP) {
    return;
  }
  for (size_t i = 0; i < kParamCountTwo; ++i) {
    OpTracePtr trace = CastOpTrace(GetParam(i), CALL_FUNCTION);
    ConstTracePtr const_op = trace ? CastConstTrace(trace->GetParam(kParamIndexOne)) : nullptr;
    PyObject *const_param = const_op ? const_op->GetObject() : nullptr;
    if (trace != nullptr && const_op != nullptr && const_param != nullptr &&
        py::isinstance<mindspore::PrimitivePyAdapter>(const_param) &&
        py::cast<mindspore::PrimitivePyAdapterPtr>(const_param)->name() == kDTypePrimName) {
      // Compare for output of DType primitive
      continue;
    } else if ((trace = CastOpTrace(GetParam(i), LOAD_ATTR)) != nullptr && trace->GetName() == kDTypeAttrName) {
      // Compare for attribute dtype
      continue;
    }
    return;
  }
  // data type comparison should be kept as const
  EnableRelax();
}

void OpTrace::JudgeDTypeScopePass() {
  if (opcode_ != CONTAINS_OP) {
    return;
  }
  OpTracePtr trace;
  if ((trace = CastOpTrace(GetParam(kParamIndexOne), LOAD_ATTR)) != nullptr && trace->GetName() == kDTypeAttrName) {
    // data type to check whether to be contained should be const
    EnableRelax();
  }
}

void OpTrace::JudgeDTypeTensorAttrPass() {
  if (opcode_ != CALL_FUNCTION) {
    return;
  }
  RootTracePtr global_op;
  OpTracePtr call_op;
  if (params_.size() < kParamCountTwo || (global_op = CastTrace<RootTrace>(params_[kParamIndexOne])) == nullptr ||
      (call_op = CastOpTrace(params_[kParamIndexTwo], BINARY_SUBSCR)) == nullptr) {
    return;
  }
  int idx;
  std::string name;
  std::string module_name;
  global_op->GetParam(&idx, &name, &module_name);
  auto tsr = call_op->GetObject();
  if (tsr == nullptr) {
    return;
  }
  std::string type_name = std::string(py::str(reinterpret_cast<PyObject *>(Py_TYPE(tsr))));
  if (name == kDType_AttrName && module_name.find("mindspore") == 0 &&
      type_name.find(kTensorName) != std::string::npos) {
    EnableRelax();
  }
}

void OpTrace::JudgeCodeChangePass() {
  if (opcode_ != LOAD_ATTR || params_.size() < kParamCountOne || name_ != kCodeName) {
    return;
  }
  if (params_[kParamIndexOne]->IsConst()) {
    EnableRelax();
  } else {
    auto p1 = CastOpTrace(params_[kParamIndexOne], LOAD_ATTR);
    if (p1 == nullptr) {
      return;
    }
    auto p2 = p1->GetParam(kParamIndexOne)->GetObject();
    std::string type_name;
    if (p2 != nullptr) {
      type_name = std::string(py::str(reinterpret_cast<PyObject *>(Py_TYPE(p2))));
    }
    if (type_name.find(kMindTorchFlag) != std::string::npos) {
      EnableRelax();
    }
  }
}

void OpTrace::JudgeTrainFlagPass() {
  if (opcode_ != LOAD_ATTR || params_.size() < kParamCountOne) {
    return;
  }
  if (name_ == kTrainingFlag) {
    // training flag shouldn't be changed frequently
    EnableRelax();
  }
}

void OpTrace::JudgeCompareConstPass() {
  if (RelaxEnabled()) {
    return;
  }
  if (opcode_ != COMPARE_OP || params_.size() < kParamCountTwo) {
    return;
  }
  if (params_[kParamIndexOne]->GetObject() == nullptr || params_[kParamIndexTwo]->GetObject() == nullptr) {
    return;
  }
  EnableRelax();
}

void OpTrace::JudgeContainsConstPass() {
  if (RelaxEnabled()) {
    return;
  }
  if (opcode_ != CONTAINS_OP || params_.size() < kParamCountTwo) {
    return;
  }
  if (params_[kParamIndexOne]->GetObject() == nullptr || params_[kParamIndexTwo]->GetObject() == nullptr) {
    return;
  }
  EnableRelax();
}

void OpTrace::JudgeInplaceAddConstPass() {
  if (RelaxEnabled()) {
    return;
  }
  if (opcode_ != INPLACE_ADD || params_.size() < kParamCountTwo) {
    return;
  }
  if (params_[kParamIndexOne]->GetObject() == nullptr || params_[kParamIndexTwo]->GetObject() == nullptr) {
    return;
  }
  EnableRelax();
}

void OpTrace::JudgeIsConstPass() {
  if (RelaxEnabled()) {
    return;
  }
  if (opcode_ != IS_OP || params_.size() < kParamCountTwo) {
    return;
  }
  if (params_[kParamIndexOne]->GetObject() == nullptr || params_[kParamIndexTwo]->GetObject() == nullptr) {
    return;
  }
  OpTracePtr subscr_op;
  if ((subscr_op = CastTrace<OpTrace>(params_[kParamIndexOne])) != nullptr &&
      (CastConstTrace(params_[kParamIndexTwo]) != nullptr || params_[kParamIndexTwo]->IsConst())) {
    if (subscr_op->params_.size() < kParamCountTwo) {
      return;
    }
    auto tsr = subscr_op->GetParam(kParamIndexOne)->GetObject();
    if (tsr == nullptr) {
      return;
    }
    std::string type_name = std::string(py::str(reinterpret_cast<PyObject *>(Py_TYPE(tsr))));
    if (type_name.find(kTensorName) == std::string::npos) {
      return;
    }
    if (subscr_op->GetParam(kParamIndexTwo)->IsConst() && params_[kParamIndexTwo]->IsConst()) {
      EnableRelax();
    }
  }
}

void OpTrace::JudgeBoundMethodPass() {
  if (RelaxEnabled()) {
    return;
  }
  if (opcode_ != LOAD_ATTR || params_.size() < kParamCountOne) {
    return;
  }
  if (params_[kParamIndexOne]->GetObject() == nullptr) {
    return;
  }
  if (name_ == kFuncName) {
    EnableRelax();
  }
}

void OpTrace::JudgeSubScrRandPass() {
  if (RelaxEnabled()) {
    return;
  }
  if (opcode_ != BINARY_SUBTRACT || params_.size() < kParamCountTwo) {
    return;
  }
  auto call_op = CastOpTrace(params_[kParamIndexOne], CALL_FUNCTION);
  ConstTracePtr prim;
  if (call_op != nullptr && call_op->params_.size() > kParamCountOne) {
    if ((prim = CastConstTrace(call_op->params_[kParamIndexOne])) != nullptr) {
      std::string prim_name = py::cast<mindspore::PrimitivePyAdapterPtr>(prim->GetObject())->name();
      if (prim_name == kRankPrimName) {
        EnableRelax();
      }
    }
  }
}

static const std::unordered_map<size_t, size_t> &GetGuardFuncKeyMap() {
  static std::unordered_map<size_t, size_t> map = {};
  static bool init = false;
  if (init) {
    return map;
  }
  init = true;
  py::object func_map = Utils::GetModuleAttr(kFuncWhiteListModuleName, kGuardFuncMapName, true, true);
  MS_EXCEPTION_IF_CHECK_FAIL(PyDict_CheckExact(func_map.ptr()), "white list func map must be 'dict[int, int]'");
  PyObject *key;
  PyObject *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(func_map.ptr(), &pos, &key, &value)) {
    MS_EXCEPTION_IF_CHECK_FAIL(PyLong_CheckExact(key), "white list func map key must be 'int'");
    MS_EXCEPTION_IF_CHECK_FAIL(PyLong_CheckExact(value), "white list func map value must be 'int'");
    map[PyLong_AsSize_t(key)] = PyLong_AsSize_t(value);
  }
  return map;
}

static bool CheckRelaxGuardFunc(const py::object &callable) {
  static size_t guard_key_relax_func = 0;
  if (guard_key_relax_func == 0) {
    py::object key_object = Utils::GetModuleAttr(kFuncWhiteListModuleName, "GUARD_KEY_RELAX_FUNC", true, true);
    guard_key_relax_func = py::cast<size_t>(key_object);
  }

  auto iter = GetGuardFuncKeyMap().find(FunctionId(callable));
  return iter != GetGuardFuncKeyMap().end() && iter->second == guard_key_relax_func;
}

void OpTrace::JudgeRelaxGuardFuncPass() {
  if (opcode_ != CALL_FUNCTION || params_.size() < kParamCountOne) {
    return;
  }
  if (CheckRelaxGuardFunc(py::cast<py::object>(params_[0]->GetObject()))) {
    EnableRelax();
  }
}

void OpTrace::CheckSpecialize() {
  bool any_params_specialized = false;
  for (auto param : params_) {
    if (!param->IsConst() && param->IsSpecialized()) {
      any_params_specialized = true;
      break;
    }
  }
  if (opcode_ == CALL_FUNCTION) {
    if ((name_ == kShape_Name || name_ == kCastToAdapterTensor || name_ == kCastToMSTensor) &&
        !any_params_specialized) {
      is_specialized_ = true;
    } else if (params_.size() > kParamCountOne &&
               py::isinstance<mindspore::PrimitivePyAdapter>(params_[kParamIndexOne]->GetObject())) {
      std::string prim_name = py::cast<mindspore::PrimitivePyAdapterPtr>(params_[kParamIndexOne]->GetObject())->name();
      if (prim_name == kCastPrimName) {
        is_specialized_ = params_[kParamIndexTwo]->IsSpecialized();
      } else if (prim_name == kLayerNormPrimName) {
        is_specialized_ = params_[kParamIndexTwo]->IsSpecialized();
      } else if (prim_name == kReshapePrimName) {
        is_specialized_ = any_params_specialized || !params_[kParamIndexThree]->IsConst();
      } else if (prim_name == kShapePrimName) {
        is_specialized_ = params_[kParamIndexTwo]->IsSpecialized();
      }
    }
  } else {
    is_specialized_ = true;
  }
}

TracePtr OpTrace::Optimize() {
  if (is_const_ || RelaxEnabled()) {
    return nullptr;
  }
  TracePtr ret;
  if ((ret = RemoveCastDuplicatePatternPass()) != nullptr || (ret = RemoveEmptyTensorPass()) != nullptr ||
      (ret = RemovePrimOutIsTensorPass()) != nullptr || (ret = RemoveCastPass()) != nullptr) {
    return ret;
  }
  if (relax_limit_ > 0) {
    JudgeDTypeChangePass();
    JudgeDTypeScopePass();
    JudgeTrainFlagPass();
    JudgeCompareConstPass();
    JudgeContainsConstPass();
    JudgeInplaceAddConstPass();
    JudgeIsConstPass();
    JudgeCodeChangePass();
    JudgeBoundMethodPass();
    JudgeSubScrRandPass();
    JudgeDTypeTensorAttrPass();
    JudgeRelaxGuardFuncPass();
  }
  bool need_update = false;
  for (size_t i = 0; i < params_.size(); ++i) {
    params_[i] = OptimizeTrace(params_[i], &need_update);
  }
  if (need_update) {
    if (!std::any_of(params_.begin(), params_.end(), [](const TracePtr &item) { return !item->IsConst(); })) {
      is_const_ = true;
    }
    info_ = nullptr;
    strTrace_ = "";
    Info();
    return shared_from_this();
  } else {
    return nullptr;
  }
}

void OpTrace::SetRelaxCount(int cnt) {
  Trace::SetRelaxCount(cnt);
  for (auto param : params_) {
    param->SetRelaxCount(cnt);
  }
}

bool OpTrace::operator==(const Trace &trace) {
  bool ret = false;
  if (Trace::operator==(trace)) {
    const OpTrace &t = (const OpTrace &)trace;
    ret = opcode_ == t.opcode_ && opargs_ == t.opargs_ && name_ == t.name_ && params_.size() == t.params_.size();
    if (ret) {
      for (size_t i = 0; i < params_.size(); i++) {
        if (*(params_[i]) == *(t.params_[i])) {
          continue;
        } else {
          ret = false;
          break;
        }
      }
    }
  }
  return ret;
}

void OpTrace::Replace(std::shared_ptr<Trace> dst, std::shared_ptr<Trace> src) {
  Trace::Replace(dst, src);
  for (size_t i = 0; i < params_.size(); ++i) {
    if (*params_[i] == *src) {
      params_[i] = dst;
    } else {
      params_[i]->Replace(dst, src);
    }
  }
}

void OpTrace::Detach() {
  Trace::Detach();
  for (auto t : params_) {
    t->Detach();
  }
}

bool OpTrace::Support(TraceType tt) { return tt == TraceType::Operation; }

static std::map<int, TraceType> kMapBytecodeToTraceType = {
  {LOAD_CLOSURE, TraceType::Closure}, {LOAD_DEREF, TraceType::Deref},           {LOAD_GLOBAL, TraceType::Global},
  {LOAD_NAME, TraceType::Name},       {LOAD_CLASSDEREF, TraceType::ClassDeref},
};

TracePtr CreateOpTraceByBytecode(PyObject *obj, int opcode, int opargs, TraceVector params, std::string module_name,
                                 std::string name, bool strict) {
  static const std::set<int> root_op = {
    LOAD_CLOSURE, LOAD_DEREF, LOAD_GLOBAL, LOAD_NAME, LOAD_CLASSDEREF,
  };
  if (opcode == LOAD_DEREF && opargs < 0) {
    return nullptr;
  }
  if (root_op.find(opcode) != root_op.end()) {
    return std::make_shared<RootTrace>(obj, kMapBytecodeToTraceType[opcode], opargs, name, module_name);
  }
  if (opcode == LOAD_CONST) {
    return std::make_shared<ConstTrace>(obj, -1);
  }
  if (Opcode(opcode).IsCall()) {
    if (params.size() < 1 || !SupportCall(params[0]->GetObject(), name)) {
      if (strict) {
        return nullptr;
      } else {
        return std::make_shared<UnsupportedTrace>(obj, params, opcode, opargs);
      }
    }
  }
  return std::make_shared<OpTrace>(obj, opcode, opargs, params, name);
}

TracePtr CreateOpTrace(PyObject *obj, int opcode, int opargs, TraceVector params, const std::string &module_name,
                       const std::string &name, bool strict, bool print) {
  std::vector<PyObject *> vparams;
  for (auto trace : params) {
    if (trace == nullptr) {
      return nullptr;
    } else if (trace->GetTraceType() == TraceType::Unsupported) {
      return std::make_shared<UnsupportedTrace>(obj, params, opcode, opargs);
    } else {
      vparams.push_back(trace->GetObject());
    }
  }
  if (kBytecodeExecuter.find(opcode) == kBytecodeExecuter.end() || !kBytecodeExecuter[opcode].first(opargs, vparams)) {
    if (print) {
      GRAPH_JIT_LOG_F("Unsupported bytecode %d args %d!\n", opcode, opargs);
    } else {
      MS_LOG(DEBUG) << "Unsupported bytecode " << opcode << " args " << opargs << "!";
    }
    if (strict) {
      return nullptr;
    } else {
      return std::make_shared<UnsupportedTrace>(obj, params, opcode, opargs);
    }
  }
  return CreateOpTraceByBytecode(obj, opcode, opargs, params, module_name, name, strict);
}

CustomizedTrace::CustomizedTrace(PyObject *obj, RetrieveFunc rfunc, ToStringFunc sfunc)
    : Trace(obj, nullptr), retrieve_(rfunc), tostring_(sfunc) {
  curType_ = TraceType::Customized;
  depth_ = 1;
}

PyObject *CustomizedTrace::Retrieve(PTraceContext context, bool perf) {
  PyObject *ret = Trace::Retrieve(context, perf);
  if (ret != nullptr) {
    return ret;
  }
  TracePerf tp(this, perf, false);
  ret = retrieve_(context);
  Cache(context, ret);
  return ret;
}

std::string CustomizedTrace::ToString(bool include_param) {
  if (strTrace_.size() > 0) {
    return strTrace_;
  }
  std::string ret = tostring_(false);
  ret = (is_const_ ? std::string("const:") : std::string("var:")) + ret;
  ret = std::regex_replace(ret, std::regex("(\n)"), "");
  strTrace_ = ret;
  return ret;
}

const InfoPack &CustomizedTrace::Info() {
  if (info_ == nullptr) {
    InfoPack info;
    info << uint8_t(curType_);
    info.Begin();
    info << tostring_(true);
    info.End();
    info_ = std::make_shared<InfoPack>(info);
    info_->Update();
  }
  return *info_;
}

bool CustomizedTrace::Support(TraceType tt) { return tt == TraceType::Customized; }

UnsupportedTrace::UnsupportedTrace(PyObject *obj, TraceVector params, int op, int arg)
    : Trace(obj, nullptr), params_(params), op_(op), arg_(arg) {
  curType_ = TraceType::Unsupported;
  if (!std::any_of(params.begin(), params.end(), [](const TracePtr &item) { return !item->IsConst(); })) {
    is_const_ = true;
  }
  depth_ = std::accumulate(params.begin(), params.end(), 1, [](int depth, const TracePtr &i) {
    int d = i->GetDepth() + 1;
    if (d > depth) {
      return d;
    } else {
      return depth;
    }
  });
}

PyObject *UnsupportedTrace::Retrieve(PTraceContext context, bool perf) {
  PyObject *ret = Trace::Retrieve(context, perf);
  if (ret != nullptr) {
    return ret;
  }
  std::vector<PyObject *> params;
  bool fail = false;
  for (auto p : params_) {
    auto obj = p->Retrieve(context, perf);
    params.push_back(obj);
    if (p->GetTraceType() != TraceType::Unsupported) {
      // compare obj with original obj in trace for inputs of unsupported trace
      auto orig = p->GetObject();
      if (!IsPyObjectEqual(obj, orig)) {
        fail = true;
        break;
      }
    }
    if (params.back() == nullptr) {
      MS_LOG(DEBUG) << "Guard Check Retrieve fail for " + p->ToString();
      fail = true;
      break;
    }
  }
  TracePerf tp(this, perf, false);
  for (auto p : params) {
    Py_XDECREF(p);
  }
  if (fail) {
    return nullptr;
  } else {
    Cache(context, obj_);
    return obj_;
  }
}

std::string UnsupportedTrace::ToString(bool include_param) {
  if (strTrace_.size() > 0) {
    return strTrace_;
  }
  std::string ret = "unsupported ";
  ret += Opcode(op_).name();
  ret += "(arg:";
  ret += std::to_string(arg_);
  if (include_param && params_.size() > 0) {
    ret += ",";
    for (auto t : params_) {
      ret += t->ToString(include_param) + ",";
    }
    ret = ret.substr(0, ret.size() - 1);
  }
  ret = ret + ")";
  ret = (is_const_ ? std::string("const:") : std::string("var:")) + ret;
  ret = std::regex_replace(ret, std::regex("(\n)"), "");
  strTrace_ = ret;
  return ret;
}

std::string UnsupportedTrace::FormatString(std::map<Trace *, size_t> *cache) {
  std::stringstream s;
  std::stringstream params_str;
  params_str << "(";
  for (auto i : params_) {
    if (cache->find(i.get()) == cache->end()) {
      s << i->FormatString(cache) << std::endl;
    }
    params_str << "%" << (cache->find(i.get())->second) << ", ";
    if (i->GetTraceType() == TraceType::Unsupported) {
      params_str << "...";
      break;
    }
  }
  params_str << ")";

  cache->insert(std::make_pair(this, cache->size()));
  s << "%" << cache->find(this)->second << " = unsupported " << Opcode(op_).name() << " " << arg_ << ": "
    << params_str.str();
  return s.str();
}

const InfoPack &UnsupportedTrace::Info() {
  if (info_ == nullptr) {
    InfoPack info;
    info << uint8_t(curType_);
    info.Begin();
    info << op_;
    info << arg_;
    info << uint64_t(params_.size());
    for (auto i : params_) {
      info << i->Info();
    }
    info.End();
    info_ = std::make_shared<InfoPack>(info);
    info_->Update();
  }
  return *info_;
}

void UnsupportedTrace::SetRelaxCount(int cnt) {
  Trace::SetRelaxCount(cnt);
  for (auto param : params_) {
    param->SetRelaxCount(cnt);
  }
}

TraceVector UnsupportedTrace::GetParams() { return params_; }

void UnsupportedTrace::Detach() {
  Trace::Detach();
  for (auto t : params_) {
    t->Detach();
  }
}

bool UnsupportedTrace::Support(TraceType tt) { return tt == TraceType::Unsupported; }

PyObject *GetObjectFromTrace(const PyFrameObject *frame, TracePtr trace, std::map<size_t, PyObject *> *cache,
                             bool perf) {
  TraceContext context = {frame->f_globals,    frame->f_builtins, frame->f_locals,
                          frame->f_localsplus, frame->f_code,     cache};
  if (trace != nullptr) {
    return trace->Retrieve(&context, perf);
  } else {
    return nullptr;
  }
}
}  // namespace pijit
}  // namespace mindspore
