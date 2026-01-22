/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include "frontend/jit/pi/graph_guard/guard_utils.h"
#include <cstdint>
#include <regex>

#include "ir/map_tensor.h"
#include "ir/quantization_param.h"
#include "ir/tensor_new.h"
#include "pybind11/pybind11.h"
#include "include/frontend/operator/primitive_py.h"
#include "include/utils/convert_utils_py.h"
#include "frontend/jit/pi/utils/utils.h"
#include "include/utils/stub_tensor.h"
#include "frontend/jit/pi/graph_guard/strategy.h"
#include "frontend/jit/pi/graph_guard/guard.h"
#include "frontend/jit/pi/graph_guard/infer.h"
#include "frontend/jit/pi/python_adapter/pydef.h"
#include "frontend/jit/pi/graph_guard/guard_fail_reason.h"
#include "include/utils/tensor_py.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace pijit {

static PyObject *kPyAttrReprCache = nullptr;
static const char kPyAttrReprCacheStr[] = "__repr_cache__";

static PyObject *GetAttrReprCacheStr() {
  if (kPyAttrReprCache == nullptr) {
    kPyAttrReprCache = PyUnicode_FromString(kPyAttrReprCacheStr);
  }
  return kPyAttrReprCache;
}

#define DESC(op) (std::string("{") + std::string(#op) + std::string(":") + (op) + std::string("}"))
#define DESC_STRING(op) (std::string("{") + std::string(#op) + std::string(":") + std::to_string(op) + std::string("}"))
#define DESC_STRING_L(op, l)                                                                                          \
  (std::string("{") + std::string(#op) + std::string("[") + std::to_string(l) + std::string("]") + std::string(":") + \
   std::to_string(op) + std::string("}"))  // NOLINT
#define DESC_STRING_S(op, l)                                                                                          \
  (std::string("{") + std::string(#op) + std::string("[") + std::to_string(l) + std::string("]") + std::string(":") + \
   (op) + std::string("}"))  // NOLINT
#define DESC_STRING_O(obj, op) \
  (std::string("{") + std::string(#op) + std::string(":") + std::to_string(obj->op) + std::string("}"))
#define DESC_TOSTRING(op)                                                                                             \
  (std::string("{") + std::string(#op) + std::string(":") + ((op == nullptr) ? std::string("nil") : op->ToString()) + \
   std::string("}"))  // NOLINT
#define DESC_ITEM(opK, opV)                                                                          \
  (std::string("{") + ((opK == nullptr) ? std::string("nil") : opK->ToString()) + std::string(":") + \
   ((opV == nullptr) ? std::string("nil") : opV->ToString()) + std::string("}"))  // NOLINT
#define DESC_ITEM_V(op) (std::string("{") + std::to_string(op) + std::string("}"))
#define DESC_ITEM_T(op) (std::string("{") + ((op == nullptr) ? std::string("nil") : op->ToString()) + std::string("}"))
#define DESC_INDEX(op, idx)                                                                          \
  (std::string("{") + std::string(#op) + std::string("[") + std::to_string(idx) + std::string("]") + \
   std::string(":") + ((op[idx] == nullptr) ? std::string("nil") : op[idx]->ToString()) + std::string("}"))  // NOLINT
#define DESC_INDEX_V(op, idx)                                                                        \
  (std::string("{") + std::string(#op) + std::string("[") + std::to_string(idx) + std::string("]") + \
   std::string(":") + std::to_string(op[idx]) + std::string("}"))  // NOLINT
#define DESC_END ItemData::ToString()

typedef enum _ItemType {
  PyNull = 0,
  PyLong,
  PyFloat,
  PyBool,
  PyBytes,
  PyStr,
  PyList,
  PyTuple,
  PySet,
  PyFrozenSet,
  PyDict,
  PyComplex,
  PySlice,
  PyFunction,
  PyMethod,
  PyInstanceMethod,
  PyType,
  PyNumpy,
  PyUnknown,
  TensorType,
  ParamInfo,
  MetaTensor,
  Tensor,
  MapTensor,
  RowTensor,
  COOTensor,
  CSRTensor,
  Tensordata,
  Primitive,
  Cell,
} ItemType;

class ItemData {
 public:
  ItemData(ItemType itemType, bool needSpecialize, int recurseDepth)
      : tp_(itemType), specialized_(needSpecialize), recurseDepth_(recurseDepth), info_(nullptr) {}

  virtual ~ItemData() = default;

  virtual bool operator==(const ItemData &obj) const { return obj.tp_ == tp_ && specialized_ == obj.specialized_; }

  virtual bool operator==(PyObject *obj) const { return tp_ != ItemType::PyNull || obj == nullptr || obj == Py_None; }

  virtual std::string ToString() {
    if (tp_ == ItemType::PyNull) {
      return "(null)";
    } else {
      return std::string("(type:") + std::to_string(SizeToInt(tp_)) + ",specialize:" + std::to_string(specialized_) +
             ",recurse:" + std::to_string(recurseDepth_) + ")";
    }
  }

  virtual const InfoPack &Info() {
    if (info_ == nullptr) {
      info_ = std::make_shared<InfoPack>();
      InfoPack &info = *info_;
      info << uint8_t(tp_);
      info.Begin();
      if (tp_ != ItemType::PyNull && tp_ != ItemType::PyUnknown) {
        info << specialized_ << recurseDepth_;
      }
      SubInfo(&info);
      info.End();
      info_->Update();
    }
    return *info_;
  }

  ItemType GetItemType() const { return tp_; }

  mutable GuardFailReason fail_reason_{GuardFailReason::kReasonUnknown};

 protected:
  virtual void SubInfo(InfoPack *info) {}
  ItemType tp_;
  bool specialized_;
  int recurseDepth_;
  InfoPackPtr info_;
};
using ItemDataPtr = std::shared_ptr<ItemData>;

static ItemDataPtr CreateItem(PyObject *obj, bool needSpecialize = true, int recurseDepth = INT_MAX);

class IntData : public ItemData {
 public:
  IntData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyLong, needSpecialize, recurseDepth) {
    intVar_ = PyLong_AsLong(obj);
  }

  bool operator==(const ItemData &obj) const override {
    return ItemData::operator==(obj) && (!specialized_ || ((static_cast<const IntData &>(obj)).intVar_ == intVar_));
  }

  bool operator==(PyObject *obj) const override {
    if (PyLong_Check(obj)) {
      return !specialized_ || PyLong_AsLong(obj) == intVar_;
    }
    return false;
  }

  std::string ToString() override { return DESC_STRING(intVar_) + DESC_END; }

 protected:
  void SubInfo(InfoPack *info) override { (*info) << intVar_; }
  int64_t intVar_;
};

class FloatData : public ItemData {
 public:
  FloatData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyFloat, needSpecialize, recurseDepth) {
    floatVar_ = PyFloat_AsDouble(obj);
  }

  bool operator==(const ItemData &obj) const override {
    return ItemData::operator==(obj) && (!specialized_ || (static_cast<const FloatData &>(obj)).floatVar_ == floatVar_);
  }

  bool operator==(PyObject *obj) const override {
    if (PyFloat_Check(obj)) {
      return !specialized_ || PyFloat_AsDouble(obj) == floatVar_;
    }
    return false;
  }

  std::string ToString() override { return DESC_STRING(floatVar_) + DESC_END; }

 protected:
  void SubInfo(InfoPack *info) override { (*info) << floatVar_; }
  double floatVar_;
};

class BoolData : public ItemData {
 public:
  BoolData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyBool, needSpecialize, recurseDepth) {
    boolVar_ = (obj == Py_True);
  }

  bool operator==(const ItemData &obj) const override {
    return ItemData::operator==(obj) && (!specialized_ || (static_cast<const BoolData &>(obj)).boolVar_ == boolVar_);
  }

  bool operator==(PyObject *obj) const override { return (obj == Py_True) == boolVar_; }

  std::string ToString() override { return DESC_STRING(boolVar_) + DESC_END; }

 protected:
  void SubInfo(InfoPack *info) override { (*info) << boolVar_; }
  bool boolVar_;
};

class BytesData : public ItemData {
 public:
  BytesData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyBytes, needSpecialize, recurseDepth), len_(PyBytes_Size(obj)) {
    if (needSpecialize) {
      buf_ = std::make_unique<uint8_t[]>(len_);
      if (buf_ != nullptr) {
        char *pBuf = PyBytes_AS_STRING(reinterpret_cast<PyBytesObject *>(obj));
        if (pBuf != nullptr) {
          memcpy_s(buf_.get(), len_, reinterpret_cast<uint8_t *>(pBuf), len_);
        } else {
          buf_.release();
        }
      }
    } else {
      buf_.reset(nullptr);
    }
  }

  ~BytesData() override { buf_.release(); }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const BytesData &other = static_cast<const BytesData &>(obj);
      return len_ == other.len_ &&
             ((specialized_ && (len_ == 0 || (buf_ != nullptr && other.buf_ != nullptr &&
                                              memcmp(buf_.get(), other.buf_.get(), len_) == 0))) ||
              (!specialized_));
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (!!PyBytes_Check(obj)) {
      char *pBuf = PyBytes_AS_STRING(reinterpret_cast<PyBytesObject *>(obj));
      return len_ == PyBytes_Size(obj) &&
             (!specialized_ ||
              (len_ == 0 || (buf_ != nullptr && pBuf != nullptr && memcmp(buf_.get(), pBuf, len_) == 0)));
    }
    return false;
  }

  std::string ToString() override {
    uintptr_t bytes = reinterpret_cast<uintptr_t>(buf_.get());
    return DESC_STRING_L(bytes, len_) + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override { (*info) << (uint64_t)len_ << reinterpret_cast<void *>(buf_.get()); }
  Py_ssize_t len_;
  std::unique_ptr<uint8_t[]> buf_;
};

class StringData : public ItemData {
 public:
  StringData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyStr, needSpecialize, recurseDepth) {
    if (needSpecialize) {
      strVal_ = PyUnicode_AsUTF8(obj);
    }
  }

  bool operator==(const ItemData &obj) const override {
    return ItemData::operator==(obj) && (!specialized_ || strVal_ == (static_cast<const StringData &>(obj)).strVal_);
  }

  bool operator==(PyObject *obj) const override {
    if (PyUnicode_Check(obj)) {
      return !specialized_ || strVal_ == PyUnicode_AsUTF8(obj);
    }
    return false;
  }

  std::string ToString() override { return DESC(strVal_) + DESC_END; }

 protected:
  void SubInfo(InfoPack *info) override { (*info) << strVal_; }
  std::string strVal_;
};

class ListData : public ItemData {
 public:
  ListData(PyObject *obj, bool needSpecialize, int recurseDepth);

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const ListData &list = static_cast<const ListData &>(obj);
      if (list.listVar_.size() == listVar_.size()) {
        return CompareList(list);
      }
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (PyList_Check(obj)) {
      return CheckList(obj);
    } else if (PyTuple_Check(obj)) {
      return CheckTuple(obj);
    } else if (PySet_Check(obj) || PyFrozenSet_Check(obj)) {
      return CheckSet(obj);
    }
    return false;
  }

  std::string ToString() override {
    std::string ret;
    for (auto it : listVar_) {
      ret += DESC_ITEM_T(it);
    }
    switch (tp_) {
      case ItemType::PyList: {
        std::string list = ret;
        ret = DESC_STRING_S(list, listVar_.size());
      } break;
      case ItemType::PyTuple: {
        std::string tuple = ret;
        ret = DESC_STRING_S(tuple, listVar_.size());
      } break;
      case ItemType::PySet: {
        std::string set = ret;
        ret = DESC_STRING_S(set, listVar_.size());
      } break;
      case ItemType::PyFrozenSet: {
        std::string fronzen_set = ret;
        ret = DESC_STRING_S(fronzen_set, listVar_.size());
      } break;
      default:
        ret = "unknown";
        break;
    }
    return ret + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override {
    (*info) << uint8_t(tp_);
    (*info) << uint64_t(listVar_.size());
    for (auto v : listVar_) {
      (*info) << v->Info();
    }
  }
  bool CheckList(PyObject *obj) const {
    if (listVar_.size() != static_cast<size_t>(Py_SIZE(obj))) {
      return false;
    }
    for (size_t idx = 0, size = listVar_.size(); idx < size; ++idx) {
      if (!(*(listVar_[idx]) == PyList_GET_ITEM(obj, idx))) {
        return false;
      }
    }
    return true;
  }

  bool CheckTuple(PyObject *obj) const {
    if (listVar_.size() != static_cast<size_t>(Py_SIZE(obj))) {
      return false;
    }
    for (size_t idx = 0, size = listVar_.size(); idx < size; ++idx) {
      if (!(*(listVar_[idx]) == PyTuple_GET_ITEM(obj, idx))) {
        return false;
      }
    }
    return true;
  }

  bool CheckSet(PyObject *obj) const {
    if ((tp_ == ItemType::PySet && !!PySet_Check(obj)) || (tp_ == ItemType::PyFrozenSet && !!PyFrozenSet_Check(obj))) {
      if (listVar_.size() != (size_t)PySet_Size(obj)) {
        return false;
      }
    } else {
      return false;
    }
    std::vector<PyObject *> match;
    return !std::any_of(listVar_.begin(), listVar_.end(), [&match, obj](const ItemDataPtr &item) {
      Py_ssize_t pos = 0;
      PyObject *it;
      Py_hash_t hash;
      while (_PySet_NextEntry(obj, &pos, &it, &hash)) {
        if (std::find(match.begin(), match.end(), it) == match.end() && *item == it) {
          match.push_back(it);
          return false;
        }
      }
      return true;
    });
  }
  bool CompareList(const ListData &list) const {
    if (!inOrder_) {
      std::vector<ItemDataPtr> listCpy = list.listVar_;
      for (size_t i = 0, j; i < listVar_.size(); ++i) {
        size_t lenList = listCpy.size();
        for (j = 0; j < lenList; ++j) {
          if (*(listCpy[j]) == *(listVar_[i])) {
            listCpy.erase(listCpy.begin() + j);
            break;
          }
        }
        if (j == lenList) {
          return false;
        }
      }
    } else {
      for (size_t i = 0; i < listVar_.size(); ++i) {
        if (*(list.listVar_[i]) == *(listVar_[i])) {
          continue;
        } else {
          return false;
        }
      }
    }
    return true;
  }
  void InitList(PyObject *obj, bool needSpecialize, int recurseDepth) {
    tp_ = ItemType::PyList;
    for (Py_ssize_t i = 0; i < PyList_Size(obj); ++i) {
      PyObject *item = PyList_GetItem(obj, i);
      if (item != NULL) {
        if (recurseDepth > 0 || needSpecialize) {
          listVar_.push_back(CreateItem(item, needSpecialize, recurseDepth));
        } else {
          listVar_.push_back(CreateItem(reinterpret_cast<PyObject *>(Py_TYPE(item)), false, false));
        }
      }
    }
  }
  void InitTuple(PyObject *obj, bool needSpecialize, int recurseDepth) {
    tp_ = ItemType::PyTuple;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(obj); ++i) {
      PyObject *item = PyTuple_GET_ITEM(obj, i);
      if (item != NULL) {
        if (recurseDepth > 0 || needSpecialize) {
          listVar_.push_back(CreateItem(item, needSpecialize, recurseDepth));
        } else {
          listVar_.push_back(CreateItem(reinterpret_cast<PyObject *>(Py_TYPE(item)), false, false));
        }
      }
    }
  }
  void InitSet(PyObject *obj, bool needSpecialize, int recurseDepth) {
    tp_ = ItemType::PySet;
    Py_ssize_t pos = 0;
    PyObject *item;
    Py_hash_t hash;
    while (_PySet_NextEntry(obj, &pos, &item, &hash)) {
      if (recurseDepth > 0 || needSpecialize) {
        listVar_.push_back(CreateItem(item, needSpecialize, recurseDepth));
      } else {
        listVar_.push_back(CreateItem(reinterpret_cast<PyObject *>(Py_TYPE(item)), false, false));
      }
    }
    inOrder_ = false;
  }
  void InitFrozenSet(PyObject *obj, bool needSpecialize, int recurseDepth) {
    tp_ = ItemType::PyFrozenSet;
    Py_ssize_t pos = 0;
    PyObject *item;
    Py_hash_t hash;
    while (_PySet_NextEntry(obj, &pos, &item, &hash)) {
      if (recurseDepth > 0 || needSpecialize) {
        listVar_.push_back(CreateItem(item, needSpecialize, recurseDepth));
      } else {
        listVar_.push_back(CreateItem(reinterpret_cast<PyObject *>(Py_TYPE(item)), false, false));
      }
    }
    inOrder_ = false;
  }
  std::vector<ItemDataPtr> listVar_;
  bool inOrder_ = true;
};

class ComplexData : public ItemData {
 public:
  ComplexData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyComplex, needSpecialize, recurseDepth) {
    if (needSpecialize) {
      complexVar_ = std::make_pair(PyComplex_RealAsDouble(obj), PyComplex_ImagAsDouble(obj));
    }
  }

  bool operator==(const ItemData &obj) const override {
    return ItemData::operator==(obj) &&
           (!specialized_ || (static_cast<const ComplexData &>(obj)).complexVar_ == complexVar_);
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    if (!!PyComplex_Check(obj)) {
      return !specialized_ ||
             (PyComplex_RealAsDouble(obj) == complexVar_.first && PyComplex_ImagAsDouble(obj) == complexVar_.second);
    }
    return false;
  }

  std::string ToString() override {
    return "complex(" + std::to_string(complexVar_.first) + "," + std::to_string(complexVar_.second) + ")" + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override { (*info) << complexVar_.first << complexVar_.second; }
  std::pair<double, double> complexVar_;
};

class SliceData : public ItemData {
 public:
  SliceData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PySlice, needSpecialize, recurseDepth) {
    Py_ssize_t start = 0;
    Py_ssize_t stop = 0;
    Py_ssize_t step = 0;
    if (needSpecialize) {
      PySlice_Unpack(obj, &start, &stop, &step);
      sliceVar_.push_back((int64_t)start);
      sliceVar_.push_back((int64_t)stop);
      sliceVar_.push_back((int64_t)step);
    }
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const SliceData &other = static_cast<const SliceData &>(obj);
      return (!specialized_ || (other.sliceVar_[SLICE_IDX_START] == sliceVar_[SLICE_IDX_START] &&
                                other.sliceVar_[SLICE_IDX_STOP] == sliceVar_[SLICE_IDX_STOP] &&
                                other.sliceVar_[SLICE_IDX_STEP] == sliceVar_[SLICE_IDX_STEP]));
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    if (!!PySlice_Check(obj)) {
      Py_ssize_t start = 0;
      Py_ssize_t stop = 0;
      Py_ssize_t step = 0;
      PySlice_Unpack(obj, &start, &stop, &step);
      return !specialized_ || (start == sliceVar_[SLICE_IDX_START] && stop == sliceVar_[SLICE_IDX_STOP] &&
                               step == sliceVar_[SLICE_IDX_STEP]);
    }
    return false;
  }

  std::string ToString() override {
    std::string slice;
    for (auto it : sliceVar_) {
      slice += DESC_ITEM_V(it);
    }
    return DESC_STRING_S(slice, sliceVar_.size()) + DESC_END;
  }

 protected:
  static constexpr int SLICE_IDX_START = 0;
  static constexpr int SLICE_IDX_STOP = 1;
  static constexpr int SLICE_IDX_STEP = 2;

  void SubInfo(InfoPack *info) override { (*info) << sliceVar_; }
  std::vector<int64_t> sliceVar_;
};

typedef enum _DictType {
  DtDict = 0,
  DtKeys,
  DtValues,
  DtItems,
} DictType;

class DictData : public ItemData {
 public:
  DictData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyDict, needSpecialize, recurseDepth) {
    if (PyDictKeys_Check(obj)) {
      dt_ = DictType::DtKeys;
      obj = PyObject_Vectorcall(reinterpret_cast<PyObject *>(&PyList_Type), &obj, 1, nullptr);
    } else if (PyDictValues_Check(obj)) {
      dt_ = DictType::DtValues;
      obj = PyObject_Vectorcall(reinterpret_cast<PyObject *>(&PyList_Type), &obj, 1, nullptr);
    } else if (PyDictItems_Check(obj)) {
      dt_ = DictType::DtItems;
      obj = PyObject_Vectorcall(reinterpret_cast<PyObject *>(&PyDict_Type), &obj, 1, nullptr);
    } else {
      dt_ = DictType::DtDict;
    }
    Py_ssize_t pos = 0;
    PyObject *key;
    PyObject *val;
    if (dt_ == DictType::DtItems || dt_ == DictType::DtDict) {
      while (PyDict_Next(obj, &pos, &key, &val)) {
        ItemDataPtr k;
        ItemDataPtr v;
        if (recurseDepth > 0 || needSpecialize) {
          k = CreateItem(key, needSpecialize, recurseDepth);
          v = CreateItem(val, needSpecialize, recurseDepth);
        } else {
          k = CreateItem(reinterpret_cast<PyObject *>(Py_TYPE(key)), false, false);
          v = CreateItem(reinterpret_cast<PyObject *>(Py_TYPE(val)), false, false);
        }
        listK_.push_back(k);
        listV_.push_back(v);
      }
    } else {
      std::vector<ItemDataPtr> &list = dt_ == DictType::DtKeys ? listK_ : listV_;
      for (const auto &h : py::handle(obj)) {
        PyObject *item = h.ptr();
        if (recurseDepth > 0 || needSpecialize) {
          list.push_back(CreateItem(item, needSpecialize, recurseDepth));
        } else {
          list.push_back(CreateItem(reinterpret_cast<PyObject *>(Py_TYPE(item)), false, false));
        }
      }
    }
    if (dt_ != DictType::DtDict) {
      Py_DECREF(obj);
    }
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const DictData &other = static_cast<const DictData &>(obj);
      if (dt_ != other.dt_) {
        return false;
      }
      if ((dt_ == DictType::DtValues || other.listK_.size() == listK_.size()) &&
          (dt_ == DictType::DtKeys || other.listV_.size() == listV_.size())) {
        return CompareKV(other);
      }
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    if ((dt_ == DictType::DtDict && !!PyDict_Check(obj)) || (dt_ == DictType::DtItems && !!PyDictItems_Check(obj))) {
      return CheckDictItems(obj);
    } else if ((dt_ == DictType::DtKeys && !!PyDictKeys_Check(obj)) ||
               (dt_ == DictType::DtValues && !!PyDictValues_Check(obj))) {
      return CheckKeyValue(obj);
    }
    return false;
  }

  std::string ToString() override {
    std::string dict = DESC_STRING(dt_);
    size_t listSize = 0;
    if (dt_ == DictType::DtItems || dt_ == DictType::DtDict) {
      listSize = listK_.size();
      for (size_t i = 0; i < listSize; ++i) {
        dict += DESC_ITEM(listK_[i], listV_[i]);
      }
    } else if (dt_ == DictType::DtKeys) {
      listSize = listK_.size();
      for (size_t i = 0; i < listSize; ++i) {
        dict += DESC_ITEM_T(listK_[i]);
      }
    } else if (dt_ == DictType::DtValues) {
      listSize = listV_.size();
      for (size_t i = 0; i < listSize; ++i) {
        dict += DESC_ITEM_T(listV_[i]);
      }
    }
    return DESC_STRING_S(dict, listSize) + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override {
    (*info) << dt_;
    (*info) << uint64_t(listK_.size());
    for (auto i : listK_) {
      (*info) << i->Info();
    }
    (*info) << uint64_t(listV_.size());
    for (auto i : listV_) {
      (*info) << i->Info();
    }
  }

  bool CheckDictItems(PyObject *obj) const {
    if (((dt_ == DictType::DtDict) && ((size_t)PyDict_Size(obj) != listK_.size())) ||
        ((dt_ == DictType::DtItems) && ((size_t)PyObject_Size(obj) != listK_.size()))) {
      return false;
    }
    std::vector<PyObject *> match;
    for (size_t idx = 0; idx < listK_.size(); idx++) {
      ItemDataPtr itemK = listK_[idx];
      ItemDataPtr itemV = listV_[idx];
      Py_ssize_t pos = 0;
      PyObject *key;
      PyObject *val;
      bool find = false;
      while (PyDict_Next(obj, &pos, &key, &val)) {
        if (std::find(match.begin(), match.end(), key) == match.end() && *itemK == key && *itemV == val) {
          match.push_back(key);
          find = true;
          break;
        }
      }
      if (!find) {
        return false;
      }
    }
    return true;
  }

  bool CheckKeyValue(PyObject *obj) const {
    std::vector<ItemDataPtr> listA = dt_ == DictType::DtKeys ? listK_ : listV_;
    if (listA.size() != (size_t)PyObject_Size(obj)) {
      return false;
    }
    size_t idx = 0;
    for (const auto &h : py::handle(obj)) {
      ItemDataPtr item = listA[idx++];
      if (!(*item == h.ptr())) {
        return false;
      }
    }
    return true;
  }
  bool CompareKV(const DictData &other) const {
    std::vector<ItemDataPtr> listCpK = other.listK_;
    std::vector<ItemDataPtr> listCpV = other.listV_;
    size_t listSize = listK_.size();
    if (listSize < listV_.size()) {
      listSize = listV_.size();
    }
    for (size_t i = 0, j = 0; i < listSize; ++i) {
      size_t cpListSize = dt_ == DictType::DtValues ? listCpV.size() : listCpK.size();
      for (; j < cpListSize; ++j) {
        if ((dt_ == DictType::DtValues || *(listK_[i]) == *(listCpK[j])) &&
            (dt_ == DictType::DtKeys || *(listV_[i]) == *(listCpV[j]))) {
          if (dt_ != DictType::DtValues) {
            listCpK.erase(listCpK.begin() + j);
          }
          if (dt_ != DictType::DtKeys) {
            listCpV.erase(listCpV.begin() + j);
          }
          break;
        }
      }
      if (j == cpListSize) {
        return false;
      }
    }
    return true;
  }
  DictType dt_;
  std::vector<ItemDataPtr> listK_;
  std::vector<ItemDataPtr> listV_;
};

class FunctionData : public ItemData {
 public:
  FunctionData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyFunction, needSpecialize, recurseDepth) {
    if (needSpecialize || recurseDepth > 0) {
      code_ = reinterpret_cast<PyCodeObject *>(PyFunction_GetCode(obj));
      defaults_ = CreateItem(PyFunction_GetDefaults(obj), needSpecialize, recurseDepth);
      kwdefaults_ = CreateItem(PyFunction_GetKwDefaults(obj), needSpecialize, recurseDepth);
      closure_ = CreateItem(PyFunction_GetClosure(obj), needSpecialize, recurseDepth);
    } else {
      code_ = reinterpret_cast<PyCodeObject *>(PyFunction_GetCode(obj));
      PyObject *temp = PyFunction_GetDefaults(obj);
      defaults_ =
        CreateItem((temp == NULL || temp == Py_None) ? Py_None : reinterpret_cast<PyObject *>(Py_TYPE(temp)), false, 0);
      temp = PyFunction_GetKwDefaults(obj);
      kwdefaults_ =
        CreateItem((temp == NULL || temp == Py_None) ? Py_None : reinterpret_cast<PyObject *>(Py_TYPE(temp)), false, 0);
      temp = PyFunction_GetClosure(obj);
      closure_ =
        CreateItem((temp == NULL || temp == Py_None) ? Py_None : reinterpret_cast<PyObject *>(Py_TYPE(temp)), false, 0);
    }
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const FunctionData &other = static_cast<const FunctionData &>(obj);
      return code_ == other.code_ && *defaults_ == *(other.defaults_) && *kwdefaults_ == *(other.kwdefaults_) &&
             *closure_ == *(other.closure_);
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    if (!!PyFunction_Check(obj)) {
      return reinterpret_cast<PyCodeObject *>(PyFunction_GetCode(obj)) == code_ &&
             *defaults_ == PyFunction_GetDefaults(obj) && *kwdefaults_ == PyFunction_GetKwDefaults(obj) &&
             *closure_ == PyFunction_GetClosure(obj);
    }
    return false;
  }

  std::string ToString() override {
    std::string func = DESC_TOSTRING(defaults_) + DESC_TOSTRING(kwdefaults_) + DESC_TOSTRING(closure_);
    return DESC(func) + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override {
    (*info) << (defaults_ != nullptr);
    if (defaults_ != nullptr) {
      (*info) << defaults_->Info();
    }
    (*info) << (kwdefaults_ != nullptr);
    if (kwdefaults_ != nullptr) {
      (*info) << kwdefaults_->Info();
    }
    (*info) << (closure_ != nullptr);
    if (closure_ != nullptr) {
      (*info) << closure_->Info();
    }
  }
  PyCodeObject *code_;
  ItemDataPtr defaults_;
  ItemDataPtr kwdefaults_;
  ItemDataPtr closure_;
};

class MethodData : public ItemData {
 public:
  MethodData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyMethod, needSpecialize, recurseDepth),
        refFunc_(CreateItem(PyMethod_GET_FUNCTION(obj), needSpecialize, recurseDepth)),
        refSelf_(CreateItem(PyMethod_GET_SELF(obj), needSpecialize, recurseDepth)) {}

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const MethodData &other = static_cast<const MethodData &>(obj);
      return *refFunc_ == *(other.refFunc_) && *refSelf_ == *(other.refSelf_);
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    if (!!PyMethod_Check(obj)) {
      return *refFunc_ == PyMethod_GET_FUNCTION(obj) && *refSelf_ == PyMethod_GET_SELF(obj);
    }
    return false;
  }

  std::string ToString() override {
    std::string method = DESC_TOSTRING(refFunc_) + DESC_TOSTRING(refSelf_);
    return DESC(method) + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override {
    (*info) << (refFunc_ != nullptr);
    if (refFunc_ != nullptr) {
      (*info) << refFunc_->Info();
    }
    (*info) << (refSelf_ != nullptr);
    if (refSelf_ != nullptr) {
      (*info) << refSelf_->Info();
    }
  }
  ItemDataPtr refFunc_;
  ItemDataPtr refSelf_;
};

class InstanceMethodData : public ItemData {
 public:
  InstanceMethodData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyInstanceMethod, needSpecialize, recurseDepth),
        refFunc_(CreateItem(PyInstanceMethod_GET_FUNCTION(obj), needSpecialize, recurseDepth)) {}

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const InstanceMethodData &other = static_cast<const InstanceMethodData &>(obj);
      return *refFunc_ == *(other.refFunc_);
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    if (!!PyInstanceMethod_Check(obj)) {
      return *refFunc_ == PyInstanceMethod_GET_FUNCTION(obj);
    }
    return false;
  }

  std::string ToString() override {
    std::string instance_method = DESC_TOSTRING(refFunc_);
    return DESC(instance_method) + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override {
    (*info) << (refFunc_ != nullptr);
    if (refFunc_ != nullptr) {
      (*info) << refFunc_->Info();
    }
  }
  ItemDataPtr refFunc_;
};

class TypeData : public ItemData {
 public:
  TypeData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyType, needSpecialize, recurseDepth) {
    refType_ = reinterpret_cast<PyTypeObject *>(obj);
    ambiguous_tensor_type_ = false;
  }

  void set_ambiguous_tensor_type(bool value) {
    if (value && (IsTensorType<true>(refType_))) {
      ambiguous_tensor_type_ = true;
    }
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      PyTypeObject *otherType = (static_cast<const TypeData &>(obj)).refType_;
      bool ret = refType_ == otherType;
      if (!ret) {
        ret = PyType_IsSubtype(refType_, otherType) || PyType_IsSubtype(otherType, refType_);
      }
      // adapter tensor type must be check exactly
      // if exactly type check failed, check ambiguous tensor type if necessary
      if (!ret) {
        if (ambiguous_tensor_type_) {
          ret = IsTensorType<true>(otherType);
        } else {
          bool isSelfTensor = IsTensorType<true>(refType_);
          bool isOtherTensor = IsTensorType<true>(otherType);
          ret = isSelfTensor && isOtherTensor;
        }
      }
      return ret;
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    PyTypeObject *otherType = reinterpret_cast<PyTypeObject *>(obj);
    if (!PyType_Check(obj)) {
      otherType = reinterpret_cast<PyTypeObject *>(Py_TYPE(obj));
    }
    if (!!PyType_Check(otherType)) {
      bool ret =
        refType_ == otherType || PyType_IsSubtype(refType_, otherType) || PyType_IsSubtype(otherType, refType_);
      if (!ret) {
        if (ambiguous_tensor_type_) {
          ret = IsTensorType<true>(otherType);
        } else {
          bool isSelfTensor = IsTensorType<true>(refType_);
          bool isOtherTensor = IsTensorType<true>(otherType);
          ret = isSelfTensor && isOtherTensor;
        }
      }
      return ret;
    }
    return false;
  }

  std::string ToString() override {
    std::string type = refType_->tp_name;
    return DESC(type) + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override { (*info) << refType_->tp_name; }
  PyTypeObject *refType_;

  // mix the tensor type.
  // only set true if _c_expression.Tensor type, common.Tensor type and all subtype of them
  bool ambiguous_tensor_type_;
};

class NumpyData : public ItemData {
 public:
  NumpyData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyNumpy, needSpecialize, recurseDepth) {
    py::array arr = py::cast<py::array>(obj);
    dtype_ = arr.dtype();
    size_ = (uint64_t)arr.size();
    itemsize_ = (uint64_t)arr.itemsize();
    ndim_ = (int64_t)arr.ndim();
    nbytes_ = (uint64_t)arr.nbytes();
    for (ssize_t i = 0; i < ndim_; ++i) {
      shape_.push_back((int64_t)arr.shape()[i]);
      strides_.push_back((int64_t)arr.strides()[i]);
    }
    if (arr.data() != nullptr) {
      if (needSpecialize) {
        buf_ = std::make_unique<uint8_t[]>(nbytes_);
        if (buf_ != NULL) {
          memcpy_s(buf_.get(), nbytes_, reinterpret_cast<uint8_t *>(arr.mutable_data()), nbytes_);
        }
      } else {
        buf_.reset(nullptr);
      }
    } else {
      buf_.reset(nullptr);
    }
  }

  ~NumpyData() override { buf_.release(); }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const NumpyData &other = static_cast<const NumpyData &>(obj);
      return dtype_ == other.dtype_ && size_ == other.size_ && ndim_ == other.ndim_ && nbytes_ == other.nbytes_ &&
             shape_ == other.shape_ && strides_ == other.strides_ &&
             (!specialized_ ||
              (buf_ != NULL && other.buf_ != NULL && memcmp(buf_.get(), other.buf_.get(), nbytes_) == 0));
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    if (py::isinstance<py::array>(obj)) {
      py::array arr = py::cast<py::array>(obj);
      bool ret = dtype_ == arr.dtype() && size_ == (uint64_t)arr.size() && ndim_ == (int64_t)arr.ndim() &&
                 nbytes_ == (uint64_t)arr.nbytes();
      if (!ret) {
        return ret;
      }
      for (ssize_t i = 0; i < ndim_; ++i) {
        if (shape_[i] != (int64_t)arr.shape()[i] || strides_[i] != (int64_t)arr.strides()[i]) {
          return false;
        }
      }
      return (!specialized_ || (buf_ != NULL && reinterpret_cast<uint8_t *>(arr.mutable_data()) != NULL &&
                                memcmp(buf_.get(), reinterpret_cast<uint8_t *>(arr.mutable_data()), nbytes_) == 0));
    }
    return false;
  }

  std::string ToString() override {
    std::string numpy;
    char dtype_kind = dtype_.kind();
    numpy +=
      DESC_STRING(dtype_kind) + DESC_STRING(size_) + DESC_STRING(itemsize_) + DESC_STRING(ndim_) + DESC_STRING(nbytes_);
    for (size_t i = 0; i < shape_.size(); ++i) {
      numpy += DESC_INDEX_V(shape_, i) + DESC_INDEX_V(strides_, i);
    }
    if (specialized_) {
      numpy += ",data_ptr:" + std::to_string(reinterpret_cast<intptr_t>(buf_.get()));
    }
    return DESC(numpy) + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override {
    (*info) << dtype_.kind() << size_ << itemsize_ << ndim_ << nbytes_ << shape_ << strides_;
    if (specialized_) {  // hash numpy data
      (*info) << static_cast<void *>(buf_.get());
    }
  }
  py::dtype dtype_;
  uint64_t size_;
  uint64_t itemsize_;
  int64_t ndim_;
  uint64_t nbytes_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
  std::unique_ptr<uint8_t[]> buf_;
};

class TensorTypeData : public ItemData {
 public:
  TensorTypeData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::TensorType, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    tpp_ = pyObj.cast<mindspore::TypePtr>();
  }

  bool operator==(const ItemData &obj) const override {
    return ItemData::operator==(obj) && (!specialized_ || *((static_cast<const TensorTypeData &>(obj)).tpp_) == *tpp_);
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    if (py::isinstance<mindspore::Type>(obj)) {
      return !specialized_ || *(py::cast<py::object>(obj).cast<mindspore::TypePtr>()) == *tpp_;
    }
    return false;
  }

  std::string ToString() override {
    std::string tensor_type = tpp_->ToString();
    return DESC(tensor_type) + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override { (*info) << tpp_; }
  mindspore::TypePtr tpp_;
};

class ParamInfoData : public ItemData {
 public:
  ParamInfoData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::ParamInfo, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    auto ptr = pyObj.cast<mindspore::ParamInfoPtr>();
    param_ = ptr->Clone();
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      if (!specialized_) {
        return true;
      }
      const ParamInfoData &other = static_cast<const ParamInfoData &>(obj);
      return Equal(param_, other.param_);
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    if (py::isinstance<mindspore::ParamInfo>(obj)) {
      auto pyObj = py::cast<py::object>(obj);
      auto param = pyObj.cast<mindspore::ParamInfoPtr>();
      return Equal(param_, param);
    }
    return false;
  }

  static bool Equal(ParamInfoPtr a, ParamInfoPtr b) {
    if (a == b) {
      return true;
    } else if (a == nullptr || b == nullptr) {
      return false;
    }
    return a->requires_grad() == b->requires_grad() && a->comm_fusion() == b->comm_fusion() &&
           a->parallel_optimizer() == b->parallel_optimizer() &&
           a->parallel_optimizer_comm_recompute() == b->parallel_optimizer_comm_recompute() &&
           a->parameter_shape() == b->parameter_shape() && a->use_persistent_storage() == b->use_persistent_storage() &&
           a->cache_enable() == b->cache_enable() && a->param_strategy() == b->param_strategy() &&
           a->cache_shape() == b->cache_shape() && a->requires_aggr() == b->requires_aggr();
  }

  std::string ToString() override {
    std::string param_info = ToStringAttr(param_);
    return DESC(param_info) + DESC_END;
  }

  static std::string ToStringAttr(mindspore::ParamInfoPtr p) {
    if (p == nullptr) {
      return "nil";
    }
    std::string param_name = p->name();
    std::string ret = DESC(param_name) + DESC_STRING_O(p, requires_grad()) + DESC_STRING_O(p, comm_fusion()) +
                      DESC_STRING_O(p, parallel_optimizer()) + DESC_STRING_O(p, requires_aggr()) +
                      DESC_STRING_O(p, parallel_optimizer_comm_recompute()) +
                      DESC_STRING_O(p, use_persistent_storage()) + DESC_STRING_O(p, cache_enable());
    auto parameter_shape = p->parameter_shape();
    for (size_t i = 0; i < parameter_shape.size(); ++i) {
      ret += DESC_INDEX_V(parameter_shape, i);
    }
    auto cache_shape = p->cache_shape();
    for (size_t i = 0; i < cache_shape.size(); ++i) {
      ret += DESC_INDEX_V(cache_shape, i);
    }
    auto param_strategy = p->param_strategy();
    for (size_t i = 0; i < param_strategy.size(); ++i) {
      ret += DESC_INDEX_V(param_strategy, i);
    }
    return ret;
  }

  static void SubInfoInner(InfoPack *info, mindspore::ParamInfoPtr p) {
    if (p == nullptr) {
      return;
    }
    (*info) << p->name() << p->requires_grad() << p->comm_fusion() << p->parallel_optimizer() << p->requires_aggr()
            << p->parallel_optimizer_comm_recompute() << p->use_persistent_storage() << p->cache_enable()
            << p->parameter_shape() << p->cache_shape() << p->param_strategy();
  }

 protected:
  void SubInfo(InfoPack *info) override { SubInfoInner(info, param_); }
  mindspore::ParamInfoPtr param_;
};

static constexpr int64_t kDynamicDim = -2;
static constexpr int64_t kDynamicShape = -1;

static bool IsDynamicDim(const ShapeVector &shape) {
  return std::any_of(shape.begin(), shape.end(), [](ShapeValueDType dim) { return dim == kDynamicDim; });
}

static bool CheckShape(const ShapeVector &a, const ShapeVector &b) {
  if (IsDynamicDim(a) || IsDynamicDim(b)) {
    return true;
  } else if (a.size() == b.size()) {
    for (size_t idx = 0; idx < a.size(); idx++) {
      if (a[idx] != kDynamicShape && b[idx] != kDynamicShape && a[idx] != b[idx]) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

class MetaTensorData : public ItemData {
 public:
  MetaTensorData(mindspore::tensor::MetaTensorPtr tensor_ptr, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::MetaTensor, needSpecialize, recurseDepth) {
    StoreTensor(tensor_ptr);
  }

  MetaTensorData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::MetaTensor, needSpecialize, recurseDepth) {
    StoreTensor(GetStubInfo(obj));
  }

  ~MetaTensorData() override = default;

  const auto &shape() const { return shape_; }
  const auto &data_type() const { return data_type_; }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const MetaTensorData &other = static_cast<const MetaTensorData &>(obj);
      bool ret = tid_ == other.tid_ && CheckShape(shape_, other.shape_) && is_parameter_ == other.is_parameter_ &&
                 CheckDataType(other);
      if (ret && is_parameter_) {
        ret = ParamInfoData::Equal(param_, other.param_);
      }
      return ret;
    }
    return false;
  }

  bool EqualInternal(const tensor::TensorPtr &tensor_ptr) const {
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    if (tid_ == tensor_ptr->data_type() && is_parameter_ == tensor_ptr->is_parameter() &&
        CheckTypeAndShape(tensor_ptr->Dtype(), tensor_ptr->shape())) {
      return !is_parameter_ || ParamInfoData::Equal(param_, tensor_ptr->param_info());
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (tensor::IsTensorPy(py::cast<py::object>(obj))) {
      return EqualInternal(GetStubInfo(obj));
    }
    return false;
  }

  static tensor::TensorPtr GetStubInfo(PyObject *obj) {
    py::object py_tensor = py::reinterpret_borrow<py::object>(obj);
    if (!tensor::IsTensorPy(obj)) {
      // stub tensor is deprecated and will be remove
      auto py_stub = py::getattr(obj, stub::PY_ATTR_STUB, Py_None);
      if (py_stub.ptr() != Py_None) {
        return std::static_pointer_cast<tensor::Tensor>(py_stub.cast<stub::StubNodePtr>()->WaitValue());
      }
      py_tensor = py::getattr(obj, stub::PY_ATTR_TENSOR, nullptr);
      MS_EXCEPTION_IF_CHECK_FAIL(py_tensor.ptr() != nullptr && tensor::IsTensorPy(py_tensor),
                                 std::string() + "unexpected tensor type: " + Py_TYPE(obj)->tp_name);
    }
    auto value_ptr = tensor::ConvertToTensor(py_tensor);
    return value_ptr;
  }

  bool IsDynamicShape() const { return is_dynamic_shape_; }

  std::string ToString() override {
    std::string meta_tensor = ToStringIntern();
    return DESC(meta_tensor) + DESC_END;
  }

 protected:
  MetaTensorData(bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::MetaTensor, needSpecialize, recurseDepth) {}
  virtual std::string ToStringIntern() {
    std::string param_desc = ParamInfoData::ToStringAttr(param_);
    std::string shape_str = "";
    for (size_t i = 0; i < shape_.size(); ++i) {
      shape_str += DESC_INDEX_V(shape_, i);
    }
    return DESC_STRING(tid_) + DESC_TOSTRING(data_type_) + DESC_STRING(is_parameter_) + DESC(param_desc) +
           DESC(shape_str);
  }
  bool CheckTypeAndShape(const TypePtr &tp, const ShapeVector &sv) const {
    if (!CheckShape(shape_, sv)) {
      fail_reason_ = GuardFailReason::kShapeNotEqual;
      return false;
    }
    if (!((data_type_ == nullptr && tp == nullptr) ||
          (data_type_ != nullptr && tp != nullptr && *data_type_ == *(tp)))) {
      fail_reason_ = GuardFailReason::kTypeNotEqual;
      return false;
    }
    return true;
  }

  bool CheckDataType(const MetaTensorData &other) const {
    return (data_type_ == nullptr && other.data_type_ == nullptr) ||
           (data_type_ != nullptr && other.data_type_ != nullptr && *data_type_ == *(other.data_type_));
  }

  void StoreTensor(mindspore::tensor::MetaTensorPtr tensor_ptr) {
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    tid_ = tensor_ptr->data_type();
    shape_ = tensor_ptr->shape();
    data_type_ = tensor_ptr->Dtype();
    is_parameter_ = tensor_ptr->is_parameter();
    param_ = tensor_ptr->param_info();
    is_dynamic_shape_ = std::any_of(shape_.begin(), shape_.end(),
                                    [](ShapeValueDType dim) { return dim == kDynamicDim || dim == kDynamicShape; });
  }

  void SubInfo(InfoPack *info) override {
    (*info) << uint8_t(tid_) << data_type_ << is_parameter_ << shape_;
    ParamInfoData::SubInfoInner(info, param_);
  }

  mindspore::TypeId tid_ = TypeId::kTypeUnknown;
  ShapeVector shape_;
  TypePtr data_type_;
  bool is_parameter_ = false;
  bool is_dynamic_shape_ = false;
  mindspore::ParamInfoPtr param_;
};

class TensorData : public MetaTensorData {
 public:
  TensorData(mindspore::tensor::TensorPtr tensor_ptr, bool needSpecialize, int recurseDepth)
      : MetaTensorData(needSpecialize, recurseDepth) {
    tp_ = ItemType::Tensor;
    StoreTensor(tensor_ptr);
  }

  TensorData(PyObject *obj, bool needSpecialize, int recurseDepth) : MetaTensorData(needSpecialize, recurseDepth) {
    tensor_type_ = Py_TYPE(obj);
    tp_ = ItemType::Tensor;
    tensor::TensorPtr tensor_ptr = GetStubInfo(obj);
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    if (OptStrategy::MakeCalcStrategyByShape(tensor_ptr->shape()) != OptStrategy::CalcKind::kCalcValue) {
      specialized_ = false;
    }
    StoreTensor(tensor_ptr);
  }

  ~TensorData() override { data_ptr_.release(); }

  bool IsBaseShapePtr(const TensorData &other) const {
    return (other.base_shape_ptr_ == nullptr && base_shape_ptr_ == other.base_shape_ptr_) ||
           (base_shape_ptr_ != nullptr && other.base_shape_ptr_ != nullptr &&
            *(other.base_shape_ptr_) == *(base_shape_ptr_));
  }

  bool IsCastDtype(const TensorData &other) const {
    return (other.cast_dtype_ == nullptr && cast_dtype_ == nullptr) ||
           (other.cast_dtype_ != nullptr && cast_dtype_ != nullptr && *cast_dtype_ == *(other.cast_dtype_));
  }

  bool operator==(const ItemData &obj) const override {
    if (!ItemData::operator==(obj)) {
      return false;
    }
    bool ret = MetaTensorData::operator==(obj);
    const TensorData &other = static_cast<const TensorData &>(obj);
    ret = ret && other.graph_output_ == graph_output_ && other.specialized_ == specialized_ && IsBaseShapePtr(other) &&
          IsCastDtype(other) && other.compression_type_ == compression_type_ &&
          other.quant_params_.size() == quant_params_.size() && other.tensor_name_.compare(tensor_name_) == 0;
    if (!ret) {
      return ret;
    }
    for (size_t i = 0; i < quant_params_.size(); ++i) {
      if (*(quant_params_[i]) == *(other.quant_params_[i])) {
        continue;
      } else {
        return false;
      }
    }
    if (IsDynamicShape() || other.IsDynamicShape()) {
      return true;
    } else {
      return CheckData(other);
    }
  }

  // how to fast check tensor equal ?
  bool operator==(PyObject *obj) const override {
    bool type_match = Py_IS_TYPE(obj, tensor_type_);
    if (!type_match) {
      type_match = tensor::IsTensorPy(py::cast<py::object>(obj));
    }
    if (!type_match) {
      return false;
    }
    tensor::TensorPtr tensor_ptr = GetStubInfo(obj);
    if (!MetaTensorData::EqualInternal(tensor_ptr)) {
      return false;
    }
    if (!CheckTensorAndParam(tensor_ptr)) {
      return false;
    }
    if (IsDynamicShape() ||
        std::any_of(tensor_ptr->shape().begin(), tensor_ptr->shape().end(),
                    [](ShapeValueDType dim) { return dim == kDynamicDim || dim == kDynamicShape; })) {
      return true;
    }
    return CheckTensorData(tensor_ptr);
  }

  std::string ToString() override {
    std::string tensor = ToStringIntern();
    return DESC(tensor) + DESC_END;
  }

 protected:
  std::string ToStringIntern() override {
    std::string ret = MetaTensorData::ToStringIntern();
    ret += DESC_STRING(graph_output_);
    ret +=
      DESC_TOSTRING(cast_dtype_) + DESC_TOSTRING(base_shape_ptr_) + DESC_STRING(compression_type_) + DESC(tensor_name_);
    for (size_t i = 0; i < quant_params_.size(); ++i) {
      ret += DESC_INDEX(quant_params_, i);
    }
    if (specialized_) {
      ret += ",data_ptr:" + std::to_string(reinterpret_cast<intptr_t>(data_ptr_.get()));
    }
    return ret;
  }

  bool CheckTensorAndParam(const mindspore::tensor::TensorPtr &tensor_ptr) const {
    // tensor_ptr->base_shape_ptr_ should check ?
    auto tensor_tensor_ptr = std::dynamic_pointer_cast<tensor::Tensor>(tensor_ptr);
    if (tensor_tensor_ptr == nullptr) {
      return true;
    }
    // tensor_tensor_ptr->cast_dtype_ should check ?
    const bool ret = tensor_tensor_ptr->IsGraphOutput() == graph_output_ &&
                     tensor_tensor_ptr->compression_type() == compression_type_ &&
                     tensor_tensor_ptr->quant_params().size() == quant_params_.size() &&
                     tensor_tensor_ptr->name() == tensor_name_;
    if (!ret) {
      return false;
    }
    for (size_t i = 0; i < quant_params_.size(); ++i) {
      if (*(quant_params_[i]) == *(tensor_tensor_ptr->quant_params()[i])) {
        continue;
      } else {
        return false;
      }
    }
    return true;
  }

  bool CheckTensorData(const mindspore::tensor::TensorPtr &tensor_ptr) const {
    if (!specialized_) {
      return true;
    }
    // Tensor::data_sync need WaitAll pipeline, it's too expensive
    if (data_ptr_ == nullptr || tensor_ptr->device_address()->GetMutablePtr() == nullptr) {
      // If not data_sync, check data size, but shape and dtype is checked, here return true.
      // Got a tensor which is guard to constant value, but actually it's not synchronized and maybe not constant.
      // return false and mutable it as graph parameter if need.
      return data_len_ == tensor_ptr->DataNBytes();
    } else if (data_len_ == tensor_ptr->DataNBytes()) {
      // if has data_sync, check data
      auto cpu_tensor = tensor_ptr->cpu();
      return memcmp(data_ptr_.get(), reinterpret_cast<const uint8_t *>(cpu_tensor->device_address()->GetMutablePtr()),
                    data_len_) == 0;
    }
    return false;
  }

  bool CheckData(const TensorData &other) const {
    bool ret;
    if (specialized_) {
      if (data_ptr_ == nullptr || other.data_ptr_ == nullptr) {
        ret = data_len_ == other.data_len_;
      } else if (data_len_ == other.data_len_) {
        ret = memcmp(data_ptr_.get(), other.data_ptr_.get(), data_len_) == 0;
      } else {
        ret = false;
      }
    } else {
      ret = data_len_ == other.data_len_;
    }
    return ret;
  }

  void StoreTensor(mindspore::tensor::TensorPtr tensor_ptr) {
    MetaTensorData::StoreTensor(std::static_pointer_cast<tensor::MetaTensor>(tensor_ptr));
    base_shape_ptr_ = tensor_ptr->base_shape_ptr() == nullptr ? nullptr : tensor_ptr->base_shape_ptr()->Clone();

    auto tensor_tensor_ptr = std::dynamic_pointer_cast<tensor::Tensor>(tensor_ptr);
    if (tensor_tensor_ptr != nullptr) {
      graph_output_ = tensor_tensor_ptr->IsGraphOutput();
      cast_dtype_ = (tensor_tensor_ptr->cast_dtype() == nullptr) ? nullptr : tensor_tensor_ptr->cast_dtype()->Clone();
      compression_type_ = tensor_tensor_ptr->compression_type();
      const std::vector<std::shared_ptr<mindspore::QuantizationParam>> &qp = tensor_tensor_ptr->quant_params();
      tensor_name_ = tensor_tensor_ptr->name();
      for (auto quant : qp) {
        QuantizationParamPtr qptr = std::make_shared<mindspore::QuantizationParam>(quant->quant_algo_name());
        quant_params_.push_back(qptr);
        qptr->set_attrs(quant->attrs());
      }
    } else {
      compression_type_ = TensorCompressionType::kNoCompression;
    }
    if (specialized_) {
      auto cpu_tensor = tensor_ptr->cpu();
      data_len_ = size_t(cpu_tensor->DataNBytes());
      data_ptr_ = std::make_unique<uint8_t[]>(data_len_);
      if (data_ptr_ != nullptr) {
        memcpy_s(data_ptr_.get(), data_len_, reinterpret_cast<uint8_t *>(cpu_tensor->device_address()->GetMutablePtr()),
                 data_len_);
      }
    } else {
      data_ptr_.reset(nullptr);
      data_len_ = size_t(tensor_ptr->DataNBytes());
    }
  }

  void SubInfo(InfoPack *info) override {
    MetaTensorData::SubInfo(info);
    (*info) << graph_output_ << cast_dtype_ << base_shape_ptr_ << uint8_t(compression_type_) << tensor_name_;
    (*info) << uint64_t(quant_params_.size());
    for (auto qp : quant_params_) {
      (*info) << qp;
    }
    if (specialized_) {  // hash tensor data. use the real data ?
      (*info) << static_cast<void *>(data_ptr_.get());
    }
  }

  PyTypeObject *tensor_type_;
  std::unique_ptr<uint8_t[]> data_ptr_{nullptr};
  size_t data_len_{0};
  bool graph_output_{false};
  mindspore::abstract::BaseShapePtr base_shape_ptr_{nullptr};
  mindspore::TypePtr cast_dtype_{nullptr};
  mindspore::TensorCompressionType compression_type_;
  std::vector<QuantizationParamPtr> quant_params_;
  std::string tensor_name_;
};
using TensorDataPtr = std::shared_ptr<TensorData>;

static bool Equal(const TensorDataPtr &a, const TensorDataPtr &b, int recurseDepth) {
  if (recurseDepth > 0) {
    if (a == nullptr && b == nullptr) {
      return true;
    } else if (a != nullptr && b != nullptr) {
      return *a == *b;
    } else {
      return false;
    }
  } else {
    return (a == nullptr) == (b == nullptr);
  }
}

static TensorDataPtr CreateTensorData(mindspore::tensor::TensorPtr tensor, bool needSpecialize, int recurseDepth) {
  if (recurseDepth > 0) {
    return (tensor == nullptr) ? nullptr : std::make_shared<TensorData>(tensor, needSpecialize, recurseDepth);
  } else {
    return nullptr;
  }
}

class MapTensorData : public TensorData {
 public:
  MapTensorData(PyObject *obj, bool needSpecialize, int recurseDepth) : TensorData(obj, needSpecialize, recurseDepth) {
    tp_ = ItemType::MapTensor;
    needSpecialize = specialized_;
    auto pyObj = py::cast<py::object>(obj);
    auto tensor_ptr = pyObj.cast<mindspore::tensor::MapTensorPtr>();
    key_dtype_ = tensor_ptr->key_dtype();
    if (tensor_ptr->key_tensor() != nullptr) {
      key_shape_ = tensor_ptr->key_tensor()->shape();
    }
    default_value_ = tensor_ptr->default_value() == nullptr ? nullptr : tensor_ptr->default_value()->type()->Clone();
    permit_filter_value_ =
      tensor_ptr->permit_filter_value() == nullptr ? nullptr : tensor_ptr->permit_filter_value()->type()->Clone();
    evict_filter_value_ =
      tensor_ptr->evict_filter_value() == nullptr ? nullptr : tensor_ptr->evict_filter_value()->type()->Clone();
    value_shape_ = tensor_ptr->value_shape();
    key_tensor_ = CreateTensorData(tensor_ptr->key_tensor(), needSpecialize, recurseDepth);
    value_tensor_ = CreateTensorData(tensor_ptr->value_tensor(), needSpecialize, recurseDepth);
    status_tensor_ = CreateTensorData(tensor_ptr->status_tensor(), needSpecialize, recurseDepth);
  }

  bool IsPermitFilterValue(const MapTensorData &other) const {
    return (other.permit_filter_value_ == nullptr && permit_filter_value_ == nullptr) ||
           (other.permit_filter_value_ != nullptr && permit_filter_value_ != nullptr &&
            *permit_filter_value_ == *(other.permit_filter_value_));
  }

  bool IsDefaultValue(const MapTensorData &other) const {
    return (other.default_value_ == nullptr && default_value_ == nullptr) ||
           (other.default_value_ != nullptr && default_value_ != nullptr && *default_value_ == *(other.default_value_));
  }

  bool IsEvictFilterValue(const MapTensorData &other) const {
    return (other.evict_filter_value_ == nullptr && evict_filter_value_ == nullptr) ||
           (other.evict_filter_value_ != nullptr && evict_filter_value_ != nullptr &&
            *evict_filter_value_ == *(other.evict_filter_value_));
  }

  bool operator==(const ItemData &obj) const override {
    if (!ItemData::operator==(obj)) {
      return false;
    }
    const MapTensorData &other = dynamic_cast<const MapTensorData &>(obj);
    bool ret = TensorData::operator==(obj);
    return ret && other.key_dtype_ == key_dtype_ && other.key_shape_ == key_shape_ && IsDefaultValue(other) &&
           IsPermitFilterValue(other) && IsEvictFilterValue(other) && value_shape_ == other.value_shape_ &&
           Equal(key_tensor_, other.key_tensor_, recurseDepth_) &&
           Equal(value_tensor_, other.value_tensor_, recurseDepth_) &&
           Equal(status_tensor_, other.status_tensor_, recurseDepth_);
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    if (py::isinstance<mindspore::tensor::MapTensor>(obj)) {
      auto pyObj = py::cast<py::object>(obj);
      auto tensor_ptr = pyObj.cast<mindspore::tensor::MapTensorPtr>();
      auto key_dtype = tensor_ptr->key_dtype();
      ShapeVector key_shape;
      if (tensor_ptr->key_tensor() != nullptr) {
        key_shape = tensor_ptr->key_tensor()->shape();
      }
      TypePtr default_value =
        tensor_ptr->default_value() == nullptr ? nullptr : tensor_ptr->default_value()->type()->Clone();
      TypePtr permit_filter_value =
        tensor_ptr->permit_filter_value() == nullptr ? nullptr : tensor_ptr->permit_filter_value()->type()->Clone();
      TypePtr evict_filter_value =
        tensor_ptr->evict_filter_value() == nullptr ? nullptr : tensor_ptr->evict_filter_value()->type()->Clone();
      auto value_shape = tensor_ptr->value_shape();
      bool ret = TensorData::operator==(obj) && key_dtype == key_dtype_ && key_shape == key_shape_ &&
                 CheckType(permit_filter_value_, permit_filter_value) && CheckType(default_value_, default_value) &&
                 CheckType(evict_filter_value_, evict_filter_value);
      if (!ret) {
        return ret;
      }
      return value_shape_ == value_shape &&
             Equal(key_tensor_, CreateTensorData(tensor_ptr->key_tensor(), specialized_, recurseDepth_),
                   recurseDepth_) &&
             Equal(value_tensor_, CreateTensorData(tensor_ptr->value_tensor(), specialized_, recurseDepth_),
                   recurseDepth_) &&
             Equal(status_tensor_, CreateTensorData(tensor_ptr->status_tensor(), specialized_, recurseDepth_),
                   recurseDepth_);
    }
    return false;
  }

  std::string ToString() override {
    std::string map_tensor = ToStringIntern();
    return DESC(map_tensor) + DESC_END;
  }

 protected:
  bool CheckType(const TypePtr &tp1, const TypePtr &tp2) const {
    return (tp1 == nullptr && tp2 == nullptr) || (tp1 != nullptr && tp2 != nullptr && *tp1 == *(tp2));
  }
  std::string ToStringIntern() override {
    return TensorData::ToStringIntern() + DESC_STRING(key_dtype_) + DESC_TOSTRING(default_value_) +
           DESC_TOSTRING(permit_filter_value_) + DESC_TOSTRING(evict_filter_value_) + DESC_TOSTRING(key_tensor_) +
           DESC_TOSTRING(value_tensor_) + DESC_TOSTRING(status_tensor_) + DESC_END;
  }

  void SubInfo(InfoPack *info) override {
    TensorData::SubInfo(info);
    (*info) << key_dtype_ << default_value_ << permit_filter_value_ << evict_filter_value_ << value_shape_
            << key_tensor_->Info() << value_tensor_->Info() << status_tensor_->Info();
  }

  mindspore::TypeId key_dtype_;
  ShapeVector key_shape_;
  TypePtr default_value_;
  TypePtr permit_filter_value_;
  TypePtr evict_filter_value_;
  ShapeVector value_shape_;
  TensorDataPtr key_tensor_;
  TensorDataPtr value_tensor_;
  TensorDataPtr status_tensor_;
};

class RowTensorData : public ItemData {
 public:
  RowTensorData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::RowTensor, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    auto tensor_ptr = pyObj.cast<mindspore::tensor::RowTensorPtr>();
    data_type_ = tensor_ptr->data_type();
    shape_ = tensor_ptr->shape();
    indices_ = CreateTensorData(tensor_ptr->GetIndices(), needSpecialize, recurseDepth);
    values_ = CreateTensorData(tensor_ptr->GetValues(), needSpecialize, recurseDepth);
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const RowTensorData &other = static_cast<const RowTensorData &>(obj);
      return other.data_type_ == data_type_ && other.shape_ == shape_ &&
             Equal(indices_, other.indices_, recurseDepth_) && Equal(values_, other.values_, recurseDepth_);
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    if (py::isinstance<mindspore::tensor::RowTensor>(obj)) {
      auto pyObj = py::cast<py::object>(obj);
      auto tensor_ptr = pyObj.cast<mindspore::tensor::RowTensorPtr>();
      return tensor_ptr->data_type() == data_type_ && tensor_ptr->shape() == shape_ &&
             Equal(indices_, CreateTensorData(tensor_ptr->GetIndices(), specialized_, recurseDepth_), recurseDepth_) &&
             Equal(values_, CreateTensorData(tensor_ptr->GetValues(), specialized_, recurseDepth_), recurseDepth_);
    }
    return false;
  }

  std::string ToString() override {
    std::string row_tensor = DESC_TOSTRING(indices_) + DESC_TOSTRING(values_) + DESC_STRING(data_type_);
    return DESC(row_tensor) + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override { (*info) << indices_->Info() << values_->Info() << data_type_ << shape_; }
  TensorDataPtr indices_;
  TensorDataPtr values_;
  mindspore::TypeId data_type_;
  ShapeVector shape_;
};

class COOTensorData : public ItemData {
 public:
  COOTensorData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::COOTensor, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    auto tensor_ptr = pyObj.cast<mindspore::tensor::COOTensorPtr>();
    data_type_ = tensor_ptr->data_type();
    shape_ = tensor_ptr->shape();
    indices_ = CreateTensorData(tensor_ptr->GetIndices(), needSpecialize, recurseDepth);
    values_ = CreateTensorData(tensor_ptr->GetValues(), needSpecialize, recurseDepth);
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const COOTensorData &other = static_cast<const COOTensorData &>(obj);
      return other.data_type_ == data_type_ && other.shape_ == shape_ &&
             Equal(indices_, other.indices_, recurseDepth_) && Equal(values_, other.values_, recurseDepth_);
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    if (py::isinstance<mindspore::tensor::COOTensor>(obj)) {
      auto pyObj = py::cast<py::object>(obj);
      auto tensor_ptr = pyObj.cast<mindspore::tensor::COOTensorPtr>();
      return tensor_ptr->data_type() == data_type_ && tensor_ptr->shape() == shape_ &&
             Equal(indices_, CreateTensorData(tensor_ptr->GetIndices(), specialized_, recurseDepth_), recurseDepth_) &&
             Equal(values_, CreateTensorData(tensor_ptr->GetValues(), specialized_, recurseDepth_), recurseDepth_);
    }
    return false;
  }

  std::string ToString() override {
    std::string coo_tensor = DESC_TOSTRING(indices_) + DESC_TOSTRING(values_) + DESC_STRING(data_type_);
    return DESC(coo_tensor) + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override { (*info) << indices_->Info() << values_->Info() << data_type_ << shape_; }
  TensorDataPtr indices_;
  TensorDataPtr values_;
  mindspore::TypeId data_type_;
  ShapeVector shape_;
};

class CSRTensorData : public ItemData {
 public:
  CSRTensorData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::CSRTensor, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    auto tensor_ptr = pyObj.cast<mindspore::tensor::CSRTensorPtr>();
    data_type_ = tensor_ptr->data_type();
    shape_ = tensor_ptr->shape();
    indices_ = CreateTensorData(tensor_ptr->GetIndices(), needSpecialize, recurseDepth);
    values_ = CreateTensorData(tensor_ptr->GetValues(), needSpecialize, recurseDepth);
    indptr_ = CreateTensorData(tensor_ptr->GetIndptr(), needSpecialize, recurseDepth);
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const CSRTensorData &other = static_cast<const CSRTensorData &>(obj);
      return other.data_type_ == data_type_ && other.shape_ == shape_ &&
             Equal(indices_, other.indices_, recurseDepth_) && Equal(values_, other.values_, recurseDepth_) &&
             Equal(indptr_, other.indptr_, recurseDepth_);
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    if (py::isinstance<mindspore::tensor::CSRTensor>(obj)) {
      auto pyObj = py::cast<py::object>(obj);
      auto tensor_ptr = pyObj.cast<mindspore::tensor::CSRTensorPtr>();
      return tensor_ptr->data_type() == data_type_ && tensor_ptr->shape() == shape_ &&
             Equal(indices_, CreateTensorData(tensor_ptr->GetIndices(), specialized_, recurseDepth_), recurseDepth_) &&
             Equal(values_, CreateTensorData(tensor_ptr->GetValues(), specialized_, recurseDepth_), recurseDepth_) &&
             Equal(indptr_, CreateTensorData(tensor_ptr->GetIndptr(), specialized_, recurseDepth_), recurseDepth_);
    }
    return false;
  }

  std::string ToString() override {
    std::string csr_tensor =
      DESC_TOSTRING(indices_) + DESC_TOSTRING(values_) + DESC_TOSTRING(indptr_) + DESC_STRING(data_type_);
    return DESC(csr_tensor) + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override {
    (*info) << indices_->Info() << values_->Info() << indptr_->Info() << data_type_ << shape_;
  }
  TensorDataPtr indices_;
  TensorDataPtr values_;
  TensorDataPtr indptr_;
  mindspore::TypeId data_type_;
  ShapeVector shape_;
};

class TensorDataData : public ItemData {
 public:
  TensorDataData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::Tensordata, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    auto data = pyObj.cast<mindspore::tensor::TensorDataPtr>();
    size_ = (uint64_t)data->size();
    itemsize_ = (uint64_t)data->itemsize();
    nbytes_ = (uint64_t)data->nbytes();
    ndim_ = (int64_t)data->ndim();
    if (specialized_) {
      data_ptr_ = std::make_unique<uint8_t[]>(nbytes_);
      if (data_ptr_ != nullptr) {
        memcpy_s(data_ptr_.get(), nbytes_, reinterpret_cast<uint8_t *>(data->data()), nbytes_);
      }
    } else {
      data_ptr_.reset(nullptr);
    }
  }

  ~TensorDataData() override { data_ptr_.release(); }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const TensorDataData &other = static_cast<const TensorDataData &>(obj);
      if (specialized_) {
        return data_ptr_ != nullptr && other.data_ptr_ != nullptr && nbytes_ == other.nbytes_ &&
               memcmp(data_ptr_.get(), other.data_ptr_.get(), nbytes_) == 0;
      } else {
        return size_ == other.size_ && itemsize_ == other.itemsize_ && nbytes_ == other.nbytes_ &&
               ndim_ == other.ndim_ && (data_ptr_ == nullptr) == (other.data_ptr_ == nullptr);
      }
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    if (py::isinstance<mindspore::tensor::TensorData>(obj)) {
      auto pyObj = py::cast<py::object>(obj);
      auto data = pyObj.cast<mindspore::tensor::TensorDataPtr>();
      if (specialized_) {
        return data_ptr_ != nullptr && reinterpret_cast<uint8_t *>(data->data()) != nullptr &&
               nbytes_ == (uint64_t)data->nbytes() &&
               memcmp(data_ptr_.get(), reinterpret_cast<uint8_t *>(data->data()), nbytes_) == 0;
      } else {
        return size_ == (uint64_t)data->size() && itemsize_ == (uint64_t)data->itemsize() &&
               nbytes_ == (uint64_t)data->nbytes() && ndim_ == (int64_t)data->ndim() &&
               (data_ptr_ == nullptr) == (reinterpret_cast<uint8_t *>(data->data()) == nullptr);
      }
    }
    return false;
  }

  std::string ToString() override {
    std::string tensor_data = DESC_STRING(size_) + DESC_STRING(itemsize_) + DESC_STRING(nbytes_) + DESC_STRING(ndim_);
    if (specialized_) {
      tensor_data += ",data_ptr:" + std::to_string(reinterpret_cast<intptr_t>(data_ptr_.get()));
    }
    return DESC(tensor_data) + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override {
    (*info) << size_ << itemsize_ << nbytes_ << ndim_;
    if (specialized_) {
      (*info) << data_ptr_.get();
    }
  }
  std::unique_ptr<uint8_t[]> data_ptr_;
  uint64_t size_;
  uint64_t itemsize_;
  uint64_t nbytes_;
  int64_t ndim_;
};

class PrimitiveData : public ItemData {
 public:
  PrimitiveData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::Primitive, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    auto data = pyObj.cast<PrimitivePyAdapterPtr>();
    py::dict pd = data->GetAttrDict();
    auto dct = pd.ptr();
    Py_ssize_t pos = 0;
    PyObject *key;
    PyObject *val;
    while (PyDict_Next(dct, &pos, &key, &val)) {
      ItemDataPtr k;
      ItemDataPtr v;
      if (recurseDepth > 0 || needSpecialize) {
        k = CreateItem(key, needSpecialize, recurseDepth);
        v = CreateItem(val, needSpecialize, recurseDepth);
      } else {
        k =
          CreateItem((key == NULL || key == Py_None) ? NULL : reinterpret_cast<PyObject *>(Py_TYPE(key)), false, false);
        v =
          CreateItem((val == NULL || val == Py_None) ? NULL : reinterpret_cast<PyObject *>(Py_TYPE(val)), false, false);
      }
      listK_.push_back(k);
      listV_.push_back(v);
    }
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const PrimitiveData &other = static_cast<const PrimitiveData &>(obj);
      if (other.listK_.size() == listK_.size() && other.listV_.size() == listV_.size()) {
        for (size_t i = 0; i < listK_.size(); ++i) {
          if (*(listK_[i]) == *(other.listK_[i]) && *(listV_[i]) == *(other.listV_[i])) {
            continue;
          } else {
            return false;
          }
        }
        return true;
      }
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    if (obj == nullptr) {
      return false;
    }
    if (py::isinstance<mindspore::PrimitivePyAdapter>(obj)) {
      auto pyObj = py::cast<py::object>(obj);
      auto data = pyObj.cast<PrimitivePyAdapterPtr>();
      py::dict pd = data->GetAttrDict();
      if (pd.size() != listK_.size()) {
        return false;
      }
      auto dct = pd.ptr();
      Py_ssize_t pos = 0;
      PyObject *key;
      PyObject *val;
      size_t i = 0;
      while (PyDict_Next(dct, &pos, &key, &val)) {
        if (!(*(listK_[i]) == key && *(listV_[i]) == val)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  std::string ToString() override {
    std::string primitive;
    for (size_t i = 0; i < listK_.size(); ++i) {
      primitive += DESC_ITEM(listK_[i], listV_[i]);
    }
    return DESC(primitive) + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override {
    (*info) << uint64_t(listK_.size());
    for (auto item : listK_) {
      (*info) << item->Info();
    }
    (*info) << uint64_t(listV_.size());
    for (auto item : listV_) {
      (*info) << item->Info();
    }
  }
  std::vector<ItemDataPtr> listK_;
  std::vector<ItemDataPtr> listV_;
};

class CellData : public ItemData {
 public:
  CellData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::Cell, needSpecialize, recurseDepth),
        cell_ref_(obj),
        training_name_("training"),
        requires_grad_name_("requires_grad"),
        training_(false),
        requires_grad_(false) {
    auto training_attr = py::getattr(obj, training_name_, nullptr);
    auto requires_grad_attr = py::getattr(obj, requires_grad_name_, nullptr);
    PyObject *self_dict = PyObject_GenericGetDict(obj, nullptr);
    py::object scope_ref = py::reinterpret_steal<py::object>(self_dict);
    if (scope_ref.ptr() == nullptr) {
      throw py::error_already_set();  // no self.__dict__
    }
    if (training_attr.ptr() != nullptr) {
      if (training_attr.ptr() != PyDict_GetItem(self_dict, training_name_.ptr())) {
        MS_LOG(ERROR) << "self.training is not self.__dict__[\"training\"]";
      }
      training_ = training_attr.cast<bool>();
    }
    if (requires_grad_attr.ptr() != nullptr) {
      if (requires_grad_attr.ptr() != PyDict_GetItem(self_dict, requires_grad_name_.ptr())) {
        MS_LOG(ERROR) << "self.requires_grad is not self.__dict__[\"requires_grad\"]";
      }
      requires_grad_ = requires_grad_attr.cast<bool>();
    }
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const auto &other = static_cast<const CellData &>(obj);
      void *this_cell = PyWeakref_GET_OBJECT(this->cell_ref_.ptr());
      void *other_cell = PyWeakref_GET_OBJECT(other.cell_ref_.ptr());
      return this_cell == other_cell && training_ == other.training_ && requires_grad_ == other.requires_grad_;
    }
    return false;
  }

  bool operator==(PyObject *obj) const override {
    py::object strong_ref = py::reinterpret_borrow<py::object>(PyWeakref_GET_OBJECT(cell_ref_.ptr()));
    if (strong_ref.ptr() == Py_None || strong_ref.ptr() != obj) {
      return false;  // always false if cell object is freed, or id not match
    }
    // id match. obj is a mindspore::Cell
    PyObject *self_dict = PyObject_GenericGetDict(strong_ref.ptr(), nullptr);
    py::object scope_ref = py::reinterpret_steal<py::object>(self_dict);
    if (scope_ref.ptr() == nullptr) {
      PyErr_Clear();
      return false;
    }
    // assume self.training and self.requires_grad is in self.__dict__
    return (PyDict_GetItem(self_dict, training_name_.ptr()) == Py_True) == training_ &&
           (PyDict_GetItem(self_dict, requires_grad_name_.ptr()) == Py_True) == requires_grad_;
  }

  std::string ToString() override {
    PyObject *weak_ref = PyWeakref_GET_OBJECT(cell_ref_.ptr());
    PyObject *py_type = reinterpret_cast<PyObject *>(Py_TYPE(weak_ref));
    std::string type = weak_ref == Py_None ? "(freed)" : py::getattr(py_type, "__name__").cast<std::string>();

    std::stringstream ss;
    ss << static_cast<void *>(weak_ref);
    const std::string &ptr = ss.str();

    std::string cell;
    cell += DESC(type);
    cell += DESC(ptr);
    cell += DESC_STRING(training_);
    cell += DESC_STRING(requires_grad_);
    return DESC(cell) + DESC_END;
  }
  PyTypeObject *GetTypeObject() {
    PyObject *weak_ref = PyWeakref_GET_OBJECT(cell_ref_.ptr());
    return Py_TYPE(weak_ref);
  }

 protected:
  void SubInfo(InfoPack *info) override {
    void *ptr = PyWeakref_GET_OBJECT(cell_ref_.ptr());
    (*info) << ptr << training_ << requires_grad_;
  }

 private:
  py::weakref cell_ref_;
  py::str training_name_;
  py::str requires_grad_name_;
  bool training_;
  bool requires_grad_;
};

class UnknownData : public ItemData {
 public:
  UnknownData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyUnknown, needSpecialize, recurseDepth), refId_(py::reinterpret_borrow<py::object>(obj)) {}

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      return refId_.ptr() == (static_cast<const UnknownData &>(obj)).refId_.ptr();
    }
    return false;
  }

  bool operator==(PyObject *obj) const override { return refId_ == obj; }

  std::string ToString() override {
    std::string ret = "unknown:";
    return ret + std::to_string(reinterpret_cast<intptr_t>(refId_.ptr())) + ItemData::ToString();
  }

 protected:
  void SubInfo(InfoPack *info) override { (*info_) << refId_.ptr(); }

  /**
   * strong reference avoid memory address reuse
   * guard live time is same as function.__code__, which meaning object is not released until code object release.
   * remove unknown data and all unknown data must be transformed known data by trace. if not, check guard
   * generated
   */
  py::object refId_;
};

ListData::ListData(PyObject *obj, bool needSpecialize, int recurseDepth)
    : ItemData(ItemType::PyList, needSpecialize, recurseDepth) {
  if (PyList_Check(obj)) {
    InitList(obj, needSpecialize, recurseDepth);
  } else if (PyTuple_Check(obj)) {
    InitTuple(obj, needSpecialize, recurseDepth);
  } else if (PySet_Check(obj)) {
    InitSet(obj, needSpecialize, recurseDepth);
  } else if (PyFrozenSet_Check(obj)) {
    InitFrozenSet(obj, needSpecialize, recurseDepth);
  }
  if (needSpecialize) {
    return;  // value check exactly
  }
  for (const auto &data : listVar_) {
    if (data->GetItemType() == ItemType::PyType) {
      static_cast<TypeData *>(data.get())->set_ambiguous_tensor_type(true);
    }
  }
}

using CheckPyObjectFunc = bool (*)(PyObject *obj);
using CreatePyObjectFunc = ItemDataPtr (*)(PyObject *obj, bool need_specialize, int recurse_depth);
template <typename T>
ItemDataPtr CreatePyData(PyObject *obj, bool need_specialize, int recurse_depth) {
  return std::make_shared<T>(obj, need_specialize, recurse_depth);
}
template <typename T>
ItemDataPtr CreateMutablePyData(PyObject *obj, bool need_specialize, int recurse_depth) {
  return std::make_shared<T>(obj, false, recurse_depth);
}
static bool CheckMetaTensorObject(PyObject *obj) { return tensor::IsTensorPy(py::cast<py::object>(obj)); }

static bool CheckTensorObject(PyObject *obj) { return tensor::IsTensorPy(py::cast<py::object>(obj)); }

static bool CheckDictKeyValueItemObject(PyObject *obj) {
  return !!PyDict_Check(obj) || !!PyDictKeys_Check(obj) || !!PyDictValues_Check(obj) || !!PyDictItems_Check(obj);
}
static const std::vector<std::pair<CheckPyObjectFunc, CreatePyObjectFunc>> kFuncPyObjectConverter = {
  {[](PyObject *obj) -> bool { return PyLong_Check(obj) && !PyBool_Check(obj); }, CreatePyData<IntData>},
  {[](PyObject *obj) -> bool { return !!PyFloat_Check(obj); }, CreatePyData<FloatData>},
  {[](PyObject *obj) -> bool { return !!PyBool_Check(obj); }, CreatePyData<BoolData>},
  {[](PyObject *obj) -> bool { return !!PyBytes_Check(obj); }, CreatePyData<BytesData>},
  {[](PyObject *obj) -> bool { return !!PyUnicode_Check(obj); }, CreatePyData<StringData>},
  {[](PyObject *obj) -> bool { return !!PyList_Check(obj); }, CreatePyData<ListData>},
  {[](PyObject *obj) -> bool { return !!PyTuple_Check(obj); }, CreatePyData<ListData>},
  {[](PyObject *obj) -> bool { return !!PySet_Check(obj); }, CreatePyData<ListData>},
  {[](PyObject *obj) -> bool { return !!PyFrozenSet_Check(obj); }, CreatePyData<ListData>},
  {[](PyObject *obj) -> bool { return CheckDictKeyValueItemObject(obj); }, CreatePyData<DictData>},
  {[](PyObject *obj) -> bool { return !!PyComplex_Check(obj); }, CreatePyData<ComplexData>},
  {[](PyObject *obj) -> bool { return !!PySlice_Check(obj); }, CreatePyData<SliceData>},
  {[](PyObject *obj) -> bool { return !!PyFunction_Check(obj); }, CreatePyData<FunctionData>},
  {[](PyObject *obj) -> bool { return !!PyMethod_Check(obj); }, CreatePyData<MethodData>},
  {[](PyObject *obj) -> bool { return !!PyInstanceMethod_Check(obj); }, CreatePyData<InstanceMethodData>},
  {[](PyObject *obj) -> bool { return !!PyType_Check(obj); }, CreatePyData<TypeData>},
  {[](PyObject *obj) -> bool { return py::isinstance<py::array>(obj); }, CreatePyData<NumpyData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::Type>(obj); }, CreatePyData<TensorTypeData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::tensor::MapTensor>(obj); },
   CreatePyData<MapTensorData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::ParamInfo>(obj); }, CreatePyData<ParamInfoData>},
  {[](PyObject *obj) -> bool { return CheckTensorObject(obj); }, CreatePyData<TensorData>},
  {[](PyObject *obj) -> bool { return CheckMetaTensorObject(obj); }, CreatePyData<MetaTensorData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::tensor::TensorData>(obj); },
   CreatePyData<TensorDataData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::PrimitivePyAdapter>(obj); },
   CreatePyData<PrimitiveData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::Cell>(obj); }, CreatePyData<CellData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::tensor::RowTensor>(obj); },
   CreateMutablePyData<RowTensorData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::tensor::COOTensor>(obj); },
   CreateMutablePyData<COOTensorData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::tensor::CSRTensor>(obj); },
   CreateMutablePyData<CSRTensorData>},
};

static ItemDataPtr CreateData(PyObject *obj, bool need_specialize, int recurse_depth) {
  auto tar =
    std::find_if(kFuncPyObjectConverter.begin(), kFuncPyObjectConverter.end(),
                 [obj](const std::pair<CheckPyObjectFunc, CreatePyObjectFunc> &func) { return func.first(obj); });
  if (tar != kFuncPyObjectConverter.end()) {
    return tar->second(obj, need_specialize, recurse_depth);
  } else {
    return std::make_shared<UnknownData>(obj, need_specialize, recurse_depth);
  }
}

static ItemDataPtr CreateItem(PyObject *obj, bool need_specialize, int recurse_depth) {
  ReprRecursionScope scope(obj);
  if (scope.ReEnterOrError()) {
    return std::make_shared<ItemData>(ItemType::PyNull, need_specialize, recurse_depth);
  }
  if (recurse_depth < -1) {
    if (obj != NULL && obj != Py_None) {
      PyObject *py_type = reinterpret_cast<PyObject *>(Py_TYPE(obj));
      return std::make_shared<TypeData>(py_type, false, 0);
    } else {
      return std::make_shared<ItemData>(ItemType::PyNull, false, 0);
    }
  }
  recurse_depth -= 1;
  ItemDataPtr dp;
  if (obj != NULL && obj != Py_None) {
    dp = CreateData(obj, need_specialize, recurse_depth);
  } else {
    dp = std::make_shared<ItemData>(ItemType::PyNull, need_specialize, recurse_depth);
  }
  return dp;
}

GuardItem::GuardItem(TracePtr tt)
    : var_(tt), type_(GIType::GTUnknown), info_(nullptr), fail_count_(0), perf_(false), checked_(false) {}

void GuardItem::UpdateTrace(std::map<size_t, TracePtr> *unique_cache) { this->var_ = var_->UniqueAll(unique_cache); }

void GuardItem::Cache(bool success) {
  checked_ = true;
  fail_count_ = fail_count_ + !success;
  GuardContext::Data::GetInstance()->guard_cache().push_back(this);
}

void GuardItem::ClearCache() {
  fail_count_ = 0;
  checked_ = false;
}

GuardItemPtr GuardItem::Optimize() {
  auto trace = var_->Optimize();
  if (trace != nullptr) {
    var_ = trace;
    info_ = nullptr;
    Info();
    return shared_from_this();
  } else {
    return nullptr;
  }
}

TracePtr GuardItem::GetTrace() const { return var_; }

bool GuardItem::operator==(const GuardItem &obj) const { return type_ == obj.type_ && *var_ == *(obj.var_); }

static constexpr int kGuardItemTotalStage = 2;
static constexpr int kGuardItemRetrieveStage = 0;
static constexpr int kGuardItemCompareStage = 1;

class EqGuard : public GuardItem {
 public:
  EqGuard(TracePtr obj, bool needSpecialize, int recurseDepth)
      : GuardItem(obj),
        dp_(CreateItem(obj->GetObject(), needSpecialize, recurseDepth)),
        last_(py::reinterpret_borrow<py::object>(obj->GetObject())),
        specialized_(needSpecialize),
        recurse_(recurseDepth) {
    type_ = GIType::GTEqual;
    if (dp_->GetItemType() == ItemType::Tensor || dp_->GetItemType() == ItemType::Cell ||
        dp_->GetItemType() == ItemType::PyUnknown) {
      last_ = {};  // reference release
    }
  }

  virtual ~EqGuard() {}

  bool Check(PyFrameWrapper frame) override {
    dp_->fail_reason_ = GuardFailReason::kReasonUnknown;
    if (var_->IsConst()) {
      return true;
    }
    if (!var_->IsSpecialized() && var_->GetRelaxCount() > 0 && !RootTrace::Support(var_->GetTraceType()) &&
        !specialized_) {
      // it just needs to guard inputs instead of leaf node
      return true;
    }
    py::object obj = GetObjectFromTrace(frame, var_);
    bool ret = Check(obj.ptr());
    return ret;
  }

  bool Check(PyObject *obj) override {
    if (obj == nullptr) {
      return false;
    }
    return obj == last_ || CheckData(obj);
  }

  bool CheckData(PyObject *obj) {
    // It is faster to use PyObject* to compare than createitem
    return *dp_ == obj;
  }

  virtual std::string ToString() {
    if (!strGuard_.empty()) {
      return strGuard_;
    }
    strGuard_ = std::string("EqGuard(") + var_->ToString() + " == " + dp_->ToString() + ")";
    if (location_ != "") {
      strGuard_ += std::string(", location: ") + location_;
    }
    strGuard_ = std::regex_replace(strGuard_, std::regex("(\n)"), " ");

    return strGuard_;
  }

  std::string GetFailInfo() override {
    if (dp_->fail_reason_ == GuardFailReason::kReasonUnknown) {
      return std::string("");
    }
    return GetGuardFailReasonDesc(dp_->fail_reason_);
  }

  virtual const InfoPack &Info() {
    if (info_ == nullptr) {
      info_ = std::make_shared<InfoPack>();
      InfoPack &info = *info_;
      info << uint8_t(type_);
      info.Begin();
      info << var_->Info() << dp_->Info();
      info.End();
      info_->Update();
    }
    return *info_;
  }

  bool operator==(const GuardItem &obj) const override {
    if (GuardItem::operator==(obj)) {
      auto other = static_cast<const EqGuard &>(obj);
      return specialized_ == other.specialized_ && recurse_ == other.recurse_ && *dp_ == *(other.dp_);
    }
    return false;
  }

  py::object MakeDynamicShape(const tensor::TensorPtr &new_tensor) {
    MS_EXCEPTION_IF_NULL(new_tensor);
    auto type = dp_->GetItemType();
    if (type != ItemType::MetaTensor && type != ItemType::Tensor) {
      return {};
    }
    auto item = (MetaTensorData &)(*dp_);
    auto tensor_dtype = new_tensor->Dtype();
    MS_EXCEPTION_IF_NULL(tensor_dtype);
    if (item.data_type()->type_id() != tensor_dtype->type_id()) {
      return {};  // can't symbolic tensor data type
    }
    ShapeVector new_shape;
    bool has_dynamic_shape = false;
    if (item.shape().size() == new_tensor->shape().size()) {
      for (size_t i = 0; i < new_tensor->shape().size(); ++i) {
        new_shape.push_back(item.shape()[i] == new_tensor->shape()[i] ? item.shape()[i] : kDynamicShape);
        has_dynamic_shape |= item.shape()[i] != new_tensor->shape()[i];
      }
    } else {
      new_shape.push_back(kDynamicDim);
      has_dynamic_shape = true;
    }
    if (!has_dynamic_shape) {
      MS_LOG(INFO) << "maybe not a dynamic shape";
      return {};
    }
    tensor::TensorPtr copy_tensor = std::make_shared<tensor::Tensor>(new_tensor->Dtype()->type_id(), new_shape);
    auto tensor_tensor_ptr = std::dynamic_pointer_cast<tensor::Tensor>(new_tensor);
    if (tensor_tensor_ptr != nullptr) {
      if (tensor_tensor_ptr->IsGraphOutput()) {
        copy_tensor->SetIsGraphOutput();
      }
      if (tensor_tensor_ptr->cast_dtype() != nullptr) {
        copy_tensor->set_cast_dtype(tensor_tensor_ptr->cast_dtype()->Clone());
      }
      copy_tensor->set_quant_param(tensor_tensor_ptr->quant_params());
      copy_tensor->set_name(tensor_tensor_ptr->name());
    }
    return PackTensorToPyObject(copy_tensor);
  }

  bool specialized() const { return specialized_; }
  const auto &Item() const { return dp_; }
  const auto &GetObject() const { return last_; }

 protected:
  ItemDataPtr dp_;
  // strong reference
  py::object last_;
  bool specialized_;
  int recurse_;
};

class TypeGuard : public GuardItem {
 public:
  explicit TypeGuard(TracePtr obj) : GuardItem(obj) {
    type_ = GIType::GTType;
    is_const_ = false;
    refType_ = Py_TYPE(obj->GetObject());
    is_tensor_ = IsTensorOrStubTensor(refType_);
    if (obj->GetRelaxCount() > 0) {
      check_count_ = 0;
    } else {
      check_count_ = -1;
    }
  }

  bool Check(PyFrameWrapper frame) override {
    if (var_->IsConst() || is_const_) {
      return true;
    }
    py::object obj = GetObjectFromTrace(frame, var_);
    bool ret = Check(obj.ptr());
    if (check_count_ >= 0) {
      if (!ret) {
        check_count_ = -1;
      } else {
        check_count_++;
        if (check_count_ > var_->GetRelaxCount()) {
          is_const_ = true;
        }
      }
    }
    return ret;
  }

  bool Check(PyObject *obj) override {
    if (obj == NULL) {
      return false;
    }
    PyTypeObject *tp = Py_TYPE(obj);
    if (tp != refType_) {
      if (is_tensor_) {
        return IsTensorOrStubTensor(tp);
      }
      return false;
    } else {
      return true;
    }
  }

  std::string ToString() override {
    if (strGuard_.size() > 0) {
      return strGuard_;
    }
    if (var_->GetTraceType() == TraceType::Type) {
      strGuard_ = var_->ToString() + std::string("==") + refType_->tp_name;
    } else {
      strGuard_ = std::string("type(") + var_->ToString() + std::string(")==") + refType_->tp_name;
    }
    strGuard_ = std::regex_replace(strGuard_, std::regex("(\n)"), "");
    return strGuard_;
  }

  virtual const InfoPack &Info() {
    if (info_ == nullptr) {
      InfoPack info;
      info << uint8_t(type_);
      info.Begin();
      info << var_->Info() << refType_->tp_name;
      info.End();
      info_ = std::make_shared<InfoPack>(info);
      info_->Update();
    }
    return *info_;
  }

  bool operator==(const GuardItem &obj) const override {
    if (GuardItem::operator==(obj)) {
      return refType_ == (static_cast<const TypeGuard &>(obj)).refType_;
    }
    return false;
  }

  auto ref_type() const { return refType_; }

 private:
  static bool IsTensorOrStubTensor(PyTypeObject *type) { return type != nullptr && IsTensorType<true>(type); }

 protected:
  PyTypeObject *refType_;
  int check_count_;
  bool is_const_;
  bool is_tensor_{false};
};

class IdGuard : public GuardItem {
 public:
  explicit IdGuard(TracePtr obj) : GuardItem(obj), refId_(py::reinterpret_borrow<py::object>(obj->GetObject())) {
    type_ = GIType::GTId;
  }

  bool Check(PyFrameWrapper frame) override {
    if (var_->IsConst()) {
      return true;
    }
    py::object obj = GetObjectFromTrace(frame, var_);
    bool ret = Check(obj.ptr());
    return ret;
  }

  bool Check(PyObject *obj) override {
    bool ret = false;
    if (obj == NULL) {
      return ret;
    }
    if (obj != refId_) {
      ret = false;
    } else {
      ret = true;
    }
    return ret;
  }

  std::string ToString() override {
    if (strGuard_.size() > 0) {
      return strGuard_;
    }
    std::stringstream s;
    s << "id(" << var_->ToString() << ")==" << refId_.ptr();
    strGuard_ = s.str();
    return strGuard_;
  }

  virtual const InfoPack &Info() {
    if (info_ == nullptr) {
      InfoPack info;
      info << uint8_t(type_);
      info.Begin();
      info << var_->Info() << reinterpret_cast<void *>(refId_.ptr());
      info.End();
      info_ = std::make_shared<InfoPack>(info);
      info_->Update();
    }
    return *info_;
  }

  bool operator==(const GuardItem &obj) const override {
    if (GuardItem::operator==(obj)) {
      return refId_ == (static_cast<const IdGuard &>(obj)).refId_;
    }
    return false;
  }

 protected:
  // strong reference avoid memory reused
  py::object refId_;
};

class ReprGuard : public GuardItem {
 public:
  explicit ReprGuard(TracePtr obj) : GuardItem(obj) {
    type_ = GIType::GTRepr;
    refRepr_ = PyObject_Repr(obj->GetObject());
  }

  virtual ~ReprGuard() { Py_XDECREF(refRepr_); }

  bool Check(PyFrameWrapper frame) override {
    if (var_->IsConst()) {
      return true;
    }
    py::object obj = GetObjectFromTrace(frame, var_);
    bool ret = Check(obj.ptr());
    return ret;
  }

  bool Check(PyObject *obj) override {
    bool ret = false;
    if (obj == nullptr) {
      return ret;
    }
    PyObject *repr_flag = GetAttrReprCacheStr();
    PyObject *repr;
    bool from_cache = false;
    if (PyObject_HasAttr(obj, repr_flag)) {
      from_cache = true;
      repr = PyObject_GetAttr(obj, repr_flag);
    } else {
      repr = PyObject_Repr(obj);
      PyObject_SetAttr(obj, repr_flag, repr);
    }
    if (PyUnicode_Compare(repr, refRepr_)) {
      // Inplace operation may change the object without clearing the cache of guard.
      if (from_cache && !PyUnicode_Compare(PyObject_Repr(obj), refRepr_)) {
        ret = true;
      } else {
        ret = false;
      }
    } else {
      ret = true;
    }
    Py_XDECREF(repr);
    return ret;
  }

  std::string ToString() override {
    if (strGuard_.size() > 0) {
      return strGuard_;
    }
    strGuard_ = std::string(PyUnicode_AsUTF8(refRepr_));
    strGuard_ = std::regex_replace(strGuard_, std::regex("(\n)"), "");
    return strGuard_;
  }

  bool operator==(const GuardItem &obj) const override {
    if (GuardItem::operator==(obj)) {
      int ret = PyUnicode_Compare(refRepr_, static_cast<const ReprGuard &>(obj).refRepr_);
      if (ret == 0) {
        return true;
      } else if (ret == -1 && PyErr_Occurred()) {
        // This function returns -1 upon failure, so one should call PyErr_Occurred() to check for errors.
        // Refer to: https://docs.python.org/3/c-api/unicode.html
        MS_LOG(INFO) << "Python error occurs when comparing two ReprGuards";
        PyErr_Clear();
      }
    }
    return false;
  }

 protected:
  virtual const InfoPack &Info() {
    if (info_ == nullptr) {
      InfoPack info;
      info << uint8_t(type_);
      info.Begin();
      info << std::string(PyUnicode_AsUTF8(refRepr_));
      info.End();
      info_ = std::make_shared<InfoPack>(info);
      info_->Update();
    }
    return *info_;
  }
  PyObject *refRepr_;
};

/* ======================================= MatchIDGuard ======================================= */

class MatchIDGuard : public GuardItem {
 public:
  explicit MatchIDGuard(const TracePtr &tr) : GuardItem(tr) {
    type_ = GIType::kMatchIDS;
    items_.push_back(tr);
  }
  bool Check(PyFrameWrapper frame) override;
  bool Check(PyObject *obj) override { return false; }
  std::string ToString() override;
  const InfoPack &Info() override;
  bool operator==(const GuardItem &obj) const override;
  void AddAlias(const TracePtr &i);

 private:
  std::vector<TracePtr> items_;
};

bool MatchIDGuard::Check(PyFrameWrapper frame) {
  size_t index = 0;
  size_t items_size = items_.size();

  // all items has same object id
  py::object object_p = GetObjectFromTrace(frame, items_[index++]);
  void *expected = object_p.ptr();
  for (; object_p.ptr() != nullptr && object_p.ptr() == expected && index < items_size; ++index) {
    object_p = GetObjectFromTrace(frame, items_[index]);
  }
  bool ret = expected == object_p.ptr() && index == items_size;
  return ret;
}

std::string MatchIDGuard::ToString() {
  std::stringstream s;
  s << "MatchIDGuard: ";
  for (size_t i = 0, size = items_.size() - 1; i != size; ++i) {
    s << "[" << items_[i]->ToString() << "] is ";
  }
  s << "[" << items_.back()->ToString() << "]";
  std::string ret = s.str();
  return ret;
}

const InfoPack &MatchIDGuard::Info() {
  if (info_ != nullptr) {
    return *info_;
  }
  info_ = std::make_shared<InfoPack>();
  ((*info_) << static_cast<uint8_t>(type_)).Begin();
  (*info_) << static_cast<void *>(items_[0]->GetObject());
  info_->End().Update();
  return *info_;
}

bool MatchIDGuard::operator==(const GuardItem &obj) const {
  if (type_ != obj.GetType()) {
    return false;
  }
  const MatchIDGuard &other = static_cast<const MatchIDGuard &>(obj);
  return this->items_.size() == other.items_.size() && GetTrace()->GetObject() == other.GetTrace()->GetObject();
}

void MatchIDGuard::AddAlias(const TracePtr &i) {
  if (std::any_of(items_.begin(), items_.end(), [&i](const TracePtr &j) { return i->Info().Id() == j->Info().Id(); })) {
    return;
  }
  items_.push_back(i);
}

GuardItemPtr GuardIDS(const TracePtr &tr, const GuardItemPtr &reused) {
  if (reused == nullptr || reused->GetType() != GIType::kMatchIDS) {
    return std::make_shared<MatchIDGuard>(tr);
  }
  static_cast<MatchIDGuard *>(reused.get())->AddAlias(tr);
  return reused;
}

/* ============================================================================================= */

GuardItemPtr GuardEqual(TracePtr obj, bool needSpecialize, int recurseDepth) {
  return std::make_shared<EqGuard>(obj, needSpecialize, recurseDepth);
}

GuardItemPtr GuardType(TracePtr obj) { return std::make_shared<TypeGuard>(obj); }

GuardItemPtr GuardId(TracePtr obj) { return std::make_shared<IdGuard>(obj); }

GuardItemPtr GuardRepr(TracePtr obj) { return std::make_shared<ReprGuard>(obj); }

bool IsPyObjectEqual(PyObject *src, PyObject *dst) {
  if (src == dst) {
    return true;
  }
  ItemDataPtr src_item = CreateItem(src, true, INT_MAX);
  ItemDataPtr dst_item = CreateItem(dst, true, INT_MAX);
  return *src_item == *dst_item;
}

static PyObject *g_ms_module = nullptr;
static PyObject *g_ms_type = nullptr;
static PyObject *g_tensor_type = nullptr;

static bool InitMsModule() {
  if (g_ms_module == nullptr) {
    g_ms_module = PyImport_ImportModule("mindspore");
  }
  return g_ms_module != nullptr && g_ms_module != Py_None;
}

static bool InitMsType() {
  if (g_ms_type == NULL) {
    g_ms_type = PyImport_ImportModule("mindspore.common.dtype");
  }
  return g_ms_type != NULL && g_ms_type != Py_None;
}

static bool InitMsTensor() {
  if (g_tensor_type == nullptr && InitMsModule()) {
    g_tensor_type = PyObject_GetAttrString(g_ms_module, "Tensor");
  }
  return g_tensor_type != nullptr && g_tensor_type != Py_None && PyType_Check(g_tensor_type);
}

PyObject *GetMsModule() {
  if (InitMsModule()) {
    return g_ms_module;
  } else {
    return nullptr;
  }
}

PyObject *GetMsType() {
  if (InitMsType()) {
    return g_ms_type;
  } else {
    return nullptr;
  }
}

PyObject *GetMsTensorType() {
  if (InitMsTensor()) {
    return g_tensor_type;
  } else {
    return nullptr;
  }
}

template <typename S>
ValuePtr CastScalarToScalar(S in, const TypeId &type_id) {
  switch (type_id) {
    case kNumberTypeInt32:
      return MakeValue(static_cast<int>(in));
    case kNumberTypeFloat16:
      return MakeValue(static_cast<float16>(in).int_value());
    case kNumberTypeFloat32:
      return MakeValue(static_cast<float>(in));
    case kNumberTypeBool:
      return MakeValue(static_cast<bool>(in));
    case kNumberTypeInt64:
      return MakeValue(static_cast<int64_t>(in));
    case kNumberTypeFloat64:
      return MakeValue(static_cast<double>(in));
    case kNumberTypeInt16:
      return MakeValue(static_cast<int16_t>(in));
    case kNumberTypeInt8:
      return MakeValue(static_cast<int8_t>(in));
    case kNumberTypeUInt64:
      return MakeValue(static_cast<uint64_t>(in));
    case kNumberTypeUInt32:
      return MakeValue(static_cast<uint32_t>(in));
    case kNumberTypeUInt16:
      return MakeValue(static_cast<uint16_t>(in));
    case kNumberTypeUInt8:
      return MakeValue(static_cast<uint8_t>(in));
    case kNumberTypeBFloat16:
      return MakeValue(static_cast<float16>(in).int_value());
    default:
      MS_LOG(DEBUG) << "Not support cast to dst type: " << TypeIdToType(type_id)->ToString();
      return nullptr;
  }
}

template <typename S>
ValuePtr CastScalarToTensor(S in, const TypeId &type_id) {
  switch (type_id) {
    case kNumberTypeInt32:
      return tensor::from_scalar(static_cast<int>(in), kInt32);
    case kNumberTypeFloat16:
      return tensor::from_scalar(static_cast<float16>(in), kFloat16);
    case kNumberTypeFloat32:
      return tensor::from_scalar(static_cast<float>(in), kFloat32);
    case kNumberTypeBool:
      return tensor::from_scalar(static_cast<bool>(in), kBool);
    case kNumberTypeInt64:
      return tensor::from_scalar(static_cast<int64_t>(in), kInt64);
    case kNumberTypeFloat64:
      return tensor::from_scalar(static_cast<double>(in), kFloat64);
    case kNumberTypeInt16:
      return tensor::from_scalar(static_cast<int16_t>(in), kInt16);
    case kNumberTypeInt8:
      return tensor::from_scalar(static_cast<int8_t>(in), kInt8);
    case kNumberTypeUInt64:
      return tensor::from_scalar(static_cast<uint64_t>(in), kUInt64);
    case kNumberTypeUInt32:
      return tensor::from_scalar(static_cast<uint32_t>(in), kUInt32);
    case kNumberTypeUInt16:
      return tensor::from_scalar(static_cast<uint16_t>(in), kUInt16);
    case kNumberTypeUInt8:
      return tensor::from_scalar(static_cast<uint8_t>(in), kUInt8);
    case kNumberTypeBFloat16:
      return tensor::from_scalar(static_cast<bfloat16>(in), kBFloat16);
    default:
      MS_LOG(DEBUG) << "Not support cast to dst type: " << TypeIdToType(type_id)->ToString();
      return nullptr;
  }
}

template <typename S>
ValuePtr Cast(S in, const std::pair<TypeId, bool> &dst_type) {
  bool has_tensor_input = dst_type.second;
  if (has_tensor_input) {
    return CastScalarToTensor(in, dst_type.first);
  }
  return CastScalarToScalar(in, dst_type.first);
}

ValuePtr ScalarToDstDtypeValue(const ValuePtr &src_value, const std::pair<TypeId, bool> &dst_type) {
  MS_EXCEPTION_IF_NULL(src_value);
  // Tensor not do scalar cast
  if (src_value->isa<tensor::Tensor>()) {
    return nullptr;
  }
  if (src_value->isa<Int64Imm>()) {
    const auto &int64_v = src_value->cast<Int64ImmPtr>();
    return Cast<int64_t>(int64_v->value(), dst_type);
  }
  if (src_value->isa<FP32Imm>()) {
    const auto &fp32_v = src_value->cast<FP32ImmPtr>();
    return Cast<float>(fp32_v->value(), dst_type);
  }
  if (src_value->isa<Int32Imm>()) {
    const auto &int32_v = src_value->cast<Int32ImmPtr>();
    return Cast<int32_t>(int32_v->value(), dst_type);
  }
  if (src_value->isa<FP64Imm>()) {
    const auto &fp64_v = src_value->cast<FP64ImmPtr>();
    return Cast<double>(fp64_v->value(), dst_type);
  }
  if (src_value->isa<BoolImm>()) {
    const auto &bool_v = src_value->cast<BoolImmPtr>();
    return Cast<bool>(bool_v->value(), dst_type);
  }
  if (src_value->isa<Int16Imm>()) {
    const auto &int16_v = src_value->cast<Int16ImmPtr>();
    return Cast<int16_t>(int16_v->value(), dst_type);
  }
  MS_LOG(DEBUG) << "Now, the value [" << src_value->ToString() << "] is not supported to cast directly.";
  return nullptr;
}

tensor::TensorPtr TensorToDstDtypeValue(const ValuePtr &src_value, const TypeId &dst_type_id) {
  MS_EXCEPTION_IF_NULL(src_value);
  auto src_tensor = src_value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(src_tensor);
  (void)src_tensor->set_data_type(dst_type_id);
  return src_tensor;
}

py::object SymbolicFromGuard(const GuardItemPtr &item, const py::object &new_object) {
  if (item->GetType() != GIType::GTEqual) {
    return {};
  }
  auto eq_guard = static_cast<EqGuard *>(item.get());
  py::object h = new_object;
  if (!eq_guard->specialized() && CheckTensorObject(h.ptr())) {
    tensor::TensorPtr new_tenor = MetaTensorData::GetStubInfo(h.ptr());
    h = eq_guard->MakeDynamicShape(new_tenor);
    if (h.ptr() == nullptr) {
      return {};
    }
  }
  py::object res = py::module::import("mindspore.graph.mutable").attr("mutable")(h);
  PIJIT_DEBUG_LOG(LogCfg::kDynamic) << "Symbolic object value: " << Py_TYPE(new_object.ptr())->tp_name << "("
                                    << py::repr(new_object) << ") ->" << Py_TYPE(h.ptr())->tp_name << "(" << py::repr(h)
                                    << ")";
  return res;
}

bool IsSpecializedGuard(const GuardItemPtr &item) {
  if (item->GetType() != GIType::GTEqual) {
    return false;
  }
  return static_cast<EqGuard *>(item.get())->specialized();
}

bool GuardItemPyTypeMatch(const GuardItemPtr &item, const py::handle &new_object) {
  if (item->GetType() == GIType::GTType) {
    return Py_TYPE(new_object.ptr()) == static_cast<TypeGuard *>(item.get())->ref_type();
  }
  if (item->GetType() != GIType::GTEqual) {
    return false;
  }
  EqGuard *item_p = static_cast<EqGuard *>(item.get());
  if (item_p->GetObject().ptr() != nullptr) {
    return Py_TYPE(new_object.ptr()) == Py_TYPE(item_p->GetObject().ptr());
  }
  if (item_p->Item()->GetItemType() == ItemType::PyUnknown) {
    return false;
  }
  if (item_p->Item()->GetItemType() == ItemType::Tensor) {
    return CheckTensorObject(new_object.ptr());
  }
  if (item_p->Item()->GetItemType() == ItemType::Cell) {
    return Py_TYPE(new_object.ptr()) == static_cast<CellData *>(item_p->Item().get())->GetTypeObject();
  }
  return false;
}

std::string GetItemDataStr(const GuardItemPtr &item, PyObject *obj) {
  if (dynamic_cast<EqGuard *>(item.get())) {
    return " == " + CreateItem(obj, false, false)->ToString();
  }
  return "";
}
}  // namespace pijit
}  // namespace mindspore
