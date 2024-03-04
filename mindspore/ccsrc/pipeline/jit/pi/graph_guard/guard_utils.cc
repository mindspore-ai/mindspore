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
#include "pipeline/jit/pi/graph_guard/guard_utils.h"
#include <regex>
#include "pybind11/pybind11.h"
#include "pybind_api/ir/primitive_py.h"
#include "pybind_api/ir/cell_py.h"
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/pi/utils/utils.h"
#include "include/common/utils/stub_tensor.h"
#include "pipeline/jit/pi/graph_guard/strategy.h"
#include "pipeline/jit/pi/graph_guard/guard.h"

namespace mindspore {
namespace pijit {
static std::string GetObjectString(PyObject *objName) {
  std::string ret = "";
  if (objName == NULL) {
    return ret;
  }
  PyObject *pyName = PyUnicode_AsEncodedString(objName, "utf-8", NULL);
  char *strName = PyBytes_AsString(pyName);
  if (strName != nullptr) {
    ret = strName;
  }
  Py_DECREF(pyName);
  return ret;
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

  virtual bool operator==(const ItemData &obj) const { return obj.tp_ == tp_; }

  virtual std::string ToString() {
    if (tp_ == ItemType::PyNull) {
      return "(null)";
    } else {
      return std::string("(type:") + std::to_string(static_cast<int>(tp_)) +
             ",specialize:" + std::to_string(specialized_) + ",recurse:" + std::to_string(recurseDepth_) + ")";
    }
  }

  virtual const InfoPack &Info() {
    if (info_ == nullptr) {
      InfoPack info;
      info << uint8_t(tp_);
      info.Begin();
      if (tp_ != ItemType::PyNull && tp_ != ItemType::PyUnknown) {
        info << specialized_ << recurseDepth_;
      }
      SubInfo(&info);
      info.End();
      info_ = std::make_shared<InfoPack>(info);
      info_->Update();
    }
    return *info_;
  }

  virtual ItemType GetItemType() { return tp_; }

  virtual bool MatchDynamicShape(std::shared_ptr<ItemData> other) { return false; }

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
    tp_ = ItemType::PyLong;
    intVar_ = PyLong_AsLong(obj);
  }

  bool operator==(const ItemData &obj) const override {
    return ItemData::operator==(obj) && (!specialized_ || (((const IntData &)obj).intVar_ == intVar_));
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
    return ItemData::operator==(obj) && (!specialized_ || ((const FloatData &)obj).floatVar_ == floatVar_);
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
    return ItemData::operator==(obj) && (!specialized_ || ((const BoolData &)obj).boolVar_ == boolVar_);
  }

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
        char *pBuf = PyBytes_AS_STRING(obj);
        if (pBuf != nullptr) {
          memcpy(buf_.get(), pBuf, len_);
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
      const BytesData &other = (const BytesData &)obj;
      return len_ == other.len_ &&
             ((specialized_ && (len_ == 0 || (buf_ != nullptr && other.buf_ != nullptr &&
                                              memcmp(buf_.get(), other.buf_.get(), len_) == 0))) ||
              (!specialized_));
    }
    return false;
  }

  std::string ToString() override {
    size_t bytes = (size_t)(buf_.get());
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
      strVal_ = GetObjectString(obj);
    }
  }

  bool operator==(const ItemData &obj) const override {
    return ItemData::operator==(obj) &&
           ((specialized_ && ((const StringData &)obj).strVal_.compare(strVal_) == 0) || (!specialized_));
  }

  std::string ToString() override { return DESC(strVal_) + DESC_END; }

 protected:
  void SubInfo(InfoPack *info) override { (*info) << strVal_; }
  std::string strVal_;
};

class ListData : public ItemData {
 public:
  ListData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyList, needSpecialize, recurseDepth) {
    if (PyList_Check(obj)) {
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
    } else if (PyTuple_Check(obj)) {
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
    } else if (PySet_Check(obj)) {
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
    } else if (PyFrozenSet_Check(obj)) {
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
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const ListData &list = (const ListData &)obj;
      if (list.listVar_.size() == listVar_.size()) {
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
    return ItemData::operator==(obj) && (!specialized_ || ((const ComplexData &)obj).complexVar_ == complexVar_);
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
    Py_ssize_t start = 0, stop = 0, step = 0;
    if (needSpecialize) {
      PySlice_Unpack(obj, &start, &stop, &step);
      sliceVar_.push_back((int64_t)start);
      sliceVar_.push_back((int64_t)stop);
      sliceVar_.push_back((int64_t)step);
    }
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const SliceData &other = (const SliceData &)obj;
      return (!specialized_ || (other.sliceVar_[0] == sliceVar_[0] && other.sliceVar_[1] == sliceVar_[1] &&
                                other.sliceVar_[2] == sliceVar_[2]));
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
      obj = PyObject_CallOneArg(reinterpret_cast<PyObject *>(&PyList_Type), obj);
    } else if (PyDictValues_Check(obj)) {
      dt_ = DictType::DtValues;
      obj = PyObject_CallOneArg(reinterpret_cast<PyObject *>(&PyList_Type), obj);
    } else if (PyDictItems_Check(obj)) {
      dt_ = DictType::DtItems;
      obj = PyObject_CallOneArg(reinterpret_cast<PyObject *>(&PyDict_Type), obj);
    } else {
      dt_ = DictType::DtDict;
    }
    Py_ssize_t pos = 0;
    PyObject *key, *val;
    if (dt_ == DictType::DtItems || dt_ == DictType::DtDict) {
      while (PyDict_Next(obj, &pos, &key, &val)) {
        ItemDataPtr k, v;
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
      for (Py_ssize_t i = 0; i < PyList_Size(obj); ++i) {
        PyObject *item = PyList_GetItem(obj, i);
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
      const DictData &other = (const DictData &)obj;
      if (dt_ != other.dt_) {
        return false;
      }
      if ((dt_ == DictType::DtValues || other.listK_.size() == listK_.size()) &&
          (dt_ == DictType::DtKeys || other.listV_.size() == listV_.size())) {
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
      const FunctionData &other = (const FunctionData &)obj;
      return code_ == other.code_ && *defaults_ == *(other.defaults_) && *kwdefaults_ == *(other.kwdefaults_) &&
             *closure_ == *(other.closure_);
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
      const MethodData &other = (const MethodData &)obj;
      return *refFunc_ == *(other.refFunc_) && *refSelf_ == *(other.refSelf_);
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
      const InstanceMethodData &other = (const InstanceMethodData &)obj;
      return *refFunc_ == *(other.refFunc_);
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
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      PyTypeObject *otherType = ((const TypeData &)obj).refType_;
      bool ret = refType_ == otherType;
      if (!ret) {
        ret = PyType_IsSubtype(refType_, otherType) || PyType_IsSubtype(otherType, refType_);
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
          memcpy(buf_.get(), arr.data(), nbytes_);
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
      const NumpyData &other = (const NumpyData &)obj;
      return dtype_ == other.dtype_ && size_ == other.size_ && ndim_ == other.ndim_ && nbytes_ == other.nbytes_ &&
             shape_ == other.shape_ && strides_ == other.strides_ &&
             (!specialized_ ||
              (buf_ != NULL && other.buf_ != NULL && memcmp(buf_.get(), other.buf_.get(), nbytes_) == 0));
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
    return DESC(numpy) + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override {
    (*info) << dtype_.kind() << size_ << itemsize_ << ndim_ << nbytes_ << shape_ << strides_;
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
    return ItemData::operator==(obj) && (!specialized_ || *(((const TensorTypeData &)obj).tpp_) == *tpp_);
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
      const ParamInfoData &other = (const ParamInfoData &)obj;
      return Equal(param_, other.param_);
    }
    return false;
  }

  static bool Equal(ParamInfoPtr a, ParamInfoPtr b) {
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

  static void SubInfo(InfoPack *info, mindspore::ParamInfoPtr p) {
    if (p == nullptr) {
      return;
    }
    (*info) << p->name() << p->requires_grad() << p->comm_fusion() << p->parallel_optimizer() << p->requires_aggr()
            << p->parallel_optimizer_comm_recompute() << p->use_persistent_storage() << p->cache_enable()
            << p->parameter_shape() << p->cache_shape() << p->param_strategy();
  }

 protected:
  void SubInfo(InfoPack *info) override { SubInfo(info, param_); }
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
    auto pyObj = py::cast<py::object>(obj);
    constexpr char const_arg_attr[] = "const_arg";
    if (!py::hasattr(pyObj, const_arg_attr) || !py::cast<bool>(py::getattr(obj, const_arg_attr))) {
      specialized_ = needSpecialize;
    } else {
      specialized_ = true;
    }
    mindspore::tensor::MetaTensorPtr tensor_ptr = nullptr;
    if (py::isinstance<mindspore::tensor::MapTensor>(obj)) {
      tensor_ptr = pyObj.cast<mindspore::tensor::MapTensorPtr>();
    } else if (py::isinstance<mindspore::tensor::Tensor>(obj)) {
      tensor_ptr = pyObj.cast<mindspore::tensor::TensorPtr>();
    } else if (IsStubTensor(pyObj)) {
      auto stub = PyObject_GetAttrString(obj, "stub");
      if (stub != nullptr && stub != Py_None) {
        is_stubtensor_ = true;
        Py_DECREF(stub);
      } else {
        obj = PyObject_GetAttrString(obj, "tensor");
        pyObj = py::cast<py::object>(obj);
        Py_DECREF(obj);
        tensor_ptr = pyObj.cast<mindspore::tensor::TensorPtr>();
      }
    } else {
      tensor_ptr = pyObj.cast<mindspore::tensor::MetaTensorPtr>();
    }
    if (tensor_ptr != nullptr) {
      StoreTensor(tensor_ptr);
    } else {
      obj = PyObject_GetAttrString(obj, "stub");
      pyObj = py::cast<py::object>(obj);
      auto ptr = pyObj.cast<mindspore::stub::StubNodePtr>();
      StoreStubTensor(ptr);
      Py_DECREF(obj);
    }
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const MetaTensorData &other = (const MetaTensorData &)obj;
      bool ret;
      if (is_stubtensor_ || other.is_stubtensor_) {
        ret = CheckShape(shape_, other.shape_) && CheckDataType(other);
      } else {
        ret = tid_ == other.tid_ && CheckShape(shape_, other.shape_) && format_.compare(other.format_) == 0 &&
              host_format_.compare(other.host_format_) == 0 && is_parameter_ == other.is_parameter_ &&
              CheckDataType(other);
      }
      if (ret) {
        if (is_parameter_ == true) {
          ret = ((param_ == nullptr && other.param_ == nullptr) ||
                 (param_ != nullptr && other.param_ != nullptr && ParamInfoData::Equal(param_, other.param_)));
        }
      }
      return ret;
    }
    return false;
  }

  mindspore::tensor::TensorPtr MakeTensor() {
    return std::make_shared<mindspore::tensor::Tensor>(data_type_->type_id(), shape_);
  }

  bool IsDynamicShape() const {
    return std::any_of(shape_.begin(), shape_.end(),
                       [](ShapeValueDType dim) { return dim == kDynamicDim || dim == kDynamicShape; });
  }

  std::string ToString() override {
    std::string meta_tensor = ToStringIntern();
    return DESC(meta_tensor) + DESC_END;
  }

  bool MatchDynamicShape(std::shared_ptr<ItemData> other) override {
    auto type = other->GetItemType();
    if (type != ItemType::Tensor && type != ItemType::MetaTensor) {
      return false;
    }
    auto o = static_cast<MetaTensorData *>(other.get());
    if (!CheckDataType(*o) || specialized_ != false || o->specialized_ != false) {
      return false;
    }
    if (shape_.size() != o->shape_.size()) {
      shape_ = {kDynamicDim};
    } else {
      for (size_t idx = 0; idx < shape_.size(); ++idx) {
        if (shape_[idx] != kDynamicShape && shape_[idx] != o->shape_[idx]) {
          shape_[idx] = kDynamicShape;
        }
      }
    }
    return true;
  }

 protected:
  virtual std::string ToStringIntern() {
    std::string param_desc = ParamInfoData::ToStringAttr(param_);
    std::string shape = "";
    for (size_t i = 0; i < shape_.size(); ++i) {
      shape += DESC_INDEX_V(shape_, i);
    }
    std::string is_stubtensor = is_stubtensor_ ? "true" : "false";
    return DESC_STRING(tid_) + DESC(format_) + DESC(host_format_) + DESC_TOSTRING(data_type_) +
           DESC_STRING(is_parameter_) + DESC(param_desc) + DESC(shape) + DESC(is_stubtensor);
  }

  bool CheckDataType(const MetaTensorData &other) const {
    return (data_type_ == nullptr && other.data_type_ == nullptr) ||
           (data_type_ != nullptr && other.data_type_ != nullptr && *data_type_ == *(other.data_type_));
  }

  void StoreTensor(mindspore::tensor::MetaTensorPtr tensor_ptr) {
    tid_ = tensor_ptr->data_type();
    shape_ = tensor_ptr->shape();
    mindspore::tensor::DeviceInfo info = tensor_ptr->device_info();
    format_ = info.format_;
    host_format_ = info.host_format_;
    data_type_ = tensor_ptr->Dtype();
    is_parameter_ = tensor_ptr->is_parameter();
    param_ = tensor_ptr->param_info() != nullptr ? tensor_ptr->param_info()->Clone() : nullptr;
  }

  void StoreStubTensor(mindspore::stub::StubNodePtr stub_ptr) {
    auto base = stub_ptr->ToAbstract();
    auto shape = base->BuildShape()->cast<abstract::ShapePtr>();
    if (shape && !shape->IsDynamic()) {
      shape_ = shape->shape();
    } else {
      shape_ = {};
    }
    auto dt = base->BuildType();
    if (dt->isa<mindspore::TensorType>()) {
      data_type_ = dt->cast<std::shared_ptr<mindspore::TensorType>>()->element();
    } else {
      data_type_ = dt;
    }
  }

  void SubInfo(InfoPack *info) override {
    (*info) << uint8_t(tid_) << format_ << host_format_ << data_type_ << is_parameter_ << shape_ << is_stubtensor_;
    ParamInfoData::SubInfo(info, param_);
  }

  mindspore::TypeId tid_;
  ShapeVector shape_;
  std::string format_;
  std::string host_format_;
  TypePtr data_type_;
  bool is_parameter_;
  bool is_stubtensor_ = false;
  mindspore::ParamInfoPtr param_;
};

class TensorData : public MetaTensorData {
 public:
  TensorData(mindspore::tensor::TensorPtr tensor_ptr, bool needSpecialize, int recurseDepth)
      : MetaTensorData(tensor_ptr, needSpecialize, recurseDepth) {
    tp_ = ItemType::Tensor;
    StoreTensor(tensor_ptr);
  }

  TensorData(PyObject *obj, bool needSpecialize, int recurseDepth) : MetaTensorData(obj, needSpecialize, recurseDepth) {
    is_stubtensor_ = false;
    tp_ = ItemType::Tensor;
    auto pyObj = py::cast<py::object>(obj);
    mindspore::tensor::TensorPtr tensor_ptr = nullptr;
    if (py::isinstance<mindspore::tensor::MapTensor>(obj)) {
      tensor_ptr = pyObj.cast<mindspore::tensor::MapTensorPtr>();
    } else if (IsStubTensor(pyObj)) {
      auto stub = PyObject_GetAttrString(obj, "stub");
      if (stub != nullptr && stub != Py_None) {
        if (OptStrategy::MakeCalcStrategyByShape(GetStubTensorInfo(py::cast<py::object>(obj)).first) !=
            OptStrategy::CalcKind::kCalcValue) {
          specialized_ = false;
        }
      }
      if (specialized_) {
        pyObj = python_adapter::CallPyObjMethod(pyObj, "stub_sync");
        tensor_ptr = pyObj.cast<mindspore::tensor::TensorPtr>();
      } else {
        if (stub != nullptr && stub != Py_None) {
          is_stubtensor_ = true;
        } else {
          obj = PyObject_GetAttrString(obj, "tensor");
          pyObj = py::cast<py::object>(obj);
          Py_DECREF(obj);
          tensor_ptr = pyObj.cast<mindspore::tensor::TensorPtr>();
        }
      }
      if (stub != nullptr && stub != Py_None) {
        Py_DECREF(stub);
      }
    } else {
      tensor_ptr = pyObj.cast<mindspore::tensor::TensorPtr>();
    }
    if (tensor_ptr != nullptr) {
      if (OptStrategy::MakeCalcStrategyByShape(tensor_ptr->shape()) != OptStrategy::CalcKind::kCalcValue) {
        specialized_ = false;
      }
      StoreTensor(tensor_ptr);
    } else {
      obj = PyObject_GetAttrString(obj, "stub");
      pyObj = py::cast<py::object>(obj);
      auto ptr = pyObj.cast<mindspore::stub::StubNodePtr>();
      StoreStubTensor(ptr);
      Py_DECREF(obj);
    }
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
    const TensorData &other = (const TensorData &)obj;
    if (is_stubtensor_ || other.is_stubtensor_) {
      return ret;
    }
    ret = ret && other.init_flag_ == init_flag_ && other.is_forward_output_ == is_forward_output_ &&
          /*other.id_.compare(id_) == 0 &&*/ other.graph_output_ == graph_output_ &&
          other.specialized_ == specialized_ && IsBaseShapePtr(other) && IsCastDtype(other) &&
          other.compression_type_ == compression_type_ && other.quant_params_.size() == quant_params_.size() &&
          other.tensor_name_.compare(tensor_name_) == 0;
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

  std::string ToString() override {
    std::string tensor = ToStringIntern();
    return DESC(tensor) + DESC_END;
  }

 protected:
  std::string ToStringIntern() override {
    std::string ret = MetaTensorData::ToStringIntern();
    ret += DESC_STRING(is_forward_output_) + DESC_STRING(init_flag_) + DESC_STRING(graph_output_);
    ret +=
      DESC_TOSTRING(cast_dtype_) + DESC_TOSTRING(base_shape_ptr_) + DESC_STRING(compression_type_) + DESC(tensor_name_);
    for (size_t i = 0; i < quant_params_.size(); ++i) {
      ret += DESC_INDEX(quant_params_, i);
    }
    return ret;
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
    MetaTensorData::StoreTensor(tensor_ptr);
    init_flag_ = tensor_ptr->is_init();
    is_forward_output_ = tensor_ptr->is_forward_output();
    id_ = tensor_ptr->id();
    graph_output_ = tensor_ptr->IsGraphOutput();
    base_shape_ptr_ = tensor_ptr->base_shape_ptr() == nullptr ? nullptr : tensor_ptr->base_shape_ptr()->Clone();
    cast_dtype_ = (tensor_ptr->cast_dtype() == nullptr) ? nullptr : tensor_ptr->cast_dtype()->Clone();
    compression_type_ = tensor_ptr->compression_type();
    const std::vector<std::shared_ptr<mindspore::QuantizationParam>> &qp = tensor_ptr->quant_params();
    tensor_name_ = tensor_ptr->name();
    for (auto quant : qp) {
      QuantizationParamPtr qptr = std::make_shared<mindspore::QuantizationParam>(quant->quant_algo_name());
      quant_params_.push_back(qptr);
      qptr->set_attrs(quant->attrs());
    }
    if (specialized_) {
      tensor_ptr->data_sync(true);
      auto data = tensor_ptr->data_ptr();
      data_len_ = data->nbytes();
      data_ptr_ = std::make_unique<uint8_t[]>(data_len_);
      if (data_ptr_ != nullptr) {
        memcpy(data_ptr_.get(), data->data(), data_len_);
      }
    } else {
      data_ptr_.reset(nullptr);
      data_len_ = tensor_ptr->data_ptr()->nbytes();
    }
  }

  void SubInfo(InfoPack *info) override {
    MetaTensorData::SubInfo(info);
    (*info) << is_forward_output_ << init_flag_ << graph_output_ << cast_dtype_ << base_shape_ptr_
            << uint8_t(compression_type_) << tensor_name_;
    (*info) << uint64_t(quant_params_.size());
    for (auto qp : quant_params_) {
      (*info) << qp;
    }
  }

  bool init_flag_;
  bool is_forward_output_;
  std::unique_ptr<uint8_t[]> data_ptr_;
  size_t data_len_;
  std::string id_;
  bool graph_output_;
  // bool updated_by_device_{false};
  // DeviceSyncPtr device_sync_{nullptr};
  // bool need_release_device_mem_{false};
  // bool cache_enable_{false};
  mindspore::abstract::BaseShapePtr base_shape_ptr_;
  // std::shared_ptr<Tensor> cache_tensor_ptr_{nullptr};
  // std::shared_ptr<Tensor> hashmap_tensor_ptr_{nullptr};
  mindspore::TypePtr cast_dtype_;
  // std::shared_ptr<DeviceEvent> device_event_{nullptr};
  // UserData user_data_;
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
    return (other.default_value_ == nullptr && default_value_ == nullptr) ||
           (other.default_value_ != nullptr && default_value_ != nullptr && *default_value_ == *(other.default_value_));
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
    const MapTensorData &other = (const MapTensorData &)obj;
    bool ret = TensorData::operator==(obj);
    return ret && other.key_dtype_ == key_dtype_ && other.key_shape_ == key_shape_ && IsDefaultValue(other) &&
           IsPermitFilterValue(other) && IsEvictFilterValue(other) && value_shape_ == other.value_shape_ &&
           Equal(key_tensor_, other.key_tensor_, recurseDepth_) &&
           Equal(value_tensor_, other.value_tensor_, recurseDepth_) &&
           Equal(status_tensor_, other.status_tensor_, recurseDepth_);
  }

  std::string ToString() override {
    std::string map_tensor = ToStringIntern();
    return DESC(map_tensor) + DESC_END;
  }

 protected:
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
      const RowTensorData &other = (const RowTensorData &)obj;
      return other.data_type_ == data_type_ && other.shape_ == shape_ &&
             Equal(indices_, other.indices_, recurseDepth_) && Equal(values_, other.values_, recurseDepth_);
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
      const COOTensorData &other = (const COOTensorData &)obj;
      return other.data_type_ == data_type_ && other.shape_ == shape_ &&
             Equal(indices_, other.indices_, recurseDepth_) && Equal(values_, other.values_, recurseDepth_);
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
      const CSRTensorData &other = (const CSRTensorData &)obj;
      return other.data_type_ == data_type_ && other.shape_ == shape_ &&
             Equal(indices_, other.indices_, recurseDepth_) && Equal(values_, other.values_, recurseDepth_) &&
             Equal(indptr_, other.indptr_, recurseDepth_);
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
        memcpy(data_ptr_.get(), data->data(), nbytes_);
      }
    } else {
      data_ptr_.reset(nullptr);
    }
  }

  ~TensorDataData() override { data_ptr_.release(); }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const TensorDataData &other = (const TensorDataData &)obj;
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

  std::string ToString() override {
    std::string tensor_data = DESC_STRING(size_) + DESC_STRING(itemsize_) + DESC_STRING(nbytes_) + DESC_STRING(ndim_);
    return DESC(tensor_data) + DESC_END;
  }

 protected:
  void SubInfo(InfoPack *info) override { (*info) << size_ << itemsize_ << nbytes_ << ndim_; }
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
    PyObject *key, *val;
    while (PyDict_Next(dct, &pos, &key, &val)) {
      ItemDataPtr k, v;
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
      const PrimitiveData &other = (const PrimitiveData &)obj;
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
      : ItemData(ItemType::Cell, needSpecialize, recurseDepth) {
    auto pyObj = py::cast<py::object>(obj);
    auto cell = pyObj.cast<mindspore::CellPtr>();
    PyObject *ns = PyObject_GetAttrString(obj, "__dict__");
    if (!ns) {
      return;
    }
    PyObject *items = PyMapping_Items(ns);
    if (!items) {
      return;
    }
    for (Py_ssize_t pos = 0; pos < PyList_GET_SIZE(items); pos++) {
      PyObject *it = PySequence_Fast(PyList_GET_ITEM(items, pos), "items() returned non-iterable");
      if (!it || PySequence_Fast_GET_SIZE(it) != 2) {
        if (it) {
          Py_DECREF(it);
        }
        continue;
      }
      PyObject *key = PySequence_Fast_GET_ITEM(it, 0);
      PyObject *val = PySequence_Fast_GET_ITEM(it, 1);
      ItemDataPtr k, v;
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
      Py_DECREF(it);
    }
    Py_DECREF(items);
    Py_DECREF(ns);
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      const CellData &other = (const CellData &)obj;
      for (size_t i = 0; i < listK_.size(); ++i) {
        if (*(listK_[i]) == *(other.listK_[i]) && *(listV_[i]) == *(other.listV_[i])) {
          continue;
        } else {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  std::string ToString() override {
    std::string cell;
    for (size_t i = 0; i < listK_.size(); ++i) {
      cell += DESC_ITEM(listK_[i], listV_[i]);
    }
    return DESC(cell) + DESC_END;
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

class UnknownData : public ItemData {
 public:
  UnknownData(PyObject *obj, bool needSpecialize, int recurseDepth)
      : ItemData(ItemType::PyUnknown, needSpecialize, recurseDepth) {
    refId_ = obj;
  }

  bool operator==(const ItemData &obj) const override {
    if (ItemData::operator==(obj)) {
      return refId_ == ((const UnknownData &)obj).refId_;
    }
    return false;
  }

  std::string ToString() override {
    std::string ret = "unknown";
    return ret + ItemData::ToString();
  }

 protected:
  PyObject *refId_;
};

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
static bool CheckTensorObject(PyObject *obj) {
  return py::isinstance<mindspore::tensor::Tensor>(obj) || IsStubTensor(py::cast<py::object>(obj));
}
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
  {[](PyObject *obj) -> bool { return CheckTensorObject(obj); }, CreatePyData<TensorData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::ParamInfo>(obj); }, CreatePyData<ParamInfoData>},
  {[](PyObject *obj) -> bool { return py::isinstance<mindspore::tensor::MetaTensor>(obj); },
   CreatePyData<MetaTensorData>},
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
      PyObject *py_type;
      py::object py_obj = py::reinterpret_borrow<py::object>(obj);
      if (IsStubTensor(py_obj)) {
        py_type = GetMsTensorType();
      } else {
        py_type = reinterpret_cast<PyObject *>(Py_TYPE(obj));
      }
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

GuardItem::GuardItem(TracePtr tt) : var_(tt), type_(GIType::GTUnknown), info_(nullptr) {}

void GuardItem::Replace(TracePtr dst, TracePtr src) {
  if (!var_) {
    return;
  }
  if (*var_ == *src) {
    var_ = dst;
  } else {
    var_->Replace(dst, src);
  }
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

TracePtr GuardItem::GetTrace() { return var_; }

bool GuardItem::operator==(const GuardItem &obj) const { return type_ == obj.type_ && *var_ == *(obj.var_); }

static constexpr int kGuardItemTotalStage = 2;
static constexpr int kGuardItemRetrieveStage = 0;
static constexpr int kGuardItemCompareStage = 1;

static void GuardItemPerfStart(bool enable, int total) {
  if (enable) {
    OptGuardPerf::GetGuardPerf()->LogItemPerfStart(total);
  }
}

static void GuardItemPerfStage(bool enable, GuardItem *item, int stage) {
  if (enable) {
    OptGuardPerf::GetGuardPerf()->LogItemPerfEnd(item, stage);
  }
}

class EqGuard : public GuardItem {
 public:
  EqGuard(TracePtr obj, bool needSpecialize, int recurseDepth)
      : GuardItem(obj),
        dp_(CreateItem(obj->GetObject(), needSpecialize, recurseDepth)),
        specialized_(needSpecialize),
        recurse_(recurseDepth) {
    type_ = GIType::GTEqual;
  }

  virtual bool Check(const PyFrameObject *frame, std::map<size_t, PyObject *> *cache, bool perf) {
    if (var_->IsConst()) {
      return true;
    }
    GuardItemPerfStart(perf, kGuardItemTotalStage);
    PyObject *obj = GetObjectFromTrace(frame, var_, cache, perf);
    GuardItemPerfStage(perf, this, kGuardItemRetrieveStage);
    bool ret = Check(obj);
    GuardItemPerfStage(perf, this, kGuardItemCompareStage);
    if (obj != NULL) {
      Py_DECREF(obj);
    }
    return ret;
  }

  virtual bool Check(PyObject *obj) {
    ItemDataPtr other = CreateItem(obj, specialized_, recurse_);
    return *dp_ == *other;
  }

  virtual std::string ToString() {
    if (strGuard_.size() > 0) {
      return strGuard_;
    }
    strGuard_ = var_->ToString() + "==" + dp_->ToString();
    strGuard_ = std::regex_replace(strGuard_, std::regex("(\n)"), "");
    return strGuard_;
  }

  virtual const InfoPack &Info() {
    if (info_ == nullptr) {
      InfoPack info;
      info << uint8_t(type_);
      info.Begin();
      info << var_->Info() << dp_->Info();
      info.End();
      info_ = std::make_shared<InfoPack>(info);
      info_->Update();
    }
    return *info_;
  }

  bool operator==(const GuardItem &obj) const override {
    if (GuardItem::operator==(obj)) {
      auto other = (const EqGuard &)obj;
      return specialized_ == other.specialized_ && recurse_ == other.recurse_ && *dp_ == *(other.dp_);
    }
    return false;
  }

  bool MatchDynamicShape(std::shared_ptr<GuardItem> other) override {
    var_->Detach();
    other->GetTrace()->Detach();
    if (other->GetType() != GIType::GTEqual || !(*var_ == *(other->GetTrace())) ||
        !dp_->MatchDynamicShape((static_cast<EqGuard *>(other.get()))->dp_)) {
      return false;
    } else {
      return true;
    }
  }

  PyObject *ApplyDynamicShape(PyObject *obj) override {
    auto type = dp_->GetItemType();
    if (type != ItemType::MetaTensor && type != ItemType::Tensor) {
      return nullptr;
    }
    auto item = (MetaTensorData &)(*dp_);
    if (item.IsDynamicShape()) {
      return py::cast(item.MakeTensor()).inc_ref().ptr();
    } else {
      return nullptr;
    }
  }

 protected:
  ItemDataPtr dp_;
  bool specialized_;
  int recurse_;
};

class TypeGuard : public GuardItem {
 public:
  explicit TypeGuard(TracePtr obj) : GuardItem(obj) {
    type_ = GIType::GTType;
    if (obj->GetTraceType() == TraceType::Type) {
      refType_ = std::dynamic_pointer_cast<TypeTrace>(obj)->GetType();
    } else {
      refType_ = Py_TYPE(obj->GetObject());
    }
  }

  virtual bool Check(const PyFrameObject *frame, std::map<size_t, PyObject *> *cache, bool perf) {
    if (var_->IsConst()) {
      return true;
    }
    GuardItemPerfStart(perf, kGuardItemTotalStage);
    PyObject *obj = GetObjectFromTrace(frame, var_, cache, perf);
    GuardItemPerfStage(perf, this, kGuardItemRetrieveStage);
    bool ret = Check(obj);
    GuardItemPerfStage(perf, this, kGuardItemCompareStage);
    if (var_->GetTraceType() != TraceType::Type && obj != NULL) {
      Py_DECREF(obj);
    }
    return ret;
  }

  virtual bool Check(PyObject *obj) {
    if (obj == NULL) {
      return false;
    }
    PyTypeObject *tp;
    if (var_->GetTraceType() == TraceType::Type) {
      tp = reinterpret_cast<PyTypeObject *>(obj);
    } else {
      tp = Py_TYPE(obj);
    }
    if (tp != refType_) {
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
      return refType_ == ((const TypeGuard &)obj).refType_;
    }
    return false;
  }

 protected:
  PyTypeObject *refType_;
};

class IdGuard : public GuardItem {
 public:
  explicit IdGuard(TracePtr obj) : GuardItem(obj) {
    type_ = GIType::GTId;
    refId_ = obj->GetObject();
  }

  virtual bool Check(const PyFrameObject *frame, std::map<size_t, PyObject *> *cache, bool perf) {
    if (var_->IsConst()) {
      return true;
    }
    GuardItemPerfStart(perf, kGuardItemTotalStage);
    PyObject *obj = GetObjectFromTrace(frame, var_, cache, perf);
    GuardItemPerfStage(perf, this, kGuardItemRetrieveStage);
    bool ret = Check(obj);
    GuardItemPerfStage(perf, this, kGuardItemCompareStage);
    if (obj != NULL) {
      Py_DECREF(obj);
    }
    return ret;
  }

  virtual bool Check(PyObject *obj) {
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
    strGuard_ = std::string("id(") + var_->ToString() + std::string(")==") + std::to_string((size_t)refId_);
    strGuard_ = std::regex_replace(strGuard_, std::regex("(\n)"), "");
    return strGuard_;
  }

  virtual const InfoPack &Info() {
    if (info_ == nullptr) {
      InfoPack info;
      info << uint8_t(type_);
      info.Begin();
      info << var_->Info() << reinterpret_cast<void *>(refId_);
      info.End();
      info_ = std::make_shared<InfoPack>(info);
      info_->Update();
    }
    return *info_;
  }

  bool operator==(const GuardItem &obj) const override {
    if (GuardItem::operator==(obj)) {
      return refId_ == ((const IdGuard &)obj).refId_;
    }
    return false;
  }

 protected:
  PyObject *refId_;
};

class ReprGuard : public GuardItem {
 public:
  explicit ReprGuard(TracePtr obj) : GuardItem(obj) {
    type_ = GIType::GTRepr;
    refRepr_ = PyObject_Repr(obj->GetObject());
  }

  virtual ~ReprGuard() { Py_XDECREF(refRepr_); }

  virtual bool Check(const PyFrameObject *frame, std::map<size_t, PyObject *> *cache, bool perf) {
    if (var_->IsConst()) {
      return true;
    }
    GuardItemPerfStart(perf, kGuardItemTotalStage);
    PyObject *obj = GetObjectFromTrace(frame, var_, cache, perf);
    GuardItemPerfStage(perf, this, kGuardItemRetrieveStage);
    bool ret = Check(obj);
    GuardItemPerfStage(perf, this, kGuardItemCompareStage);
    if (obj != nullptr) {
      Py_DECREF(obj);
    }
    return ret;
  }

  virtual bool Check(PyObject *obj) {
    bool ret = false;
    if (obj == nullptr) {
      return ret;
    }

    auto repr = PyObject_Repr(obj);
    if (PyUnicode_Compare(repr, refRepr_)) {
      ret = false;
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
      return refRepr_ == ((const ReprGuard &)obj).refRepr_;
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

class AttrGuard : public GuardItem {
 public:
  explicit AttrGuard(TracePtr pObj) : GuardItem(pObj) {
    type_ = GIType::GTAttr;
    AttrTracePtr t = std::dynamic_pointer_cast<AttrTrace>(pObj);
    PyObject *obj = t->GetOrigin()->GetObject();
    nameAttr_ = t->GetAttribute();
    if (PyObject_HasAttrString(obj, nameAttr_.c_str()) != 0) {
      hasAttr_ = true;
    } else {
      hasAttr_ = false;
      bool is_dict = PyDict_CheckExact(obj);
      PyObject *itemName = PyUnicode_FromString(nameAttr_.c_str());
      PyObject *attr = NULL;
      if (is_dict) {
        attr = PyDict_GetItem(obj, itemName);
        if (attr != NULL) {
          Py_INCREF(attr);
        }
      } else if (PyMapping_Check(obj) || PySequence_Check(obj)) {
        attr = PyObject_GetItem(obj, itemName);
      }
      hasAttr_ = attr != NULL;
      Py_DECREF(itemName);
      if (attr != NULL) {
        Py_DECREF(attr);
      }
    }
  }

  ~AttrGuard() = default;

  virtual bool Check(const PyFrameObject *frame, std::map<size_t, PyObject *> *cache, bool perf) {
    if (var_->IsConst()) {
      return true;
    }
    GuardItemPerfStart(perf, kGuardItemTotalStage);
    PyObject *obj = GetObjectFromTrace(frame, var_, cache, perf);
    GuardItemPerfStage(perf, this, kGuardItemRetrieveStage);
    bool ret = CheckIntern(obj);
    GuardItemPerfStage(perf, this, kGuardItemCompareStage);
    if (obj != NULL) {
      Py_DECREF(obj);
    }
    return ret;
  }

  virtual bool Check(PyObject *obj) {
    bool ret;
    if (PyObject_HasAttrString(obj, nameAttr_.c_str()) != 0) {
      ret = hasAttr_;
    } else {
      bool is_dict = PyDict_CheckExact(obj);
      PyObject *itemName = PyUnicode_FromString(nameAttr_.c_str());
      PyObject *attr = NULL;
      if (is_dict) {
        attr = PyDict_GetItem(obj, itemName);
        if (attr != NULL) {
          Py_INCREF(attr);
        }
      } else if (PyMapping_Check(obj) || PySequence_Check(obj)) {
        attr = PyObject_GetItem(obj, itemName);
      }
      ret = CheckIntern(attr);
      Py_DECREF(itemName);
      if (attr != NULL) {
        Py_DECREF(attr);
      }
    }
    return ret;
  }

  virtual bool CheckIntern(PyObject *obj) {
    bool ret;
    if ((obj == NULL && !hasAttr_) || (obj != NULL && hasAttr_)) {
      ret = true;
    } else {
      ret = false;
    }
    return ret;
  }

  virtual std::string ToString() {
    if (strGuard_.size() > 0) {
      return strGuard_;
    }
    strGuard_ = std::string("exist(") + var_->ToString() + std::string(".") + nameAttr_ +
                "==" + std::to_string(hasAttr_) + std::string(")");
    strGuard_ = std::regex_replace(strGuard_, std::regex("(\n)"), "");
    return strGuard_;
  }

  bool operator==(const GuardItem &obj) const override {
    if (GuardItem::operator==(obj)) {
      return hasAttr_ == ((const AttrGuard &)obj).hasAttr_ && nameAttr_ == ((const AttrGuard &)obj).nameAttr_;
    }
    return false;
  }

 protected:
  virtual const InfoPack &Info() {
    if (info_ == nullptr) {
      InfoPack info;
      info << uint8_t(type_);
      info.Begin();
      info << var_->Info() << nameAttr_ << hasAttr_;
      info.End();
      info_ = std::make_shared<InfoPack>(info);
      info_->Update();
    }
    return *info_;
  }
  bool hasAttr_;
  std::string nameAttr_;
};

GuardItemPtr GuardEqual(TracePtr obj, bool needSpecialize, int recurseDepth) {
  return std::make_shared<EqGuard>(obj, needSpecialize, recurseDepth);
}

GuardItemPtr GuardType(TracePtr obj) { return std::make_shared<TypeGuard>(obj); }

GuardItemPtr GuardId(TracePtr obj) {
  auto py_obj = obj->GetObject();
  auto pyObj = py::cast<py::object>(obj->GetObject());
  if (IsStubTensor(pyObj) || py::isinstance<mindspore::tensor::Tensor>(py_obj)) {
    return GuardEqual(obj, false, INT_MAX);
  } else {
    return std::make_shared<IdGuard>(obj);
  }
}

GuardItemPtr GuardRepr(TracePtr obj) { return std::make_shared<ReprGuard>(obj); }

GuardItemPtr GuardAttr(TracePtr obj) {
  if (obj->GetTraceType() != TraceType::Attr) {
    return nullptr;
  } else {
    return std::make_shared<AttrGuard>(obj);
  }
}

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

}  // namespace pijit
}  // namespace mindspore
