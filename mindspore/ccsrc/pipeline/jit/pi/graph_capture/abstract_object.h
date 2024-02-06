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
#ifndef MINDSPORE_PI_JIT_GRAPH_CAPTURE_ABSTRACT_OBJECT_H
#define MINDSPORE_PI_JIT_GRAPH_CAPTURE_ABSTRACT_OBJECT_H

#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "pybind11/pybind11.h"
#include "pipeline/jit/pi/utils/mempool.h"

namespace py = pybind11;
namespace mindspore {
namespace pijit {

class AbstractObjectBase;
using AObject = AbstractObjectBase;

class AbstractObjectBase {
 public:
  enum Type {
#define ABSTRACT_TYPE_DEF(unit) kType##unit,
#include "abstract_type_kind.def"
#undef ABSTRACT_TYPE_DEF
  };
  enum MindsporeFlag {
#define ABSTRACT_MS_FLAG_DEF(unit, bit) kMsFlag##unit = 1 << (bit),
#include "abstract_ms_flag.def"
#undef ABSTRACT_MS_FLAG_DEF
  };
  static_assert(static_cast<int>(kTypeSlice) + 8 == static_cast<int>(kTypeType));  // builtin type
  static_assert(static_cast<int>(kTypeAnyValue) == 0);

  enum BoolCache {
    kBoolFalse = 0,
    kBoolTrue,
    kBoolUnknown,
  };

  // record PyObject and check self reference for list,tuple,dict
  using RecMap = std::unordered_map<PyObject *, AObject *>;

  static MemPool<AbstractObjectBase> aobject_mem_pool_;

  explicit AbstractObjectBase(Type type) : type_object_(nullptr), type_(type), ms_flag_(0) {}
  virtual ~AbstractObjectBase() {}

  void SetTypeObject(PyTypeObject *tp) { type_object_ = tp; }
  PyTypeObject *GetTypeObject() const { return type_object_; }
  Type GetType() const { return type_; }

  virtual py::object GetPyObject() { return py::object(); }

  virtual AObject *Binary(AObject *other, int op) { return MakeAObject(kTypeAnyValue); }
  virtual AObject *Unary(int op) const { return MakeAObject(kTypeAnyValue); }
  virtual AObject *GetIter() const { return MakeAObject(kTypeAnyValue); }

  virtual AObject *GetAttr(const std::string &name);
  virtual AObject *GetItem(AObject *key) { return MakeAObject(kTypeAnyValue); }

  // return false if has an python exception
  virtual bool SetAttr(const std::string &name, AObject *value) { return true; }
  virtual bool SetItem(AObject *key, AObject *value) { return true; }
  virtual bool DelItem(AObject *key) { return SetItem(key, nullptr); }
  virtual bool DelAttr(const std::string &name) { return SetAttr(name, nullptr); }
  virtual bool IsMindSporeSupportedType();
  virtual std::string ToString() const;

  void SetMsFlag(MindsporeFlag flag) { ms_flag_ |= flag; }
  void ClearMsFlag(int flag) { ms_flag_ &= ~flag; }
  bool HasMsFlag(MindsporeFlag flag) { return ms_flag_ & flag; }
  bool TestMsFlag(int flag) { return ms_flag_ & flag; }

  static Type GetPyType(PyObject *op);
  static Type GetPyType(PyTypeObject *tp);
  static Type GetMsType(PyTypeObject *tp);
  static AObject *Convert(const py::object &o) { return Convert(o.ptr()); }
  static AObject *Convert(PyObject *o) { return MakeAObject(GetPyType(o), o ? Py_TYPE(o) : nullptr, o); }
  static AObject *MakeAObject(Type real_type) { return MakeAObject(real_type, nullptr, nullptr); }

  static AObject *MakeFunction(const std::vector<AObject *> &args, const py::object &globals, int oparg);

  /**
   * BUILD_SLICE,BUILD_STRING,BUILD_SET,BUILD_LIST,BUILD_TUPLE,BUILD_CONST_KEY_MAP,BUILD_MAP
   * \return a new AbstractObject if success, else a empty AbstractObject
   **/
  static AObject *BuildOperations(const std::vector<AObject *> &args, int opcode);
  static py::object BuildOperations(const std::vector<py::object> &args, int opcode);

  /**
   * LIST_EXTEND,LIST_APPEND,DICT_MERGE,DICT_UPDATE,SET_UPDATE,SET_ADD,MAP_ADD
   * \return container if success, else a empty AbstractObject
   **/
  static AObject *MergeOperations(AObject *container, std::vector<AObject *> args, int opcode);
  static const char *GetTypeDesc(AObject::Type type);
  static std::string ToString(PyObject *);

 protected:
  static AObject *MakeAObject(Type type, PyTypeObject *tp, PyObject *op, RecMap *rec = nullptr);
  PyTypeObject *type_object_;
  const Type type_;
  int ms_flag_;
};

class AbstractObject : public AbstractObjectBase {
 public:
  AbstractObject(Type type, const py::object &o);
  virtual ~AbstractObject() {}

  py::object GetPyObject() override { return value_; }

  AObject *Binary(AObject *other, int op) override;
  AObject *Unary(int op) const override;
  AObject *UnaryValue(int op) const;
  AObject *GetIter() const override;
  AObject *GetAttr(const std::string &name) override;
  AObject *GetItem(AObject *key);
  bool SetAttr(const std::string &n, AObject *v) override;

 protected:
  py::object value_;
  std::unordered_map<std::string, AObject *> attrs_;  // cache
};

class AbstractType : public AbstractObject {
 public:
  explicit AbstractType(py::object cls)
      : AbstractObject(kTypeType, cls), type_type_(GetPyType(reinterpret_cast<PyTypeObject *>(cls.ptr()))) {
    this->SetTypeObject(&PyType_Type);
  }
  virtual ~AbstractType() {}
  std::string ToString() const override { return std::string(py::str(value_.ptr())); }
  bool IsMindSporeSupportedType() override { return false; }

  Type GetTypeType() const { return type_type_; }
  AObject *BuildAbstractInstance(const std::vector<AObject *> &args, int opcode);
  py::object BuildInstance(const std::vector<py::object> &args, int opcode);

 private:
  Type type_type_;
};

class AbstractSequence : public AbstractObject {
 public:
  explicit AbstractSequence(Type type, const py::object &o) : AbstractObject(type, o) {}
  virtual ~AbstractSequence() {}

  AObject *GetItem(AObject *key) override;
  bool SetItem(AObject *key, AObject *value) override;

  py::object GetPyObject() override { return write_cache_.size() ? py::object() : value_; }

 protected:
  std::unordered_map<AObject *, AObject *> write_cache_;  // cache
};

class AbstractTuple : public AbstractSequence {
 public:
  explicit AbstractTuple(const py::object &l, RecMap *m = nullptr) : AbstractTuple(kTypeTuple, l, m) {}
  virtual ~AbstractTuple() {}
  auto &items() { return items_; }
  Py_ssize_t size() const { return IsElementValid() ? items_.size() : -1; }

  py::object GetPyObject() override { return value_; }
  AObject *Binary(AObject *other, int op) override;
  AObject *Unary(int op) const override;
  AObject *GetAttr(const std::string &name) override;
  bool SetAttr(const std::string &name, AObject *) override { return false; };
  AObject *GetItem(AObject *k) override;
  std::string ToString() const override;
  bool IsMindSporeSupportedType() override;

  void SetElementType(Type type) { element_type_ = type; }
  Type GetElementType() const { return element_type_; }
  bool IsElementValid() const { return element_valid_; }
  void MarkElementInValid() {
    element_type_ = kTypeAnyValue;
    element_valid_ = false;
    modify_ = false;
    value_ = py::object();
    items_.clear();
    write_cache_.clear();
  }
  auto begin() const { return items_.begin(); }
  auto end() const { return items_.end(); }
  bool IsModify() const { return modify_ || this->write_cache_.size() > 0; }
  void MarkModify() { modify_ = true; }
  bool Update(const std::vector<AObject *> &items);
  bool Update();

 protected:
  AbstractTuple(Type type, py::object list, RecMap *m);
  std::vector<AObject *> items_;
  BoolCache ms_support_;
  Type element_type_;
  bool element_valid_;
  bool modify_;
};

class AbstractList : public AbstractTuple {
 public:
  explicit AbstractList(const py::object &l, RecMap *m) : AbstractTuple(kTypeList, l, m) {}
  virtual ~AbstractList() {}

  py::object GetPyObject() override;
  bool SetItem(AObject *k, AObject *v) override;

  bool ListAppend(AObject *item);
  bool ListExtend(AObject *list);
  AbstractTuple *ListToTuple();
};

class AbstractDict : public AbstractSequence {
 public:
  explicit AbstractDict(const py::object &dict, RecMap *m = nullptr) : AbstractDict(kTypeDict, dict, m) {}
  virtual ~AbstractDict() {}
  Py_ssize_t size() const { return IsElementValid() ? dict_.size() : -1; }

  std::string ToString() const override;
  py::object GetPyObject() override;
  AObject *Unary(int op) const override;
  AObject *Binary(AObject *, int op) override;
  AObject *GetAttr(const std::string &name) override;
  bool SetAttr(const std::string &name, AObject *) override { return false; };
  AObject *GetItem(AObject *key) override;
  bool IsMindSporeSupportedType() override;

  Type KeyType() const { return k_type_; }
  Type ValueType() const { return v_type_; }

  bool IsModify() const { return modify_ || this->write_cache_.size() > 0; }
  void MarkModify() { modify_ = true; }
  bool DictMerge(AObject *o, int update = 0);
  bool DictUpdate(AObject *o);
  bool MapAdd(AObject *k, AObject *v);
  bool IsElementValid() const { return element_valid_; }
  void MarkElementInValid() {
    k_type_ = kTypeAnyValue;
    v_type_ = kTypeAnyValue;
    element_valid_ = false;
    modify_ = false;
    value_ = py::object();
    dict_.clear();
    write_cache_.clear();
  }
  bool Update();

  class ValueIter {
   public:
    explicit ValueIter(const AbstractDict *dict) : map_(dict->dict_.ptr()), pos_(0) { ++(*this); }
    ValueIter() : map_(nullptr) {}
    py::object key() { return py::cast<py::object>(key_); }
    AObject *operator*() { return AbstractDict::ConvertValue(val_); }
    bool operator!=(const ValueIter &o) { return map_ != nullptr; }
    ValueIter &operator++() {
      map_ = PyDict_Next(map_, &pos_, &key_, &val_) ? map_ : nullptr;
      return *this;
    }

   private:
    PyObject *map_, *key_, *val_;
    Py_ssize_t pos_;
  };
  auto begin() const { return ValueIter(this); }
  auto end() const { return ValueIter(); }

  static AObject *ConvertValue(PyObject *i) { return reinterpret_cast<AObject *>(PyLong_AsVoidPtr(i)); }
  static py::object ConvertValue(AObject *i) { return py::reinterpret_steal<py::object>(PyLong_FromVoidPtr(i)); }

 protected:
  AbstractDict(Type type, py::object o, RecMap *m);
  py::dict dict_;
  Type k_type_;
  Type v_type_;
  bool element_valid_;
  bool modify_;
};

class AbstractTensor : public AbstractObject {
 public:
  AbstractTensor(const py::object &o, bool is_stub);
  virtual ~AbstractTensor() {}
  AObject *Binary(AObject *, int op) override;
  AObject *Unary(int op) const override;
  AObject *GetAttr(const std::string &name) override;
  std::string ToString() const override;

  bool SetItem(AObject *key, AObject *value) override { return true; }
  AObject *GetItem(AObject *key) override;
  py::object GetTensor(bool sync);

  bool IsMindSporeSupportedType() override { return true; }
  bool IsStubTensor() const { return is_stub_; }

 private:
  bool is_stub_;
};

class AbstractTraceNode : public AbstractObject {
 public:
  explicit AbstractTraceNode(Type type, const py::object &o) : AbstractObject(type, o) {}
  virtual ~AbstractTraceNode() {}
  static AObject *MakeAObject(const py::object &o) {
    if (o.ptr() == nullptr) {
      return AObject::MakeAObject(AObject::kTypeAnyValue);
    }
    auto node = aobject_mem_pool_.New<AbstractObject>(kTypeTraceNode, o);
    node->SetTypeObject(Py_TYPE(o.ptr()));
    return node;
  }
};

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_GRAPH_CAPTURE_ABSTRACT_OBJECT_H
