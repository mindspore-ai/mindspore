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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_ABSTRACT_OBJECT_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_ABSTRACT_OBJECT_H

#include <algorithm>
#include <set>
#include <string>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include "pybind11/pybind11.h"
#include "frontend/jit/pi/utils/mempool.h"
#include "frontend/jit/pi/graph_capture/abstract_wrapper.h"

namespace py = pybind11;
namespace mindspore {
namespace pijit {

class AbstractObjectBase;
using AObject = AbstractObjectBase;

class AbstractObjectBase {
 private:
  class Resource {
   public:
    static Resource *Current() { return weak_this_.empty() ? nullptr : weak_this_.back(); }

   private:
    static std::vector<Resource *> weak_this_;

   public:
    Resource();
    ~Resource();
    void Release() {}
    MemPool<AbstractObjectBase> *pool() { return &pool_; }
    const std::unordered_map<const PyObject *, AObject *> &GetObjMap() const;
    void AddVobj(const py::object &obj, AObject *aobj);

   private:
    MemPool<AbstractObjectBase> pool_;
    std::unordered_map<const PyObject *, AObject *> obj_2_aobj_;
  };

  template <typename T>
  static AObject *ConstructAbstract(const py::object &obj) {
    auto resource = Resource::Current();
    MS_EXCEPTION_IF_NULL(resource);
    auto pool = resource->pool();
    MS_EXCEPTION_IF_NULL(pool);
    return pool->New<T>(obj);
  }

  template <typename T>
  static AObject *ConstructAbstract(const std::vector<AObject *> &elements) {
    auto resource = Resource::Current();
    MS_EXCEPTION_IF_NULL(resource);
    auto pool = resource->pool();
    MS_EXCEPTION_IF_NULL(pool);
    return pool->New<T>(elements);
  }

  template <typename T>
  static AObject *ConstructAbstract(const py::object &obj, const std::vector<AObject *> &elements) {
    if (obj.ptr() == nullptr) {
      return ConstructAbstract<T>(elements);
    }
    return ConstructAbstract<T>(obj);
  }

 public:
  enum Type {
#define ABSTRACT_TYPE_DEF(unit) kType##unit,
#include "abstract_type_kind.def"
#undef ABSTRACT_TYPE_DEF
  };

  static_assert(static_cast<int>(kTypeSlice) + 8 == static_cast<int>(kTypeType));  // builtin type
  static_assert(static_cast<int>(kTypeUnknown) == 0);

  enum Scope {
    SCOPE_NOT_SPECIFIED = 0,
    SCOPE_LOCAL = 1,
    SCOPE_PARAM = 1 << 1,
    SCOPE_FREE_VAR = 1 << 2,
    SCOPE_GLOBAL = 1 << 3
  };

  // record PyObject and check self reference for list,tuple,dict
  static std::unordered_map<AObject::Type, PyTypeObject *> aobj_type_map;
  PyTypeObject *GetPyTypeObject(const Type &type) const;

  explicit AbstractObjectBase(const Type &type) : type_(type), type_object_(GetPyTypeObject(type)) {}
  explicit AbstractObjectBase(const Type &type, PyTypeObject *type_object)
      : type_(type), type_object_(type_object == nullptr ? GetPyTypeObject(type) : type_object) {}
  virtual ~AbstractObjectBase() {}

  PyTypeObject *GetTypeObject() const { return type_object_; }
  Type GetType() const { return type_; }

  virtual py::object GetPyObject() const { return py::object(); }

  virtual AObject *Binary(AObject *other, int op) { return MakeAObject(kTypeAnyValue); }
  virtual AObject *GetIter() const { return MakeAObject(kTypeAnyValue); }

  virtual AObject *GetAttr(const std::string &name);
  virtual AObject *GetItem(AObject *key) { return MakeAObject(kTypeAnyValue); }
  AObject *GetItem(AObject *key, AObject *defalut_value);
  // return false if has an python exception
  virtual bool SetAttr(const std::string &name, AObject *value) { return true; }
  virtual bool IsMindSporeSupportedType();
  virtual std::string ToString() const;

  static Type GetPyType(PyObject *op);
  static Type GetPyType(PyTypeObject *tp);
  static Type GetMsType(PyTypeObject *tp);
  static AObject *Convert(const AbstractWrapperPtr &wrapper);
  static AObject *Convert(const py::object &o) { return Convert(o.ptr()); }
  static AObject *Convert(PyObject *o) { return MakeAObject(GetPyType(o), o ? Py_TYPE(o) : nullptr, o); }
  static AObject *MakeAObject(Type real_type) { return MakeAObject(real_type, nullptr, nullptr); }
  static auto MakeResource() { return Resource(); }
  static AObject *TryConvertDynamicLengthSequence(const abstract::AbstractBasePtr &abstract);

  static AObject *MakeFunction(const std::vector<AObject *> &args, const py::object &globals, int oparg);

  static AObject *FuncAObjectUpdater(const py::object &func, const std::vector<AObject *> &args);

  /**
   * BUILD_SLICE,BUILD_STRING,BUILD_SET,BUILD_LIST,BUILD_TUPLE,BUILD_CONST_KEY_MAP,BUILD_MAP
   * \return a new AbstractObject if success, else a empty AbstractObject
   **/
  static AObject *BuildOperations(const std::vector<AObject *> &args, int opcode, const AbstractWrapperPtr &wrapper);
  static py::object BuildOperations(const std::vector<py::object> &args, int opcode);

  /**
   * LIST_EXTEND,LIST_APPEND,DICT_MERGE,DICT_UPDATE,SET_UPDATE,SET_ADD,MAP_ADD
   * \return container if success, else a empty AbstractObject
   **/
  static AObject *MergeOperations(AObject *container, std::vector<AObject *> args, int opcode);

  static int BinaryContains(AObject *l, AObject *r);
  static int BinaryIs(AObject *l, AObject *r);

  static const char *GetTypeDesc(AObject::Type type);
  static std::string ToString(PyObject *, bool print_type = true, size_t limit = SIZE_MAX);
  bool IsLatestVersion() const { return next_version_ == nullptr; }
  AObject *GetLatestVersion() const;
  const AObject *GetPreVersion() const { return pre_version_; }
  void SetPreVersion(AObject *pre_version);
  const AObject *GetNextVersion() const { return next_version_; }
  void SetNextVersion(AObject *next_version);
  const AObject *GetBaseVersion() const;
  bool IsBaseVersion() const { return this == GetBaseVersion(); }
  const std::set<AObject *> &GetUsers() const { return users_; }
  void AddUser(AObject *user) { users_.insert(user); }
  void RemoveUser(AObject *user) { users_.erase(user); }
  bool HasMultiVersion() const { return pre_version_ != nullptr || next_version_ != nullptr; }
  virtual void CreateVersionWithNewValue() {}
  void SetScope(Scope scope) { scope_ = scope; }
  void AddScope(Scope scope) { scope_ = static_cast<Scope>(static_cast<int>(scope_) | static_cast<int>(scope)); }
  Scope GetScope() const { return scope_; }
  const std::string &GetScopeDesc() const;

 protected:
  static AObject *Convert(const abstract::AbstractBasePtr &abstract);
  static AObject *MakeAObject(Type type, PyTypeObject *tp, PyObject *op, const std::vector<AObject *> &elements = {});
  const Type type_;
  PyTypeObject *const type_object_;
  AObject *pre_version_{nullptr};
  AObject *next_version_{nullptr};
  std::set<AObject *> users_;
  AbstractWrapperPtr abstract_wrapper_{nullptr};
  Scope scope_{SCOPE_NOT_SPECIFIED};
};

class AbstractObject : public AbstractObjectBase {
 public:
  AbstractObject(const Type &type, const py::object &obj)
      : AbstractObjectBase(type, obj.ptr() == nullptr ? nullptr : Py_TYPE(obj.ptr())), value_(obj) {}
  virtual ~AbstractObject() {}

  py::object GetPyObject() const override { return value_; }

  AObject *GetIter() const override;
  AObject *GetAttr(const std::string &name) override;
  AObject *GetItem(AObject *key) override;
  bool SetAttr(const std::string &n, AObject *v) override;
  std::string ToString() const override;

 protected:
  py::object value_;
  std::unordered_map<std::string, AObject *> attrs_;  // cache
};

class AbstractString : public AbstractObject {
 public:
  AbstractString() : AbstractObject(kTypeString, py::object()), str_() {}
  explicit AbstractString(const py::object &str)
      : AbstractObject(kTypeString, str), str_(str.ptr() == nullptr ? std::string() : py::cast<std::string>(str)) {}
  ~AbstractString() override = default;
  AObject *GetItem(AObject *index) override;

 protected:
  std::string str_;
};

class AbstractType : public AbstractObject {
 public:
  explicit AbstractType(const py::object &cls)
      : AbstractObject(kTypeType, cls), type_type_(GetPyType(reinterpret_cast<PyTypeObject *>(cls.ptr()))) {}
  ~AbstractType() override = default;
  bool IsMindSporeSupportedType() override { return false; }

  Type GetTypeType() const { return type_type_; }
  AObject *BuildAbstractInstance(const std::vector<AObject *> &args, int opcode);
  py::object BuildInstance(const std::vector<py::object> &args, int opcode, const py::object &kw);

 private:
  Type type_type_;
};

class AbstractSequence : public AbstractObject {
 public:
  explicit AbstractSequence(Type type, const py::object &obj)
      : AbstractObject(type, obj), element_type_(kTypeUnknown), elements_({}) {}
  explicit AbstractSequence(Type type, const std::vector<AObject *> &elements);
  ~AbstractSequence() override = default;

  AObject *Binary(AObject *other, int op) override;
  AObject *GetAttr(const std::string &name) override;
  bool SetAttr(const std::string &name, AObject *) override { return false; };
  AObject *GetItem(AObject *key) override;
  void CreateVersionWithNewValue() override;

  /// \brief Get the element size of AbstractSequence.
  ///
  /// \return The size of elements_.
  std::size_t size() const;

  /// \brief The elements of AbstractSequence object.
  ///
  /// \return The vector of AObject objects.
  const std::vector<AObject *> &GetElements() const { return elements_; }
  const std::vector<AObject *> &GetElementsWithInit() {
    InitElementsListIfNeed();
    return elements_;
  }
  bool IsMindSporeSupportedType() override;

  void SetElementType(Type type) { element_type_ = type; }
  Type GetElementType() const { return element_type_; }
  auto begin() const { return elements_.begin(); }
  auto end() const { return elements_.end(); }
  std::string ToString() const override;

 protected:
  void InitElementsListIfNeed();

  Type element_type_;
  std::vector<AObject *> elements_;
};

class AbstractTuple : public AbstractSequence {
 public:
  explicit AbstractTuple(const py::object &tuple) : AbstractSequence(kTypeTuple, tuple) {}
  explicit AbstractTuple(const std::vector<AObject *> &elements) : AbstractSequence(kTypeTuple, elements) {}
  ~AbstractTuple() override = default;
};

class AbstractNamedTuple : public AbstractObject {
 public:
  AbstractNamedTuple(const py::object &o, PyTypeObject *tp);
  ~AbstractNamedTuple() override = default;

  static bool IsNamedTuple(PyTypeObject *tp);

  bool HasKey(const std::string &name) const;
  int GetIndexOfKey(const std::string &name) const;

  const std::string &type_name() const { return type_name_; }
  const std::vector<std::string> &keys() const { return keys_; }
  size_t Size() const { return keys_.size(); }

 private:
  std::string type_name_;
  std::vector<std::string> keys_;
};

class AbstractList : public AbstractSequence {
 public:
  explicit AbstractList(const py::object &list) : AbstractSequence(kTypeList, list) {}
  explicit AbstractList(const std::vector<AObject *> &elements) : AbstractSequence(kTypeList, elements) {}
  ~AbstractList() override = default;

  AbstractList *ListAppend(AObject *item);
  AbstractList *ListExtend(AObject *list);
  AbstractTuple *ListToTuple();
};

class AbstractCellList : public AbstractSequence {
 public:
  explicit AbstractCellList(const py::object &cells) : AbstractSequence(kTypeNNCellList, cells) {}
  explicit AbstractCellList(const std::vector<AObject *> &cells) : AbstractSequence(kTypeNNCellList, cells) {}
  ~AbstractCellList() override = default;
};

class AbstractDictKeys : public AbstractSequence {
 public:
  explicit AbstractDictKeys(const py::object &keys) : AbstractSequence(kTypeDictKeys, keys) {}
  explicit AbstractDictKeys(const std::vector<AObject *> &keys) : AbstractSequence(kTypeDictKeys, keys) {}
  ~AbstractDictKeys() override = default;
};

class AbstractDictValues : public AbstractSequence {
 public:
  explicit AbstractDictValues(const py::object &values) : AbstractSequence(kTypeDictValues, values) {}
  explicit AbstractDictValues(const std::vector<AObject *> &values) : AbstractSequence(kTypeDictValues, values) {}
  ~AbstractDictValues() override = default;
};

class AbstractDictItems : public AbstractSequence {
 public:
  explicit AbstractDictItems(const py::object &items) : AbstractSequence(kTypeDictItems, items) {}
  explicit AbstractDictItems(const std::vector<AObject *> &items) : AbstractSequence(kTypeDictItems, items) {}
  ~AbstractDictItems() override = default;
};

using AObjectPair = std::pair<AObject *, AObject *>;
using AObjectPairList = std::vector<AObjectPair>;

class AbstractDict : public AbstractObject {
 public:
  explicit AbstractDict(const py::object &dict) : AbstractObject(kTypeDict, dict) {}
  explicit AbstractDict(const std::vector<AObject *> &key_values);
  virtual ~AbstractDict() {}

  std::string ToString() const override;
  AObject *GetAttr(const std::string &name) override;
  bool SetAttr(const std::string &name, AObject *) override { return false; };
  AObject *GetItem(AObject *key) override;
  bool IsMindSporeSupportedType() override;

  AObject *Keys();
  AObject *Values();
  AObject *Items();

  Type KeyType() const { return k_type_; }
  Type ValueType() const { return v_type_; }

  bool DictMerge(const AObject *dict);
  bool DictUpdate(const AObject *dict);
  bool MapAdd(AObject *k, AObject *v);
  void CreateVersionWithNewValue() override;
  /// \brief Get the size of dictionary.
  ///
  /// \return The size of dictionary.
  std::size_t size() const;

  /// \brief The elements of AbstractDict object.
  ///
  /// \return The elements of AbstractDict object.
  const AObjectPairList &GetElements() const { return key_values_; }
  const AObjectPairList &GetElementsWithInit() {
    InitKeyValuesListIfNeed();
    return key_values_;
  }

  class ValueIter {
   public:
    explicit ValueIter(const AbstractDict *dict) : map_(dict->value_.ptr()), pos_(0) { ++(*this); }
    ValueIter() : map_(nullptr) {}
    py::object key() { return py::cast<py::object>(key_); }
    AObject *operator*() { return AbstractDict::ConvertValue(val_); }
    bool operator!=(const ValueIter &o) { return map_ != nullptr; }
    ValueIter &operator++() {
      map_ = PyDict_Next(map_, &pos_, &key_, &val_) ? map_ : nullptr;
      return *this;
    }

   private:
    PyObject *map_{nullptr};
    PyObject *key_{nullptr};
    PyObject *val_{nullptr};
    Py_ssize_t pos_;
  };
  auto begin() const { return ValueIter(this); }
  auto end() const { return ValueIter(); }

  static AObject *ConvertValue(PyObject *i) { return reinterpret_cast<AObject *>(PyLong_AsVoidPtr(i)); }
  static py::object ConvertValue(AObject *i) { return py::reinterpret_steal<py::object>(PyLong_FromVoidPtr(i)); }

 protected:
  void InitKeyValuesListIfNeed();

  Type k_type_;
  Type v_type_;
  AObjectPairList key_values_;
};

class AbstractTensor : public AbstractObject {
 public:
  static py::object Binary(int op, const py::object &, const py::object &);

 public:
  AbstractTensor(const py::object &o, bool is_stub);
  virtual ~AbstractTensor() {}
  AObject *GetAttr(const std::string &name) override;
  std::string ToString() const override;

  AObject *GetItem(AObject *key) override;
  py::object GetTensor(bool sync);

  bool IsMindSporeSupportedType() override { return true; }

 private:
  bool is_stub_;
};

void PrintPyObject(std::ostream *out_s, const py::handle &obj, bool print_type);
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_ABSTRACT_OBJECT_H
