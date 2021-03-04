/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_ABSTRACT_ABSTRACT_VALUE_H_
#define MINDSPORE_CORE_ABSTRACT_ABSTRACT_VALUE_H_

#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

#include "utils/log_adapter.h"
#include "utils/hashing.h"
#include "utils/any.h"
#include "utils/flags.h"
#include "base/base.h"
#include "ir/dtype.h"
#include "ir/value.h"
#include "ir/tensor.h"
#include "abstract/dshape.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace abstract {
class AbstractBase;
using AbstractBasePtrList = std::vector<AbstractBasePtr>;

// The base class for abstract value. The abstract value is used in evaluating
// to express the type, shape, and value of the real value.
class AbstractBase : public Base {
 public:
  explicit AbstractBase(const ValuePtr &value = nullptr, const TypePtr &type = kAnyType,
                        const BaseShapePtr &shape = kNoShape)
      : value_(value), type_(type), shape_(shape) {}
  ~AbstractBase() override = default;
  MS_DECLARE_PARENT(AbstractBase, Base)

  std::size_t hash() const override { return tid(); }
  std::string ToString() const override;

  virtual bool operator==(const AbstractBase &other) const;
  void set_value(const ValuePtr &value) { value_ = value; }
  void set_type(const TypePtr &type) { type_ = type; }
  void set_shape(const BaseShapePtr &shape) { shape_ = shape; }
  void set_value_desc(const std::string &desc) { value_desc_ = desc; }
  const std::string &value_desc() const { return value_desc_; }
  ValuePtr GetValueTrack() const { return value_; }
  TypePtr GetTypeTrack() const { return type_; }
  BaseShapePtr GetShapeTrack() const { return shape_; }

  // Try build a real value from an abstract value. If the value cannot be built,
  // a default value (AnyValue) is returned.
  ValuePtr BuildValue() const;

  virtual TypePtr BuildType() const = 0;
  virtual BaseShapePtr BuildShape() const { return kNoShape; }
  virtual AbstractBasePtr Clone() const = 0;

  // mask for Broaden config
  inline static const uint8_t kBroadenTensorOnly = 1;
  inline static const uint8_t kBroadenParameterOnly = 2;
  // Each bit for on config.
  // 00000001 -> 1: only boarden tensor
  // 00000010 -> 2: only boarden parameter
  virtual AbstractBasePtr Broaden(uint8_t config = 0) const;
  virtual AbstractBasePtr Join(const AbstractBasePtr &) { return shared_from_base<AbstractBase>(); }

  friend std::ostream &operator<<(std::ostream &os, const std::shared_ptr<AbstractBase> &a) {
    os << a->ToString();
    return os;
  }

 protected:
  // default implementation, it can be overwritten by subclass;
  virtual ValuePtr RealBuildValue() const { return kAnyValue; }

 private:
  ValuePtr value_;
  TypePtr type_;
  BaseShapePtr shape_;
  std::string value_desc_;  // store initial value description for error report
};

class AbstractScalar : public AbstractBase {
 public:
  AbstractScalar() : AbstractBase(kAnyValue, kAnyType) {}
  explicit AbstractScalar(const ValuePtr &value, const TypePtr &type) : AbstractBase(value, type) {}
  explicit AbstractScalar(const ValuePtr &value) : AbstractBase(value, value->type()) {}
  explicit AbstractScalar(int value) : AbstractBase(MakeValue(value), kInt32) {}
  explicit AbstractScalar(int64_t value) : AbstractBase(MakeValue(value), kInt64) {}
  explicit AbstractScalar(float value) : AbstractBase(MakeValue(value), kFloat32) {}
  explicit AbstractScalar(double value) : AbstractBase(MakeValue(value), kFloat64) {}
  explicit AbstractScalar(bool value) : AbstractBase(MakeValue(value), kBool) {}
  explicit AbstractScalar(const std::string &value) : AbstractBase(MakeValue(value), kString) {}
  explicit AbstractScalar(const TypePtr &type) : AbstractBase(kAnyValue, type) {}
  ~AbstractScalar() override = default;
  MS_DECLARE_PARENT(AbstractScalar, AbstractBase)

  std::size_t hash() const override { return hash_combine({tid(), GetValueTrack()->hash(), GetTypeTrack()->hash()}); }

  TypePtr BuildType() const override { return GetTypeTrack(); }
  AbstractBasePtr Clone() const override {
    return std::make_shared<AbstractScalar>(GetValueTrack(), GetTypeTrack()->Clone());
  }
  AbstractBasePtr Broaden(uint8_t config = 0) const override;
  AbstractBasePtr Join(const AbstractBasePtr &other) override;
};
using AbstractScalarPtr = std::shared_ptr<AbstractScalar>;

class AbstractType : public AbstractBase {
 public:
  explicit AbstractType(const TypePtr &type) : AbstractBase(type, kTypeType) {
    if (type == nullptr) {
      MS_LOG(EXCEPTION) << "type is nullptr";
    }
  }
  ~AbstractType() override = default;
  MS_DECLARE_PARENT(AbstractType, AbstractBase)

  std::string ToString() const override;
  bool operator==(const AbstractBase &other) const override;

  TypePtr BuildType() const override { return std::make_shared<TypeType>(); }
  AbstractBasePtr Clone() const override;
  AbstractBasePtr Broaden(uint8_t config = 0) const override { return Clone(); }
};
using AbstractTypePtr = std::shared_ptr<AbstractType>;

class AbstractError : public AbstractBase {
 public:
  explicit AbstractError(const StringImmPtr &err, const AnfNodePtr &node) : AbstractBase(err), node_(node) {
    if (err == nullptr || node == nullptr) {
      MS_LOG(EXCEPTION) << "err or node is nullptr";
    }
  }
  ~AbstractError() override = default;
  MS_DECLARE_PARENT(AbstractError, AbstractBase)

  TypePtr BuildType() const override { return std::make_shared<Problem>(); }
  AbstractBasePtr Broaden(uint8_t config = 0) const override { return Clone(); }

  AbstractBasePtr Clone() const override {
    return std::make_shared<AbstractError>(GetValueTrack()->cast<StringImmPtr>(), node_);
  }

  std::string ToString() const override;

 private:
  // Origin node been specialized to AbstractError, for debug purpose only.
  const AnfNodePtr node_;
};

class Evaluator;
using EvaluatorPtr = std::shared_ptr<Evaluator>;
class AnalysisEngine;
using AnalysisEnginePtr = std::shared_ptr<AnalysisEngine>;

class AbstractFunction;
using AbstractFunctionPtr = std::shared_ptr<AbstractFunction>;
class AbstractFuncAtom;
using AbstractFuncAtomPtr = std::shared_ptr<AbstractFuncAtom>;
using AbstractFuncAtomPtrList = std::vector<AbstractFuncAtomPtr>;

class AbstractFunction : public AbstractBase {
 public:
  AbstractFunction() = default;
  ~AbstractFunction() override = default;
  MS_DECLARE_PARENT(AbstractFunction, AbstractBase)

  // If there is exactly one possible function, return it. Otherwise, raise an Exception.
  // Caller should ensure the uniqueness.
  virtual AbstractFunctionPtr GetUnique() = 0;

  TypePtr BuildType() const override { return std::make_shared<Function>(); }
  AbstractBasePtr Clone() const override { return Copy(); }
  // For Function, no need to broaden.
  AbstractBasePtr Broaden(uint8_t config = 0) const override {
    return const_cast<AbstractFunction *>(this)->shared_from_base<AbstractFunction>();
  }
  virtual AbstractFunctionPtr Copy() const = 0;

  AbstractBasePtr Join(const AbstractBasePtr &other) final;
  virtual AbstractFunctionPtr Join(const AbstractFunctionPtr &other) = 0;

  virtual void Visit(std::function<void(const AbstractFuncAtomPtr &)>) const = 0;
  bool operator==(const AbstractBase &other) const final;
  virtual bool operator==(const AbstractFunction &other) const = 0;

  static AbstractFunctionPtr MakeAbstractFunction(const AbstractFuncAtomPtrList &func_list);

  virtual AnfNodePtr tracking_id() const { return nullptr; }
  virtual void set_tracking_id(AnfNodePtr) {}
  virtual AnalysisContextPtr context() const { return nullptr; }
};
using AbstractFunctionPtrList = std::vector<AbstractFunctionPtr>;

// Represents a key-value pair used in function's parameters.
class AbstractKeywordArg : public AbstractBase {
 public:
  AbstractKeywordArg(const std::string &key, const AbstractBasePtr &argument) : arg_name_(key), arg_value_(argument) {}
  ~AbstractKeywordArg() override = default;
  MS_DECLARE_PARENT(AbstractKeywordArg, AbstractBase)

  TypePtr BuildType() const override;
  AbstractBasePtr Clone() const override;
  AbstractBasePtr Broaden(uint8_t config = 0) const override;
  std::size_t hash() const override;

  bool operator==(const AbstractKeywordArg &other) const;
  bool operator==(const AbstractBase &other) const override;
  std::string get_key() const { return arg_name_; }
  AbstractBasePtr get_arg() const { return arg_value_; }

  std::string ToString() const override;

 protected:
  ValuePtr RealBuildValue() const override;

 private:
  std::string arg_name_;
  AbstractBasePtr arg_value_;
};
using AbstractKeywordArgPtr = std::shared_ptr<AbstractKeywordArg>;

class AbstractUndetermined : public AbstractBase {
 public:
  // shape and type are all unknown
  AbstractUndetermined() : AbstractBase(kAnyValue) {}
  // only element_ and value, shape track are valid member, type track are unknown.
  explicit AbstractUndetermined(const AbstractBasePtr &element, const BaseShapePtr &shape = std::make_shared<Shape>())
      : AbstractBase(kAnyValue), element_(element) {
    if (element == nullptr) {
      MS_LOG(EXCEPTION) << "element is nullptr";
    }
    if (element->isa<AbstractUndetermined>()) {
      MS_LOG(EXCEPTION) << "element type error";
    }
    set_shape(shape);
  }
  AbstractUndetermined(const TypePtr &element_type, const ShapeVector &shape)
      : AbstractBase(kAnyValue), element_(std::make_shared<AbstractScalar>(kAnyValue, element_type)) {
    if (element_type == nullptr) {
      MS_LOG(EXCEPTION) << "element_type is nullptr";
    }
    set_shape(std::make_shared<Shape>(shape));
  }
  explicit AbstractUndetermined(const TypePtr &element_type, const BaseShapePtr &shape = std::make_shared<Shape>())
      : AbstractBase(kAnyValue), element_(std::make_shared<AbstractScalar>(kAnyValue, element_type)) {
    if (element_type == nullptr) {
      MS_LOG(EXCEPTION) << "element_type is nullptr";
    }
    set_shape(shape);
  }
  ~AbstractUndetermined() override = default;
  MS_DECLARE_PARENT(AbstractUndetermined, AbstractBase)
  TypePtr BuildType() const override { return std::make_shared<UndeterminedType>(); }
  AbstractBasePtr Clone() const override { return std::make_shared<AbstractUndetermined>(); }
  const AbstractBasePtr element() const { return element_; }
  ShapePtr shape() const;

 protected:
  AbstractBasePtr element_;
};

class AbstractTensor : public AbstractUndetermined {
 public:
  // only element_ and value, shape track are valid member, type track are unknown.
  explicit AbstractTensor(const AbstractBasePtr &element, const BaseShapePtr &shape = std::make_shared<Shape>())
      : AbstractUndetermined(element, shape) {}
  AbstractTensor(const TypePtr &element_type, const ShapeVector &shape) : AbstractUndetermined(element_type, shape) {}
  explicit AbstractTensor(const tensor::TensorPtr &tensor) : AbstractUndetermined(tensor->Dtype(), tensor->shape()) {}
  explicit AbstractTensor(const TypePtr &element_type, const BaseShapePtr &shape = std::make_shared<Shape>())
      : AbstractUndetermined(element_type, shape) {}
  ~AbstractTensor() override = default;
  MS_DECLARE_PARENT(AbstractTensor, AbstractUndetermined)

  void set_value_range(const ValuePtr &min_value, const ValuePtr &max_value) {
    min_value_ = min_value;
    max_value_ = max_value;
  }
  const ValuePtr &get_min_value() const { return min_value_; }
  const ValuePtr &get_max_value() const { return max_value_; }

  TypePtr BuildType() const override;
  BaseShapePtr BuildShape() const override;
  AbstractBasePtr Clone() const override;
  AbstractBasePtr Broaden(uint8_t config = 0) const override;
  AbstractBasePtr BroadenWithShape() const;
  AbstractBasePtr Join(const AbstractBasePtr &other);
  bool operator==(const AbstractTensor &other) const;
  bool operator==(const AbstractBase &other) const override;
  std::string ToString() const override;
  std::size_t hash() const override {
    auto value = GetValueTrack();
    auto hash_sum = hash_combine(tid(), element_->hash());
    if (value != nullptr) {
      auto tensor = value->cast<tensor::TensorPtr>();
      if (tensor != nullptr) {
        hash_sum = hash_combine(hash_sum, LongToSize(tensor->DataSize()));
      }
    }
    return hash_sum;
  }

 protected:
  bool equal_to(const AbstractTensor &other) const;
  ValuePtr min_value_ = nullptr;
  ValuePtr max_value_ = nullptr;
};
using AbstractTensorPtr = std::shared_ptr<AbstractTensor>;
using AbstractTensorPtrList = std::vector<AbstractTensorPtr>;

class AbstractSequeue : public AbstractBase {
 public:
  explicit AbstractSequeue(const AbstractBasePtrList &elements) : elements_(elements) {}
  ~AbstractSequeue() override = default;
  MS_DECLARE_PARENT(AbstractSequeue, AbstractBase)

  TypePtrList ElementsType() const;
  BaseShapePtrList ElementsShape() const;
  AbstractBasePtrList ElementsClone() const;
  AbstractBasePtrList ElementsBroaden(uint8_t config = 0) const;

  template <typename T>
  ValuePtr ElementsBuildValue() const;

  template <typename T>
  AbstractBasePtr ElementsJoin(const AbstractBasePtr &other);

  std::size_t size() const { return elements_.size(); }
  const AbstractBasePtrList &elements() const { return elements_; }

  std::size_t hash() const override;
  std::string ToString() const override;
  const AbstractBasePtr operator[](const std::size_t &dim) const;

 protected:
  AbstractBasePtrList elements_;
};
using AbstractSequeuePtr = std::shared_ptr<AbstractSequeue>;

class AbstractTuple : public AbstractSequeue {
 public:
  explicit AbstractTuple(const AbstractBasePtrList &elements) : AbstractSequeue(elements) {}

  ~AbstractTuple() override = default;
  MS_DECLARE_PARENT(AbstractTuple, AbstractSequeue)

  TypePtr BuildType() const override { return std::make_shared<Tuple>(ElementsType()); }

  BaseShapePtr BuildShape() const override { return std::make_shared<TupleShape>(ElementsShape()); }

  AbstractBasePtr Clone() const override { return std::make_shared<AbstractTuple>(ElementsClone()); }

  AbstractBasePtr Broaden(uint8_t config = 0) const override {
    return std::make_shared<AbstractTuple>(ElementsBroaden(config));
  }

  AbstractBasePtr Join(const AbstractBasePtr &other) override { return ElementsJoin<AbstractTuple>(other); }

  std::string ToString() const override { return type_name() + "(" + AbstractSequeue::ToString() + ")"; }

  bool operator==(const AbstractTuple &other) const;
  bool operator==(const AbstractBase &other) const override;

 protected:
  ValuePtr RealBuildValue() const override { return ElementsBuildValue<ValueTuple>(); }
};
using AbstractTuplePtr = std::shared_ptr<AbstractTuple>;

class AbstractList : public AbstractSequeue {
 public:
  explicit AbstractList(const AbstractBasePtrList &elements) : AbstractSequeue(elements) {}

  ~AbstractList() override = default;
  MS_DECLARE_PARENT(AbstractList, AbstractSequeue)

  TypePtr BuildType() const override { return std::make_shared<List>(ElementsType()); }

  BaseShapePtr BuildShape() const override { return std::make_shared<ListShape>(ElementsShape()); }

  AbstractBasePtr Clone() const override { return std::make_shared<AbstractList>(ElementsClone()); }

  AbstractBasePtr Broaden(uint8_t config = 0) const override {
    return std::make_shared<AbstractList>(ElementsBroaden(config));
  }

  AbstractBasePtr Join(const AbstractBasePtr &other) override { return ElementsJoin<AbstractList>(other); }

  std::string ToString() const override { return type_name() + "[" + AbstractSequeue::ToString() + "]"; }

  bool operator==(const AbstractList &other) const;
  bool operator==(const AbstractBase &other) const override;

 protected:
  ValuePtr RealBuildValue() const override { return ElementsBuildValue<ValueList>(); }
};
using AbstractListPtr = std::shared_ptr<AbstractList>;

class AbstractClass : public AbstractBase {
 public:
  AbstractClass(const Named &tag, const std::vector<AbstractAttribute> &attributes,
                const std::unordered_map<std::string, ValuePtr> &methods)
      : attributes_(attributes), tag_(tag), methods_(methods) {}

  ~AbstractClass() override = default;
  MS_DECLARE_PARENT(AbstractClass, AbstractBase)

  TypePtr BuildType() const override;
  bool operator==(const AbstractClass &other) const;
  bool operator==(const AbstractBase &other) const override;
  const std::vector<AbstractAttribute> &attributes() const { return attributes_; }
  std::unordered_map<std::string, ValuePtr> methods() { return methods_; }
  AbstractBasePtr GetAttribute(const std::string &name);
  ValuePtr GetMethod(const std::string &name);
  AbstractBasePtr Clone() const override;
  AbstractBasePtr Broaden(uint8_t config = 0) const override;
  std::string ToString() const override;
  Named tag() const { return tag_; }
  std::size_t hash() const override;

 protected:
  ValuePtr RealBuildValue() const override;

 private:
  std::vector<AbstractAttribute> attributes_;
  Named tag_;
  std::unordered_map<std::string, ValuePtr> methods_;
};
using AbstractClassPtr = std::shared_ptr<AbstractClass>;

class AbstractDictionary : public AbstractBase {
 public:
  explicit AbstractDictionary(const std::vector<AbstractAttribute> &key_values) : key_values_(key_values) {}
  ~AbstractDictionary() override = default;
  MS_DECLARE_PARENT(AbstractDictionary, AbstractBase)

  TypePtr BuildType() const override;
  bool operator==(const AbstractDictionary &other) const;
  bool operator==(const AbstractBase &other) const override;
  AbstractBasePtr Clone() const override;
  AbstractBasePtr Broaden(uint8_t config = 0) const override;
  std::string ToString() const override;
  std::size_t hash() const override;
  std::size_t size() const { return key_values_.size(); }
  const std::vector<AbstractAttribute> &elements() const { return key_values_; }

  std::vector<AbstractAttribute> key_values_;

 protected:
  ValuePtr RealBuildValue() const override;
};
using AbstractDictionaryPtr = std::shared_ptr<AbstractDictionary>;

class AbstractSlice : public AbstractBase {
 public:
  AbstractSlice(const AbstractBasePtr &start, const AbstractBasePtr &stop, const AbstractBasePtr &step)
      : start_(start), stop_(stop), step_(step) {}
  ~AbstractSlice() override = default;
  MS_DECLARE_PARENT(AbstractSlice, AbstractBase)

  TypePtr BuildType() const override;
  bool operator==(const AbstractSlice &other) const;
  bool operator==(const AbstractBase &other) const override;
  AbstractBasePtr Clone() const override;
  AbstractBasePtr Broaden(uint8_t config = 0) const override;
  std::string ToString() const override;
  std::size_t hash() const override;
  AbstractBasePtr start() const { return start_; }
  AbstractBasePtr stop() const { return stop_; }
  AbstractBasePtr step() const { return step_; }

 protected:
  ValuePtr RealBuildValue() const override;

 private:
  AbstractBasePtr start_;
  AbstractBasePtr stop_;
  AbstractBasePtr step_;
};
using AbstractSlicePtr = std::shared_ptr<AbstractSlice>;

class AbstractJTagged : public AbstractBase {
 public:
  explicit AbstractJTagged(const AbstractBasePtr &element) : element_(element) {}

  ~AbstractJTagged() override = default;
  MS_DECLARE_PARENT(AbstractJTagged, AbstractBase)

  TypePtr BuildType() const override;
  AbstractBasePtr Clone() const override { return std::make_shared<AbstractJTagged>(element_->Clone()); }
  AbstractBasePtr Broaden(uint8_t config = 0) const override {
    return std::make_shared<AbstractJTagged>(element_->Broaden(config));
  }
  AbstractBasePtr Join(const AbstractBasePtr &other) override;

  bool operator==(const AbstractJTagged &other) const;
  bool operator==(const AbstractBase &other) const override;
  std::string ToString() const override;
  AbstractBasePtr element() { return element_; }
  std::size_t hash() const override { return hash_combine(tid(), element_->hash()); }

 private:
  AbstractBasePtr element_;
};
using AbstractJTaggedPtr = std::shared_ptr<AbstractJTagged>;

class AbstractNone : public AbstractBase {
 public:
  AbstractNone() : AbstractBase() { set_type(std::make_shared<TypeNone>()); }
  ~AbstractNone() override = default;
  MS_DECLARE_PARENT(AbstractNone, AbstractBase)

  TypePtr BuildType() const override { return std::make_shared<TypeNone>(); }
  bool operator==(const AbstractNone &other) const;
  bool operator==(const AbstractBase &other) const override;
  AbstractBasePtr Clone() const override { return std::make_shared<AbstractNone>(); }
  std::string ToString() const override;

 protected:
  ValuePtr RealBuildValue() const override;
};
using AbstractNonePtr = std::shared_ptr<AbstractNone>;

// the un assigned state value for variable, which means the variable is not assigned
class AbstractNull : public AbstractBase {
 public:
  AbstractNull() : AbstractBase(kNull) { set_type(std::make_shared<TypeNull>()); }
  ~AbstractNull() override = default;
  MS_DECLARE_PARENT(AbstractNull, AbstractBase)

  TypePtr BuildType() const override { return std::make_shared<TypeNull>(); }
  bool operator==(const AbstractNull &other) const;
  bool operator==(const AbstractBase &other) const override;
  AbstractBasePtr Clone() const override { return std::make_shared<AbstractNull>(); }
  std::string ToString() const override;
};
using AbstractNullPtr = std::shared_ptr<AbstractNull>;

class AbstractEllipsis : public AbstractBase {
 public:
  AbstractEllipsis() : AbstractBase(kEllipsis) { set_type(std::make_shared<TypeEllipsis>()); }
  ~AbstractEllipsis() override = default;
  MS_DECLARE_PARENT(AbstractEllipsis, AbstractBase)

  TypePtr BuildType() const override { return std::make_shared<TypeEllipsis>(); }
  bool operator==(const AbstractEllipsis &other) const;
  bool operator==(const AbstractBase &other) const override;
  AbstractBasePtr Clone() const override { return std::make_shared<AbstractEllipsis>(); }
  std::string ToString() const override;
};
using AbstractEllipsisPtr = std::shared_ptr<AbstractEllipsis>;

class AbstractRefKey : public AbstractBase {
 public:
  AbstractRefKey() : AbstractBase(), ref_key_value_(nullptr) { set_type(std::make_shared<RefKeyType>()); }
  ~AbstractRefKey() override = default;
  MS_DECLARE_PARENT(AbstractRefKey, AbstractBase)

  TypePtr BuildType() const override { return std::make_shared<RefKeyType>(); }
  bool operator==(const AbstractRefKey &other) const;
  bool operator==(const AbstractBase &other) const override;
  AbstractBasePtr Clone() const override {
    auto cloned = std::make_shared<AbstractRefKey>();
    cloned->set_value(GetValueTrack());
    return cloned;
  }
  inline void set_value(const ValuePtr &value) {
    AbstractBase::set_value(value);
    if (value != nullptr) {
      ref_key_value_ = value->cast<RefKeyPtr>();
    }
  }
  RefKeyPtr ref_key_value() const { return ref_key_value_; }
  AbstractBasePtr Join(const AbstractBasePtr &other) override;
  AbstractBasePtr Broaden(uint8_t config = 0) const override;
  std::string ToString() const override;

 private:
  // cache for ref_key after build value, when value is null, return nullptr.
  RefKeyPtr ref_key_value_{nullptr};
};
using AbstractRefKeyPtr = std::shared_ptr<AbstractRefKey>;

class AbstractRef : public AbstractTensor {
 public:
  AbstractRef(const AbstractBasePtr &ref_key, const AbstractTensorPtr &ref_value);

  ~AbstractRef() override = default;
  MS_DECLARE_PARENT(AbstractRef, AbstractTensor)

  TypePtr BuildType() const override;
  bool operator==(const AbstractRef &other) const;
  bool operator==(const AbstractBase &other) const override;
  AbstractBasePtr Clone() const override {
    auto abs_tensor = AbstractTensor::Clone()->cast<AbstractTensorPtr>();
    if (abs_tensor == nullptr) {
      return nullptr;
    }
    return std::make_shared<AbstractRef>(ref_key_->Clone(), abs_tensor);
  }
  AbstractBasePtr CloneAsTensor() const { return AbstractTensor::Clone(); }
  std::string ToString() const override;
  inline AbstractTensorPtr ref() { return shared_from_base<AbstractTensor>(); }
  inline AbstractBasePtr ref_key() const { return ref_key_; }
  inline RefKeyPtr ref_key_value() const { return ref_key_value_; }
  AbstractBasePtr Broaden(uint8_t config = 0) const override {
    // always broaden for ref
    auto abs_tensor = AbstractTensor::Broaden()->cast<AbstractTensorPtr>();
    if (abs_tensor == nullptr) {
      return nullptr;
    }
    return std::make_shared<AbstractRef>(ref_key_->Broaden(config), abs_tensor);
  }
  AbstractBasePtr Join(const AbstractBasePtr &other) override;
  std::size_t hash() const override {
    return AbstractTensor::hash() ^ (std::hash<uint32_t>{}(this->tid()) << 1);  // ref_key_->hash() ^
  }

 private:
  AbstractBasePtr ref_key_;
  // cache for ref_key after build value, when value is null, return nullptr.
  RefKeyPtr ref_key_value_;
};
using AbstractRefPtr = std::shared_ptr<AbstractRef>;

struct AbstractBasePtrListHasher {
  std::size_t operator()(const AbstractBasePtrList &args_spec_list) const;
};

struct AbstractBasePtrListEqual {
  bool operator()(const AbstractBasePtrList &lhs, const AbstractBasePtrList &rhs) const;
};

std::size_t AbstractBasePtrListHash(const AbstractBasePtrList &args_spec_list);
bool AbstractBasePtrListDeepEqual(const AbstractBasePtrList &lhs, const AbstractBasePtrList &rhs);

// RowTensor
class AbstractRowTensor : public AbstractUndetermined {
 public:
  explicit AbstractRowTensor(const AbstractBasePtr &element, const BaseShapePtr &shape = std::make_shared<Shape>())
      : AbstractUndetermined(element, shape) {}
  AbstractRowTensor(const TypePtr &element_type, const ShapeVector &shape)
      : AbstractUndetermined(element_type, shape) {}
  ~AbstractRowTensor() override = default;
  MS_DECLARE_PARENT(AbstractRowTensor, AbstractUndetermined)

  const AbstractTensorPtr indices() const { return indices_; }
  void set_indices(const AbstractTensorPtr &indices) { indices_ = indices; }
  const AbstractTensorPtr values() const { return values_; }
  void set_values(const AbstractTensorPtr &values) { values_ = values; }
  const AbstractTuplePtr dense_shape() const { return dense_shape_; }
  void set_dense_shape(const AbstractTuplePtr &dense_shape) { dense_shape_ = dense_shape; }
  TypePtr BuildType() const override;
  AbstractBasePtr Clone() const override;
  AbstractBasePtr Broaden(uint8_t config = 0) const override;
  AbstractBasePtr BroadenWithShape() const;

  std::string ToString() const override;

 private:
  AbstractTensorPtr indices_;
  AbstractTensorPtr values_;
  AbstractTuplePtr dense_shape_;
};

// SparseTensor
class AbstractSparseTensor : public AbstractUndetermined {
 public:
  explicit AbstractSparseTensor(const AbstractBasePtr &element, const BaseShapePtr &shape = std::make_shared<Shape>())
      : AbstractUndetermined(element, shape) {}
  AbstractSparseTensor(const TypePtr &element_type, const ShapeVector &shape)
      : AbstractUndetermined(element_type, shape) {}
  ~AbstractSparseTensor() override = default;
  MS_DECLARE_PARENT(AbstractSparseTensor, AbstractUndetermined)

  const AbstractTensorPtr indices() const { return indices_; }
  void set_indices(const AbstractTensorPtr &indices) { indices_ = indices; }
  const AbstractTensorPtr values() const { return values_; }
  void set_values(const AbstractTensorPtr &values) { values_ = values; }
  const AbstractTuplePtr dense_shape() const { return dense_shape_; }
  void set_dense_shape(const AbstractTuplePtr &dense_shape) { dense_shape_ = dense_shape; }
  TypePtr BuildType() const override;
  AbstractBasePtr Clone() const override;
  AbstractBasePtr Broaden(uint8_t config = 0) const override;
  AbstractBasePtr BroadenWithShape() const;

  std::string ToString() const override;

 private:
  AbstractTensorPtr indices_;
  AbstractTensorPtr values_;
  AbstractTuplePtr dense_shape_;
};

class AbstractMonad : public AbstractBase {
 public:
  ~AbstractMonad() override = default;
  MS_DECLARE_PARENT(AbstractMonad, AbstractBase)

  std::size_t hash() const override { return hash_combine({tid()}); }
  TypePtr BuildType() const override { return GetTypeTrack(); }
  AbstractBasePtr Broaden(uint8_t config) const override { return AbstractBase::Broaden(config); }
  AbstractBasePtr Join(const AbstractBasePtr &other) override = 0;
  std::string ToString() const override {
    std::ostringstream buffer;
    buffer << type_name() << "(" << GetValueTrack()->ToString() << ")";
    return buffer.str();
  }

 protected:
  AbstractMonad(const ValuePtr &value, const TypePtr &type) : AbstractBase(value, type) {}
};
using AbstractMonadPtr = std::shared_ptr<AbstractMonad>;

class AbstractUMonad : public AbstractMonad {
 public:
  explicit AbstractUMonad(const ValuePtr &value = kUMonad) : AbstractMonad(value, kUMonadType) {}
  ~AbstractUMonad() override = default;
  MS_DECLARE_PARENT(AbstractUMonad, AbstractMonad)

  AbstractBasePtr Clone() const override { return std::make_shared<AbstractUMonad>(GetValueTrack()); }
  AbstractBasePtr Join(const AbstractBasePtr &other) override;
  bool operator==(const AbstractUMonad &other) const;
  bool operator==(const AbstractBase &other) const override;
};
using AbstractUMonadPtr = std::shared_ptr<AbstractUMonad>;

class AbstractIOMonad : public AbstractMonad {
 public:
  explicit AbstractIOMonad(const ValuePtr &value = kIOMonad) : AbstractMonad(value, kIOMonadType) {}
  ~AbstractIOMonad() override = default;
  MS_DECLARE_PARENT(AbstractIOMonad, AbstractMonad)

  AbstractBasePtr Clone() const override { return std::make_shared<AbstractIOMonad>(GetValueTrack()); }
  AbstractBasePtr Join(const AbstractBasePtr &other) override;
  bool operator==(const AbstractIOMonad &other) const;
  bool operator==(const AbstractBase &other) const override;
};
using AbstractIOMonadPtr = std::shared_ptr<AbstractIOMonad>;

}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_ABSTRACT_VALUE_H_
