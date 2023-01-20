/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include <cstdint>
#include <utility>
#include <vector>
#include <string>
#include <memory>

#include "utils/log_adapter.h"
#include "utils/hashing.h"
#include "utils/any.h"
#include "utils/hash_map.h"
#include "base/base.h"
#include "base/user_data.h"
#include "ir/dtype.h"
#include "ir/value.h"
#include "ir/tensor.h"
#include "ir/map_tensor.h"
#include "abstract/dshape.h"
#include "abstract/utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace abstract {
class AbstractBase;
using AbstractBasePtrList = std::vector<AbstractBasePtr>;
/// \brief The base class for abstract value of an anf node.
///
/// The abstract value is used in evaluator to express
/// the type, shape and value of an anf node.
class MS_CORE_API AbstractBase : public Base {
 public:
  using TraceNodeProvider = std::function<void(AnfNodePtr *node)>;

  /// \brief Constructor of AbstractBase.
  ///
  /// \param[in] value The real value (if any) of an anf node. Default: nullptr.
  /// \param[in] type The type of an anf node. Default: kAnyType.
  /// \param[in] shape The dimension of an anf node. Default: kNoShape.
  explicit AbstractBase(const ValuePtr &value = nullptr, const TypePtr &type = kAnyType,
                        const BaseShapePtr &shape = kNoShape)
      : value_(value), type_(type), shape_(shape) {}

  /// \brief Destructor of AbstractBase.
  ~AbstractBase() override = default;
  MS_DECLARE_PARENT(AbstractBase, Base)

  /// \brief Get the hash number of the abstract.
  ///
  /// \return The hash of the object.
  std::size_t hash() const override { return tid(); }

  /// \brief Get the formatted text to describe the abstract.
  ///
  /// \return A string.
  std::string ToString() const override;

  /// \brief Get the formatted text to describe the abstract.
  ///
  /// \return A string.
  virtual std::string ToString(bool verbose) const;

  /// \brief Overwrite the operator '==' to compare other abstract.
  ///
  /// \param[in] other The other abstract to be joined.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  virtual bool operator==(const AbstractBase &other) const;

  /// \brief Set the value for the AbstractBase.
  ///
  /// \param[in] value The value of an anf node.
  void set_value(const ValuePtr &value) {
    MS_EXCEPTION_IF_NULL(value);
    value_ = value;
  }

  /// \brief Set the type for the AbstractBase.
  ///
  /// \param[in] type The type of an anf node.
  void set_type(const TypePtr &type) {
    MS_EXCEPTION_IF_NULL(type);
    type_ = type;
  }

  /// \brief Set the shape for the AbstractBase.
  ///
  /// \param[in] shape The shape of an anf node.
  virtual void set_shape(const BaseShapePtr &shape) {
    MS_EXCEPTION_IF_NULL(shape);
    shape_ = shape;
  }

  /// \brief Set the value description for the AbstractBase.
  ///
  /// \param[in] desc The description of value.
  void set_value_desc(const std::string &desc) { value_desc_ = desc; }

  /// \brief Get the value description.
  ///
  /// \return A string of the value description.
  const std::string &value_desc() const { return value_desc_; }

  /// \brief Get the abstract value, which is tracked.
  ///
  /// \return A pointer to the Value.
  const ValuePtr &GetValueTrack() const { return value_; }

  /// \brief Get the abstract type, which is tracked.
  ///
  /// \return A pointer to the Type.
  const TypePtr &GetTypeTrack() const { return type_; }

  /// \brief Get the abstract shape, which is tracked.
  ///
  /// \return A pointer to the BaseShape.
  const BaseShapePtr &GetShapeTrack() const { return shape_; }

  /// \brief Try to build a real value from an abstract value.
  ///
  /// \note If the value cannot be built, a default value (AnyValue) is returned.
  ///
  /// \return A pointer to the Value.
  ValuePtr BuildValue() const;

  /// \brief Build the type of the abstract.
  ///
  /// \note Use this function to get the actual type, while track type is not enough accurate.
  ///
  /// \return A pointer to the Type.
  virtual TypePtr BuildType() const = 0;

  /// \brief Build the shape of the abstract.
  ///
  /// \note Use this function to get the actual shape, while track shape is not enough accurate.
  ///
  /// \return A pointer to the BaseShape.
  virtual BaseShapePtr BuildShape() const { return kNoShape; }

  /// \brief Clone an abstract from the abstract.
  ///
  /// \return A pointer to the cloned abstract.
  virtual AbstractBasePtr Clone() const = 0;

  /// \brief Set the function, which prints the debug info.
  ///
  /// \param[in] trace_node_provider The function.
  static void set_trace_node_provider(const TraceNodeProvider &trace_node_provider) {
    trace_node_provider_ = trace_node_provider;
  }

  static TraceNodeProvider trace_node_provider_;

  /// \brief Broaden the abstract. It will upgrade the abstract to a higher level.
  ///
  /// \return A pointer to the broadened abstract.
  virtual AbstractBasePtr Broaden() const;

  /// \brief Combine two abstracts. If two abstracts are different, it will broaden the abstract value.
  ///
  /// \param[in] other The other abstract to be joined.
  ///
  /// \return A pointer to the combined abstract.
  virtual AbstractBasePtr Join(const AbstractBasePtr &other) { return shared_from_base<AbstractBase>(); }
  bool IsBroaden() const { return value_ == kAnyValue; }

  /// \brief Write the abstract's string to the std::ostream.
  ///
  /// \param[in] os A std::ostream.
  /// \param[in] a An abstract.
  ///
  /// \return A std::ostream.
#ifndef _MSC_VER
  friend std::ostream &operator<<(std::ostream &os, const std::shared_ptr<AbstractBase> &a) {
    os << a->ToString();
    return os;
  }
#endif
  /// \brief Broaden abstract with constraints.
  ///
  /// \return A pointer to the broadened abstract.
  virtual AbstractBasePtr PartialBroaden() const;

  /// \brief Set user data.
  ///
  /// \param[in] key The key of user data.
  /// \param[in] value The value of user data.
  template <typename T>
  void set_user_data(const std::string &key, const std::shared_ptr<T> &value) {
    user_data_.set<T>(key, value);
  }

  /// \brief Set user data.
  ///
  /// \param[in] value The value of user data.
  template <typename T>
  void set_user_data(const std::shared_ptr<T> &value) {
    user_data_.set<T>(T::key, value);
  }

  /// \brief Get user data.
  ///
  /// \param[in] key The key of user data.
  /// \return Pointer to user data.
  template <typename T>
  std::shared_ptr<T> user_data(const std::string &key) const {
    return user_data_.get<T>(key);
  }

  /// \brief Set user data.
  ///
  /// \return Pointer to user data.
  template <typename T>
  std::shared_ptr<T> user_data() const {
    return user_data_.get<T>(T::key);
  }

  /// \brief Check whether there is corresponding user data by the given key.
  ///
  /// \param[in] key The key of user data.
  /// \return True if it exists, otherwise false.
  bool has_user_data(const std::string &key) const { return user_data_.has(key); }

  /// \brief Check if there is user data.
  ///
  /// \return True if it exists, otherwise false.
  template <typename T>
  bool has_user_data() const {
    return user_data_.has(T::key);
  }

  /// \brief Clone user data.
  ///
  /// \param[in] abstract Abstract used to copy user data.
  void CloneUserData(const AbstractBasePtr &abstract) { user_data_ = abstract->user_data_; }

  /// \brief Process the abstract with InterpretedObject.
  using InterpretBoolChecker = std::pair<bool, bool> (*)(const AbstractBasePtr &cond);
  static inline InterpretBoolChecker interpret_bool_checker_ = nullptr;
  static void set_interpret_bool_checker(InterpretBoolChecker checker) { interpret_bool_checker_ = checker; }
  static inline InterpretBoolChecker interpret_bool_checker() { return interpret_bool_checker_; }

  std::string name() const { return name_; }

  void set_name(const std::string &name) { name_ = name; }

 protected:
  /// \brief Build a value when value is not set.
  ///
  /// \return A pointer to the Value.
  virtual ValuePtr RealBuildValue() const { return kAnyValue; }
  std::string name_;

 private:
  ValuePtr value_;
  TypePtr type_;
  BaseShapePtr shape_;
  std::string value_desc_;  // store initial value description for error report
  UserData user_data_;
};

/// \brief Class AbstractScalar describes a scalar's type and value.
class MS_CORE_API AbstractScalar final : public AbstractBase {
 public:
  /// \brief Constructor of AbstractScalar.
  AbstractScalar() : AbstractBase(kAnyValue, kAnyType) {}

  /// \brief Constructor of AbstractScalar.
  ///
  /// \param[in] value The real value of an anf node.
  /// \param[in] type The type of an anf node.
  AbstractScalar(const ValuePtr &value, const TypePtr &type) : AbstractBase(value, type) {}

  /// \brief Constructor of AbstractScalar.
  ///
  /// \param[in] value The real value of an anf node.
  explicit AbstractScalar(const ValuePtr &value) : AbstractBase(value, value->type()) {}

  /// \brief Constructor of AbstractScalar, inited with an int number.
  ///
  /// \param[in] value An int number.
  explicit AbstractScalar(int value) : AbstractBase(MakeValue(value), kInt32) {}

  /// \brief Constructor of AbstractScalar, inited with an int64 number.
  ///
  /// \param[in] value An int64 number.
  explicit AbstractScalar(int64_t value) : AbstractBase(MakeValue(value), kInt64) {}

  /// \brief Constructor of AbstractScalar, inited with a float number.
  ///
  /// \param[in] value A float number.
  explicit AbstractScalar(float value) : AbstractBase(MakeValue(value), kFloat32) {}

  /// \brief Constructor of AbstractScalar, inited with a double number.
  ///
  /// \param[in] value A double number.
  explicit AbstractScalar(double value) : AbstractBase(MakeValue(value), kFloat64) {}

  /// \brief Constructor of AbstractScalar, inited with a bool.
  ///
  /// \param[in] value A boolean variable.
  explicit AbstractScalar(bool value) : AbstractBase(MakeValue(value), kBool) {}

  /// \brief Constructor of AbstractScalar, inited with a string.
  ///
  /// \param[in] value A string.
  explicit AbstractScalar(const std::string &value) : AbstractBase(MakeValue(value), kString) {}

  /// \brief Constructor of AbstractScalar, inited with a type.
  ///
  /// \param[in] type The type.
  explicit AbstractScalar(const TypePtr &type) : AbstractBase(kAnyValue, type) {}

  /// \brief Destructor of AbstractScalar.
  ~AbstractScalar() override = default;
  MS_DECLARE_PARENT(AbstractScalar, AbstractBase)

  /// \brief Set the flag 'is_variable_' for scalar.
  ///
  /// \param[in] is_variable Boolean value for flag 'is_variable_'.
  void set_is_variable(bool is_variable) { is_variable_ = is_variable; }

  std::size_t hash() const override { return hash_combine({tid(), GetValueTrack()->hash(), GetTypeTrack()->hash()}); }

  TypePtr BuildType() const override { return GetTypeTrack(); }

  AbstractBasePtr Clone() const override {
    auto abs = std::make_shared<AbstractScalar>(GetValueTrack(), GetTypeTrack()->Clone());
    abs->set_is_variable(is_variable_);
    return abs;
  }

  AbstractBasePtr Broaden() const override;

  AbstractBasePtr Join(const AbstractBasePtr &other) override;

 private:
  bool is_variable_{false};
};
using AbstractScalarPtr = std::shared_ptr<AbstractScalar>;

/// \brief Class AbstractType describes the abstract value from a Typeof node.
class MS_CORE_API AbstractType final : public AbstractBase {
 public:
  /// \brief Constructor of AbstractType.
  ///
  /// \param[in] type The type of an anf node.
  explicit AbstractType(const TypePtr &type) : AbstractBase(type, kTypeType) {
    if (type == nullptr) {
      MS_LOG(EXCEPTION) << "type is nullptr";
    }
  }

  /// \brief Destructor of AbstractType.
  ~AbstractType() override = default;
  MS_DECLARE_PARENT(AbstractType, AbstractBase)

  std::string ToString() const override;

  bool operator==(const AbstractBase &other) const override;

  TypePtr BuildType() const override { return std::make_shared<TypeType>(); }

  AbstractBasePtr Clone() const override;

  AbstractBasePtr Broaden() const override { return Clone(); }
};
using AbstractTypePtr = std::shared_ptr<AbstractType>;

/// \brief Class AbstractError describes the abstract value from an error.
class MS_CORE_API AbstractError final : public AbstractBase {
 public:
  /// \brief Constructor of AbstractError.
  ///
  /// \param[in] err the error string.
  /// \param[in] node the binding anf node.
  AbstractError(const ErrorValuePtr &err, const AnfNodePtr &node) : AbstractBase(err), node_(node) {
    if (err == nullptr || node == nullptr) {
      MS_LOG(EXCEPTION) << "err or node is nullptr";
    }
  }

  /// \brief Destructor of AbstractError.
  ~AbstractError() override = default;
  MS_DECLARE_PARENT(AbstractError, AbstractBase)

  TypePtr BuildType() const override { return std::make_shared<Problem>(); }

  AbstractBasePtr Broaden() const override { return Clone(); }

  AbstractBasePtr Clone() const override {
    return std::make_shared<AbstractError>(GetValueTrack()->cast<ErrorValuePtr>(), node_);
  }

  std::string ToString() const override;

 private:
  // Origin node been specialized to AbstractError, for debug purpose only.
  const AnfNodePtr node_;
};

/// \brief Class AbstractScript describes the script node's type, shape and value.
class MS_CORE_API AbstractScript final : public AbstractBase {
 public:
  /// \brief Constructor of AbstractScript.
  AbstractScript() : AbstractBase(kAnyValue, kAnyType) {}

  /// \brief Constructor of AbstractScript.
  ///
  /// \param[in] value The real value of an anf node.
  /// \param[in] type The type of an anf node.
  AbstractScript(const ValuePtr &value, const TypePtr &type) : AbstractBase(value, type) {}

  /// \brief Constructor of AbstractScript.
  ///
  /// \param[in] value The real value to be set.
  explicit AbstractScript(const ValuePtr &value) : AbstractBase(value, kString) {}

  /// \brief Destructor of AbstractScript.
  ~AbstractScript() override = default;
  MS_DECLARE_PARENT(AbstractScript, AbstractBase)

  std::size_t hash() const override { return hash_combine({tid(), GetValueTrack()->hash(), GetTypeTrack()->hash()}); }

  TypePtr BuildType() const override { return GetTypeTrack(); }

  AbstractBasePtr Clone() const override {
    return std::make_shared<AbstractScript>(GetValueTrack(), GetTypeTrack()->Clone());
  }

  AbstractBasePtr Broaden() const override { return Clone(); }
};
using AbstractScriptPtr = std::shared_ptr<AbstractScript>;

class Evaluator;
using EvaluatorPtr = std::shared_ptr<Evaluator>;
class AnalysisEngine;
using AnalysisEnginePtr = std::shared_ptr<AnalysisEngine>;

class AbstractFunction;
using AbstractFunctionPtr = std::shared_ptr<AbstractFunction>;
class AbstractFuncAtom;
using AbstractFuncAtomPtr = std::shared_ptr<AbstractFuncAtom>;
using AbstractFuncAtomPtrList = std::vector<AbstractFuncAtomPtr>;

/// \brief The base class for the abstract value of the function node.
class MS_CORE_API AbstractFunction : public AbstractBase {
 public:
  /// \brief Constructor of AbstractFunction.
  AbstractFunction() = default;
  /// \brief Destructor of AbstractFunction.
  ~AbstractFunction() override = default;
  MS_DECLARE_PARENT(AbstractFunction, AbstractBase)

  /// \brief Get the unique AbstractFunction.
  ///
  /// If there is exactly one possible function, return it. Otherwise, raise an Exception.
  /// Caller should ensure the uniqueness.
  ///
  /// \return A pointer to AbstractFunction.
  virtual AbstractFunctionPtr GetUnique() = 0;

  TypePtr BuildType() const override { return std::make_shared<Function>(); }

  AbstractBasePtr Clone() const override { return Copy(); }

  AbstractBasePtr Broaden() const override {
    return const_cast<AbstractFunction *>(this)->shared_from_base<AbstractFunction>();
  }

  /// \brief Copy an AbstractFunction.
  ///
  /// \return A pointer to the copied abstract.
  virtual AbstractFunctionPtr Copy() const = 0;

  /// \brief Combine two abstracts. If two abstracts are different, it will broaden the abstract value.
  ///
  /// \param[in] other The other abstract to be joined.
  ///
  /// \return A pointer to the combined abstract.
  AbstractBasePtr Join(const AbstractBasePtr &other) final;

  /// \brief Combine two abstracts. If two abstracts are different, it will broaden the abstract value.
  ///
  /// \param[in] other The other abstract to be joined.
  ///
  /// \return A pointer to the combined abstract.
  virtual AbstractFunctionPtr Join(const AbstractFunctionPtr &other) = 0;

  /// \brief Handle something with the outer visit function.
  virtual void Visit(std::function<void(const AbstractFuncAtomPtr &)>) const = 0;

  /// \brief Overwrite the operator '==' to compare other abstract.
  ///
  /// \param[in] other The other abstract to be joined.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractBase &other) const final;

  /// \brief Overwrite the operator '==' to compare other AbstractFunction.
  ///
  /// \param[in] other The other instance of AbstractFunction.
  ///
  /// \return A boolean, which indicates whether the other AbstractFunction is same.
  virtual bool operator==(const AbstractFunction &other) const = 0;

  /// \brief Make a AbstractFuncUnion from a list of AbstractFuncAtom.
  ///
  /// \param[in] func_list A list of AbstractFuncAtomPtrList.
  /// \return A point to the AbstractFunction.
  static AbstractFunctionPtr MakeAbstractFunction(const AbstractFuncAtomPtrList &func_list);

  /// \brief Get the tracking id as the memory address of the anf node.
  ///
  /// \return The memory address of to the anf node.
  virtual std::uintptr_t tracking_id() const { return 0; }

  /// \brief Copy an AbstractFunction without copying tracking id.
  ///
  /// \return A pointer to the copied abstract.
  virtual AbstractFunctionPtr CopyWithoutTrackingId() const { return Copy(); }

  /// \brief Get the context which manages the abstract.
  ///
  /// \return A point to the context.
  virtual AnalysisContextPtr context() const { return nullptr; }

  static std::uintptr_t ToTrackingId(const AnfNodePtr &node) { return reinterpret_cast<std::uintptr_t>(node.get()); }
};

using AbstractFunctionPtrList = std::vector<AbstractFunctionPtr>;

/// \brief Class AbstractKeywordArg describes an abstract value from a key-value node.
///
/// Represents a key-value pair used in function's parameters.
class MS_CORE_API AbstractKeywordArg final : public AbstractBase {
 public:
  /// \brief Constructor of AbstractKeywordArg.
  ///
  /// \param[in] key The key name of the key-value pair.
  /// \param[in] argument The key value of the key-value pair.
  AbstractKeywordArg(const std::string &key, const AbstractBasePtr &argument) : arg_name_(key), arg_value_(argument) {}

  /// \brief Destructor of AbstractKeywordArg.
  ~AbstractKeywordArg() override = default;
  MS_DECLARE_PARENT(AbstractKeywordArg, AbstractBase)

  TypePtr BuildType() const override;

  AbstractBasePtr Clone() const override;

  AbstractBasePtr Broaden() const override;

  std::size_t hash() const override;

  /// \brief Overwrite the operator '==' to compare other key-value abstract.
  ///
  /// \param[in] other The other abstract to be joined.
  ///
  /// \return  A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractKeywordArg &other) const;

  bool operator==(const AbstractBase &other) const override;

  /// \brief Get the key name of the key-value pair.
  ///
  /// \return A string.
  std::string get_key() const { return arg_name_; }

  /// \brief Get the key value of the key-value pair.
  ///
  /// \return A point to the abstract.
  AbstractBasePtr get_arg() const { return arg_value_; }

  std::string ToString() const override;

 protected:
  ValuePtr RealBuildValue() const override;

 private:
  std::string arg_name_;
  AbstractBasePtr arg_value_;
};
using AbstractKeywordArgPtr = std::shared_ptr<AbstractKeywordArg>;

/// \brief Class AbstractUndetermined describes the abstract if anf node has unknown shape, type or value.
class MS_CORE_API AbstractUndetermined : public AbstractBase {
 public:
  /// \brief Constructor of AbstractUndetermined.
  ///
  /// Shape and type are all unknown.
  AbstractUndetermined() : AbstractBase(kAnyValue) {}

  /// \brief Constructor of AbstractUndetermined.
  ///
  /// Only element, value and shape track are valid member, type track are unknown.
  ///
  /// \param[in] element The abstract which is undetermined.
  /// \param[in] shape The dimension of value.
  explicit AbstractUndetermined(const AbstractBasePtr &element, const BaseShapePtr &shape = std::make_shared<Shape>())
      : AbstractBase(kAnyValue), element_(element) {
    if (element == nullptr) {
      MS_LOG(EXCEPTION) << "element is nullptr";
    }
    if (element->isa<AbstractUndetermined>()) {
      MS_LOG(EXCEPTION) << "element type error";
    }
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->isa<NoShape>()) {
      MS_LOG(EXCEPTION) << "AbstractUndetermined can't set shape as NoShape.";
    }
    AbstractBase::set_shape(shape);
  }

  /// \brief Constructor of AbstractUndetermined.
  ///
  /// \param[in] element_type A type of the undetermined abstract.
  /// \param[in] shape A vector of shape.
  AbstractUndetermined(const TypePtr &element_type, const ShapeVector &shape)
      : AbstractBase(kAnyValue), element_(std::make_shared<AbstractScalar>(kAnyValue, element_type)) {
    if (element_type == nullptr) {
      MS_LOG(EXCEPTION) << "element_type is nullptr";
    }
    AbstractBase::set_shape(std::make_shared<Shape>(shape));
  }

  /// \brief Constructor of AbstractUndetermined.
  ///
  /// \param[in] element_type A type of the undetermined abstract.
  /// \param[in] shape A shape of the undetermined abstract.
  explicit AbstractUndetermined(const TypePtr &element_type, const BaseShapePtr &shape = std::make_shared<Shape>())
      : AbstractBase(kAnyValue), element_(std::make_shared<AbstractScalar>(kAnyValue, element_type)) {
    if (element_type == nullptr) {
      MS_LOG(EXCEPTION) << "element_type is nullptr";
    }
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->isa<NoShape>()) {
      MS_LOG(EXCEPTION) << "AbstractUndetermined can't set shape as NoShape.";
    }
    AbstractBase::set_shape(shape);
  }

  /// \brief Destructor of AbstractUndetermined.
  ~AbstractUndetermined() override = default;
  MS_DECLARE_PARENT(AbstractUndetermined, AbstractBase)

  TypePtr BuildType() const override { return std::make_shared<UndeterminedType>(); }

  AbstractBasePtr Clone() const override { return std::make_shared<AbstractUndetermined>(); }

  /// \brief Get the element, which is the tracked undetermined abstract.
  ///
  /// \return A pointer to the bind abstract, which is undetermined.
  AbstractBasePtr element() const { return element_; }

  /// \brief Get the shape of the undetermined abstract.
  ///
  /// \return A pointer to the shape.
  ShapePtr shape() const;

  void set_shape(const BaseShapePtr &shape) override;

 protected:
  AbstractBasePtr element_;
};

/// \brief Class AbstractTensor describes a tensor's type, shape and value.
class MS_CORE_API AbstractTensor : public AbstractUndetermined {
 public:
  /// \brief Constructor of AbstractTensor.
  ///
  /// \param[in] element The abstract to be wrapper as a abstract tensor.
  /// \param[in] shape The dimension of abstract tensor.
  explicit AbstractTensor(const AbstractBasePtr &element, const BaseShapePtr &shape = std::make_shared<Shape>())
      : AbstractUndetermined(element, shape) {}

  /// \brief Constructor of AbstractTensor.
  ///
  /// \param[in] element_type The type of abstract tensor.
  /// \param[in] shape A vector of the tensor's shape.
  AbstractTensor(const TypePtr &element_type, const ShapeVector &shape) : AbstractUndetermined(element_type, shape) {}

  /// \brief Constructor of AbstractTensor.
  ///
  /// \param[in] tensor The tensor to be abstracted.
  explicit AbstractTensor(const tensor::TensorPtr &tensor) : AbstractUndetermined(tensor->Dtype(), tensor->shape()) {}

  /// \brief Constructor of AbstractTensor.
  ///
  /// \param[in] element_type The type of a tensor.
  /// \param[in] shape The dimension of a tensor.
  explicit AbstractTensor(const TypePtr &element_type, const BaseShapePtr &shape = std::make_shared<Shape>())
      : AbstractUndetermined(element_type, shape) {}

  /// \brief Destructor of AbstractTensor.
  ~AbstractTensor() override = default;
  MS_DECLARE_PARENT(AbstractTensor, AbstractUndetermined)

  /// \brief Set min value and max value.
  ///
  /// \param[in] min_value The min value of tensor.
  /// \param[in] max_value The max value of tensor.
  void set_value_range(const ValuePtr &min_value, const ValuePtr &max_value) {
    min_value_ = min_value;
    max_value_ = max_value;
  }

  /// \brief Get the min value.
  ///
  /// \return A pointer to a value.
  const ValuePtr &get_min_value() const { return min_value_; }

  /// \brief Get the max value.
  ///
  /// \return A pointer to a value.
  const ValuePtr &get_max_value() const { return max_value_; }

  /// \brief Set shape value
  ///
  /// \param[in] shape_value The shape value of tensor.
  void set_shape_value(const ValuePtr &shape_value) { shape_value_ = shape_value; }

  /// \brief Get the shape value.
  ///
  /// \return A pointer to a value.
  const ValuePtr &get_shape_value() const { return shape_value_; }

  TypePtr BuildType() const override;

  BaseShapePtr BuildShape() const override;

  AbstractBasePtr Clone() const override;

  AbstractBasePtr Broaden() const override;

  /// \brief Broaden the abstract. It will upgrade the abstract to a higher level.
  ///
  /// \note The shape will be remained.
  ///
  /// \return A pointer to the broadened abstract.
  AbstractBasePtr BroadenWithShape() const;

  AbstractBasePtr Join(const AbstractBasePtr &other) override;

  /// \brief Overwrite the operator '==' to compare other abstract tensor.
  ///
  /// \param[in] other The other instance of AbstractTensor.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  virtual bool operator==(const AbstractTensor &other) const;

  bool operator==(const AbstractBase &other) const override;

  std::string ToString() const override;

  std::size_t hash() const override {
    // We have to exclude value pointer from hash, because CSE (Common Subexpression Elimination)
    // will use this hash to find duplicate ValueNodes that Tensor values are equal.
    auto hash_sum = hash_combine(tid(), element_->hash());
    const auto &shape = GetShapeTrack();
    if (shape != nullptr) {
      hash_sum = hash_combine(hash_sum, shape->hash());
    }
    return hash_sum;
  }

  AbstractBasePtr PartialBroaden() const override;

  bool is_adapter() const { return is_adapter_; }
  void set_is_adapter(bool is_adapter) { is_adapter_ = is_adapter; }

 protected:
  bool equal_to(const AbstractTensor &other) const;
  ValuePtr min_value_ = nullptr;
  ValuePtr max_value_ = nullptr;
  ValuePtr shape_value_ = nullptr;
  bool is_adapter_ = false;
};
using AbstractTensorPtr = std::shared_ptr<AbstractTensor>;
using AbstractTensorPtrList = std::vector<AbstractTensorPtr>;

/// \brief Class AbstractSequence describes the abstract value of a tuple or list.
class MS_CORE_API AbstractSequence : public AbstractBase {
 public:
  /// \brief Constructor of AbstractSequence.
  ///
  /// \param[in] elements A list of abstracts.
  /// \param[in] sequence_nodes The nodes of tuple/list, usually are MakeTuple/MakeList CNodes or tuple/list ValueNodes.
  explicit AbstractSequence(AbstractBasePtrList &&elements, const std::shared_ptr<AnfNodeWeakPtrList> &sequence_nodes);

  /// \brief Constructor of AbstractSequence.
  ///
  /// \param[in] elements A list of abstracts.
  /// \param[in] sequence_nodes The nodes of tuple/list, usually are MakeTuple/MakeList CNodes or tuple/list ValueNodes.
  explicit AbstractSequence(const AbstractBasePtrList &elements,
                            const std::shared_ptr<AnfNodeWeakPtrList> &sequence_nodes);

  /// \brief Destructor of AbstractSequence.
  ~AbstractSequence() override = default;
  MS_DECLARE_PARENT(AbstractSequence, AbstractBase)

  /// \brief Get the all of types.
  ///
  /// \return A vector of types.
  TypePtrList ElementsType() const;

  /// \brief Get the all of shapes.
  ///
  /// \return A vector of shapes.
  BaseShapePtrList ElementsShape() const;

  /// \brief Clone all of the abstracts.
  ///
  /// \return A vector of the cloned abstracts.
  AbstractBasePtrList ElementsClone() const;

  /// \brief Broaden the list of abstracts.
  ///
  /// \return A vector of the broadened abstracts.
  AbstractBasePtrList ElementsBroaden() const;

  /// \brief Broaden abstract with constraints, only when cond_func is true.
  ///
  /// \return A pointer to the broadened abstract.
  AbstractBasePtrList ElementsPartialBroaden() const;

  /// \brief Get real value by specific template.
  ///
  /// \tparam T the class type of value.
  /// \return A point to value.
  template <typename T>
  ValuePtr ElementsBuildValue() const;

  /// \brief Combine other abstract to the sequence of abstracts.
  ///
  /// \tparam T param other's class type.
  /// \param[in] other The other abstract to be joined.
  /// \return A pointer to the combined abstract.
  template <typename T>
  AbstractBasePtr ElementsJoin(const AbstractBasePtr &other);

  /// \brief Combine other sequence nodes with this one.
  ///
  /// \param[in] other The other abstract to be joined.
  /// \return A sequence nodes list combined.
  AnfNodeWeakPtrList SequenceNodesJoin(const AbstractBasePtr &other);

  /// \brief Get the size of the stored elements.
  ///
  /// \return A size_t.
  std::size_t size() const;

  /// \brief Get the size of the stored elements.
  ///
  /// \return A size_t.
  bool empty() const;

  /// \brief Get the stored elements.
  ///
  /// \return A vector of elements.
  const AbstractBasePtrList &elements() const { return elements_; }

  /// \brief Purify the elements list, and clean unused elements.
  ///
  /// \return A boolean, which indicates whether success.
  bool PurifyElements();

  /// \brief Get the sequence nodes where these 'AbstractSequence' evaluated from.
  ///
  /// \return The nodes of tuple/list, usually are MakeTuple/MakeList CNodes or tuple/list ValueNodes.
  const std::shared_ptr<AnfNodeWeakPtrList> &sequence_nodes() const { return sequence_nodes_; }

  /// \brief Set the sequence nodes where these 'AbstractSequence' evaluated from.
  ///
  /// \param[in] sequence_nodes The nodes of tuple/list, usually are MakeTuple/MakeList CNodes or tuple/list ValueNodes.
  void set_sequence_nodes(const std::shared_ptr<AnfNodeWeakPtrList> &sequence_nodes) {
    sequence_nodes_ = sequence_nodes;
  }

  /// \brief Insert a node into the sequence nodes.
  ///
  /// \param[in] sequence_node The node to intert into sequence nodes.
  void InsertSequenceNode(const AnfNodePtr &sequence_node);

  /// \brief Insert nodes into the sequence nodes.
  ///
  /// \param[in] sequence_nodes The nodes to intert into sequence nodes.
  void InsertSequenceNodes(const AnfNodeWeakPtrList &sequence_nodes);

  /// \brief Update the sequence nodes.
  ///
  /// \param[in] old_sequence_node The old node in sequence nodes.
  /// \param[in] new_sequence_node The new node to replace old node in sequence nodes.
  void UpdateSequenceNode(const AnfNodePtr &old_sequence_node, const AnfNodePtr &new_sequence_node);

  std::size_t hash() const override;

  std::string ToStringInternal() const;
  std::string ToString() const override;
  std::string ToString(bool verbose) const override;

  /// \brief Overwrite the operator '[]' to get an element.
  ///
  /// \param[in] dim The index.
  /// \return A pointer to the abstract.
  const AbstractBasePtr operator[](const std::size_t &dim) const;

  /// \brief Overwrite the operator '==' to compare other abstract sequence.
  ///
  /// \param[in] other The other instance of AbstractSequence.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractBase &other) const override;

  /// \brief Indicate whether the sequence is dynamic length.
  ///
  /// \return Boolean value indicates whether the sequence is dynamic length.
  bool dynamic_len() const { return dynamic_len_; }

  /// \brief Set the sequence to be dynamic length or not.
  ///
  /// \param[in] dynamic_len Boolean value to decide whether the sequence is dynamic length.
  void set_dynamic_len(bool dynamic_len);

  /// \brief Return the abstract of element for variable len sequence.
  ///
  /// \return Abstract for element for variable len sequence.
  AbstractBasePtr dynamic_len_element_abs() const { return dynamic_len_element_abs_; }

  /// \brief Set the abstract of element for variable len sequence.
  ///
  /// \param[in] dynamic_len_element_abs Abstract of element for variable len sequence.
  void set_dynamic_len_element_abs(const AbstractBasePtr &dynamic_len_element_abs);

  /// \brief Check and convert the sequence to dynamic length sequence.
  void CheckAndConvertToDynamicLenSequence();

 protected:
  AbstractBasePtrList elements_;
  // Since there're not too many nodes, we just use vector here.
  std::shared_ptr<AnfNodeWeakPtrList> sequence_nodes_;
  // Dynamic len sequence related.
  bool dynamic_len_ = false;
  AbstractBasePtr dynamic_len_element_abs_ = nullptr;

  template <typename T>
  AbstractBasePtr DynamicLenSequenceJoin(const AbstractBasePtr &other);
};
using AbstractSequencePtr = std::shared_ptr<AbstractSequence>;

/// \brief Class AbstractTuple describes a tuple.
class MS_CORE_API AbstractTuple : public AbstractSequence {
 public:
  /// \brief Constructor of AbstractTuple.
  ///
  /// \param[in] elements A list of abstracts.
  /// \param[in] tuple_nodes The nodes of tuple, usually are MakeTuple CNodes or tuple ValueNodes.
  explicit AbstractTuple(AbstractBasePtrList &&elements,
                         const std::shared_ptr<AnfNodeWeakPtrList> &tuple_nodes = nullptr)
      : AbstractSequence(std::move(elements), tuple_nodes) {}

  /// \brief Constructor of AbstractTuple.
  ///
  /// \param[in] elements A list of abstracts.
  /// \param[in] tuple_nodes The nodes of tuple, usually are MakeTuple CNodes or tuple ValueNodes.
  explicit AbstractTuple(const AbstractBasePtrList &elements,
                         const std::shared_ptr<AnfNodeWeakPtrList> &tuple_nodes = nullptr)
      : AbstractSequence(elements, tuple_nodes) {}

  /// \brief Destructor of AbstractTuple.
  ~AbstractTuple() override = default;
  MS_DECLARE_PARENT(AbstractTuple, AbstractSequence)

  /// \brief Set the shape for the AbstractTuple, only use for dynamic shape.
  ///
  /// \param[in] shape The shape that will be set to the AbstractTuple.
  void set_shape(const BaseShapePtr &shape) override;

  TypePtr BuildType() const override;

  BaseShapePtr BuildShape() const override;

  AbstractBasePtr Clone() const override;

  AbstractBasePtr Broaden() const override;

  AbstractBasePtr PartialBroaden() const override;

  AbstractBasePtr Join(const AbstractBasePtr &other) override;

  /// \brief Check whether all elements of the tuple are tensors.
  ///
  /// \return Whether all elements of the tuple are tensors.
  bool ContainsAllBroadenTensors() const;

  /// \brief Check whether all elements of the tuple are constants.
  ///
  /// \return Whether all elements of the tuple are constants.
  bool ContainsAllConstants() const;

  /// \brief Overwrite the operator '==' to compare other abstract tuple.
  ///
  /// \param[in] other The other instance of AbstractTuple.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractBase &other) const override;

 protected:
  ValuePtr RealBuildValue() const override;
};
using AbstractTuplePtr = std::shared_ptr<AbstractTuple>;

/// \brief Class AbstractList describes a list.
class MS_CORE_API AbstractList final : public AbstractSequence {
 public:
  /// \brief Constructor of AbstractList.
  ///
  /// \param[in] elements A list of abstracts.
  /// \param[in] list_nodes The nodes of list, usually are MakeList CNodes or list ValueNodes.
  explicit AbstractList(AbstractBasePtrList &&elements, const std::shared_ptr<AnfNodeWeakPtrList> &list_nodes = nullptr)
      : AbstractSequence(std::move(elements), list_nodes) {}

  /// \brief Constructor of AbstractList.
  ///
  /// \param[in] elements A list of abstracts.
  /// \param[in] list_nodes The nodes of list, usually are MakeList CNodes or list ValueNodes.
  explicit AbstractList(const AbstractBasePtrList &elements,
                        const std::shared_ptr<AnfNodeWeakPtrList> &list_nodes = nullptr)
      : AbstractSequence(elements, list_nodes) {}

  /// \brief Destructor of AbstractList.
  ~AbstractList() override = default;
  MS_DECLARE_PARENT(AbstractList, AbstractSequence)

  TypePtr BuildType() const override;

  BaseShapePtr BuildShape() const override;

  AbstractBasePtr Clone() const override;

  AbstractBasePtr Broaden() const override;

  AbstractBasePtr PartialBroaden() const override;

  AbstractBasePtr Join(const AbstractBasePtr &other) override;

  /// \brief Overwrite the operator '==' to compare other abstract list.
  ///
  /// \param[in] other The other instance of AbstractList.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractBase &other) const override;

 protected:
  ValuePtr RealBuildValue() const override;
};
using AbstractListPtr = std::shared_ptr<AbstractList>;

/// \brief Class AbstractDictionary describes a dictionary node's abstract value.
class MS_CORE_API AbstractDictionary final : public AbstractBase {
 public:
  /// \brief Constructor of AbstractDictionary.
  ///
  /// \param[in] key_values The vector of AbstractElementPair.
  explicit AbstractDictionary(const std::vector<AbstractElementPair> &key_values) : key_values_(key_values) {}

  /// \brief Destructor of AbstractDictionary.
  ~AbstractDictionary() override = default;
  MS_DECLARE_PARENT(AbstractDictionary, AbstractBase)

  TypePtr BuildType() const override;

  bool operator==(const AbstractBase &other) const override;

  AbstractBasePtr Clone() const override;

  AbstractBasePtr Broaden() const override;

  std::string ToString() const override;

  std::size_t hash() const override;

  /// \brief Get the size of key values.
  ///
  /// \return A size_t.
  std::size_t size() const { return key_values_.size(); }

  /// \brief Get the key values.
  ///
  /// \return A vector of AbstractElementPair.
  const std::vector<AbstractElementPair> &elements() const { return key_values_; }

 protected:
  ValuePtr RealBuildValue() const override;
  std::vector<AbstractElementPair> key_values_;
};
using AbstractDictionaryPtr = std::shared_ptr<AbstractDictionary>;

/// \brief Class AbstractSlice describes a slice node's abstract value.
class MS_CORE_API AbstractSlice final : public AbstractBase {
 public:
  /// \brief Constructor of AbstractSlice.
  ///
  /// \param[in] start The start index of slice.
  /// \param[in] stop The stop index of slice.
  /// \param[in] step The step size of slice.
  AbstractSlice(const AbstractBasePtr &start, const AbstractBasePtr &stop, const AbstractBasePtr &step)
      : start_(start), stop_(stop), step_(step) {}

  /// \brief Destructor of AbstractSlice.
  ~AbstractSlice() override = default;
  MS_DECLARE_PARENT(AbstractSlice, AbstractBase)

  TypePtr BuildType() const override;

  /// \brief Overwrite the operator '==' to compare other abstract lice.
  ///
  /// \param[in] other The other instance of AbstractSlice.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractBase &other) const override;

  AbstractBasePtr Clone() const override;

  AbstractBasePtr Broaden() const override;

  std::string ToString() const override;

  std::size_t hash() const override;

  /// \brief Get the start index of slice.
  ///
  /// \return A point to the abstract of start index.
  AbstractBasePtr start() const { return start_; }

  /// \brief Get the stop index of slice.
  ///
  /// \return A point to the abstract of stop index.
  AbstractBasePtr stop() const { return stop_; }

  /// \brief Get the step size of slice.
  ///
  /// \return A point to the abstract of step number.
  AbstractBasePtr step() const { return step_; }

 protected:
  ValuePtr RealBuildValue() const override;

 private:
  AbstractBasePtr start_;
  AbstractBasePtr stop_;
  AbstractBasePtr step_;
};
using AbstractSlicePtr = std::shared_ptr<AbstractSlice>;

/// \brief Class AbstractJTagged describes a J node's abstract value.
class MS_CORE_API AbstractJTagged final : public AbstractBase {
 public:
  /// \brief Constructor of AbstractJTagged.
  ///
  /// \param[in] element The value to be processed.
  explicit AbstractJTagged(const AbstractBasePtr &element) : element_(element) {}

  /// \brief Destructor of AbstractJTagged.
  ~AbstractJTagged() override = default;
  MS_DECLARE_PARENT(AbstractJTagged, AbstractBase)

  TypePtr BuildType() const override;

  AbstractBasePtr Clone() const override { return std::make_shared<AbstractJTagged>(element_->Clone()); }

  AbstractBasePtr Broaden() const override { return std::make_shared<AbstractJTagged>(element_->Broaden()); }

  AbstractBasePtr Join(const AbstractBasePtr &other) override;

  /// \brief Overwrite the operator '==' to compare other AbstractJTagged.
  ///
  /// \param[in] other The other abstract to be joined.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractBase &other) const override;

  std::string ToString() const override;

  /// \brief Get the element.
  ///
  /// \return A pointer to a abstract, which is the element_.
  AbstractBasePtr element() { return element_; }

  std::size_t hash() const override { return hash_combine(tid(), element_->hash()); }

 private:
  AbstractBasePtr element_;
};
using AbstractJTaggedPtr = std::shared_ptr<AbstractJTagged>;

/// \brief Class AbstractNone describes a None node's abstract value.
class MS_CORE_API AbstractNone final : public AbstractBase {
 public:
  /// \brief Constructor of AbstractNone.
  AbstractNone() : AbstractBase() { set_type(std::make_shared<TypeNone>()); }

  /// \brief Destructor of AbstractNone.
  ~AbstractNone() override = default;
  MS_DECLARE_PARENT(AbstractNone, AbstractBase)

  TypePtr BuildType() const override { return std::make_shared<TypeNone>(); }

  /// \brief Overwrite the operator '==' to compare other AbstractNone.
  ///
  /// \param[in] other The other instance of AbstractNone.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractBase &other) const override;

  AbstractBasePtr Clone() const override { return std::make_shared<AbstractNone>(); }

  std::string ToString() const override;

 protected:
  ValuePtr RealBuildValue() const override;
};
using AbstractNonePtr = std::shared_ptr<AbstractNone>;

/// \brief Class AbstractNone describes a Null node's abstract value.
///
/// The unassigned state value for variable,
/// which means the variable is not assigned.
class MS_CORE_API AbstractNull final : public AbstractBase {
 public:
  /// \brief Constructor of AbstractNull.
  AbstractNull() : AbstractBase(kNull) { set_type(std::make_shared<TypeNull>()); }

  /// \brief Destructor of AbstractNull.
  ~AbstractNull() override = default;
  MS_DECLARE_PARENT(AbstractNull, AbstractBase)

  TypePtr BuildType() const override { return std::make_shared<TypeNull>(); }

  /// \brief Overwrite the operator '==' to compare other AbstractNull.
  ///
  /// \param[in] other The other instance of AbstractNull.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractBase &other) const override;

  AbstractBasePtr Clone() const override { return std::make_shared<AbstractNull>(); }

  std::string ToString() const override;
};
using AbstractNullPtr = std::shared_ptr<AbstractNull>;

/// \brief Class AbstractTimeOut describes a TimeOut node's abstract value.
///
/// The timeout state value for variable, which means
/// the variable is not assigned because it is timeout.
class MS_CORE_API AbstractTimeOut final : public AbstractBase {
 public:
  /// \brief Constructor of AbstractTimeOut.
  AbstractTimeOut() : AbstractBase(kNull) { set_type(std::make_shared<TypeNull>()); }

  /// \brief Destructor of AbstractTimeOut.
  ~AbstractTimeOut() override = default;
  MS_DECLARE_PARENT(AbstractTimeOut, AbstractBase)

  TypePtr BuildType() const override { return std::make_shared<TypeNull>(); }

  /// \brief Overwrite the operator '==' to compare other AbstractTimeOut.
  ///
  /// \param[in] other The other instance of AbstractTimeOut.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractBase &other) const override;

  AbstractBasePtr Clone() const override { return std::make_shared<AbstractTimeOut>(); }

  std::string ToString() const override;
};
using AbstractTimeOutPtr = std::shared_ptr<AbstractTimeOut>;

/// \brief Class AbstractEllipsis describes a Ellipsis node's abstract value.
class MS_CORE_API AbstractEllipsis final : public AbstractBase {
 public:
  /// \brief Constructor of AbstractEllipsis.
  AbstractEllipsis() : AbstractBase(kEllipsis) { set_type(std::make_shared<TypeEllipsis>()); }

  /// \brief Destructor of AbstractEllipsis.
  ~AbstractEllipsis() override = default;
  MS_DECLARE_PARENT(AbstractEllipsis, AbstractBase)

  TypePtr BuildType() const override { return std::make_shared<TypeEllipsis>(); }

  /// \brief Overwrite the operator '==' to compare other AbstractEllipsis.
  ///
  /// \param[in] other The other instance of AbstractTimeOut.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractBase &other) const override;

  AbstractBasePtr Clone() const override { return std::make_shared<AbstractEllipsis>(); }

  std::string ToString() const override;
};
using AbstractEllipsisPtr = std::shared_ptr<AbstractEllipsis>;

/// \brief Class AbstractRefTensor describes a RefTensor's abstract value.
class MS_CORE_API AbstractRefTensor final : public AbstractTensor {
 public:
  /// \brief Constructor of AbstractRef.
  ///
  /// \param[in] ref_value The tensor.
  /// \param[in] ref_key_value The ref key of tensor.
  AbstractRefTensor(const AbstractTensorPtr &ref_value, const ValuePtr &ref_key_value);

  /// \brief Destructor of AbstractEllipsis.
  ~AbstractRefTensor() override = default;
  MS_DECLARE_PARENT(AbstractRefTensor, AbstractTensor)

  TypePtr BuildType() const override;

  /// \brief Overwrite the operator '==' to compare other AbstractRef.
  ///
  /// \param[in] other The other instance of AbstractTimeOut.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractBase &other) const override;

  AbstractBasePtr Clone() const override;

  /// \brief Use parent's AbstractTensor::Clone() to clone an abstract.
  ///
  /// \return A pointer to the cloned abstract.
  AbstractBasePtr CloneAsTensor() const { return AbstractTensor::Clone(); }

  std::string ToString() const override;

  /// \brief Get the abstract tensor, which is referenced.
  ///
  /// \return A pointer to the abstract tensor.
  inline AbstractTensorPtr ref() { return shared_from_base<AbstractTensor>(); }

  /// \brief Get the ref key value, ref key is string actually.
  ///
  /// \return A point to the RefKey.
  inline ValuePtr ref_key_value() const { return ref_key_value_; }

  AbstractBasePtr Broaden() const override;

  virtual AbstractBasePtr Join(const std::shared_ptr<AbstractRefTensor> &other);
  AbstractBasePtr Join(const AbstractBasePtr &other) override;

  AbstractBasePtr PartialBroaden() const override;

 private:
  // ref_key_value is the reference key of AbstractRef, the value can be a string value or kAnyValue
  ValuePtr ref_key_value_;
};
using AbstractRefPtr = std::shared_ptr<AbstractRefTensor>;

/// \brief Compute the hash of a list of abstracts.
///
/// \param[in] args_spec_list A list of abstracts.
/// \return A hash number.
MS_CORE_API std::size_t AbstractBasePtrListHash(const AbstractBasePtrList &args_spec_list);

/// \brief Determine whether a list of abstracts is equal to another.
///
/// \param[in] lhs The first list of abstracts.
/// \param[in] rhs The second list of abstracts.
/// \return A boolean.
MS_CORE_API bool AbstractBasePtrListDeepEqual(const AbstractBasePtrList &lhs, const AbstractBasePtrList &rhs);

/// \brief Struct AbstractBasePtrListHasher provides a function to compute the hash of a list of abstracts.
struct AbstractBasePtrListHasher {
  std::size_t operator()(const AbstractBasePtrList &args_spec_list) const {
    return AbstractBasePtrListHash(args_spec_list);
  }
};

/// \brief Struct AbstractBasePtrListEqual provides a function to determine whether a list of abstracts is equal to
///        another.
struct AbstractBasePtrListEqual {
  bool operator()(const AbstractBasePtrList &lhs, const AbstractBasePtrList &rhs) const {
    return AbstractBasePtrListDeepEqual(lhs, rhs);
  }
};

class MS_CORE_API AbstractSparseTensor : public AbstractTuple {
 public:
  explicit AbstractSparseTensor(AbstractBasePtrList &&elements,
                                const std::shared_ptr<AnfNodeWeakPtrList> &tuple_nodes = nullptr)
      : AbstractTuple(std::move(elements), tuple_nodes) {}

  explicit AbstractSparseTensor(const AbstractBasePtrList &elements,
                                const std::shared_ptr<AnfNodeWeakPtrList> &tuple_nodes = nullptr)
      : AbstractTuple(elements, tuple_nodes) {}

  ~AbstractSparseTensor() override = default;
  MS_DECLARE_PARENT(AbstractSparseTensor, AbstractTuple)

  template <typename T>
  const T GetAbsPtrAt(size_t index) const;
  /// \brief If any element is a tuple, get every element shape in it.
  BaseShapePtrList ElementsShapeTupleRecursive() const;
  TypePtr BuildType() const override;
  BaseShapePtr BuildShape() const override { return std::make_shared<TupleShape>(ElementsShapeTupleRecursive()); }

  /// \brief Return the TypeId of a Tensor element in SparseTensor.
  ///
  /// \param[in] index The index of element to choose.
  /// \return A TypeId.
  const TypeId GetTensorTypeIdAt(size_t index) const;

  /// \brief Return the TypeId of a shape element in SparseTensor. Note that each element in shape will be transformed
  /// to Tensor(scalar) in the backend.
  /// \param[in] index The index of element to choose.
  /// \return A TypeId.
  const TypeId GetShapeTypeIdAt(size_t index) const;

  const AbstractTuplePtr shape() const;
};
using AbstractSparseTensorPtr = std::shared_ptr<AbstractSparseTensor>;

/// \brief Class AbstractRowTensor describes a RowTensor's abstract value.
class MS_CORE_API AbstractRowTensor final : public AbstractUndetermined {
 public:
  /// \brief Constructor of AbstractRowTensor.
  ///
  /// \param[in] element The abstract which is wrapped to a RowTensor's abstract value.
  /// \param[in] shape A dimension of the abstract.
  explicit AbstractRowTensor(const AbstractBasePtr &element, const BaseShapePtr &shape = std::make_shared<Shape>())
      : AbstractUndetermined(element, shape) {}

  /// \brief Constructor of AbstractRowTensor.
  ///
  /// \param[in] element_type The type of RowTensor.
  /// \param[in] shape A dimension of RowTensor.
  AbstractRowTensor(const TypePtr &element_type, const ShapeVector &shape)
      : AbstractUndetermined(element_type, shape) {}

  /// \brief Destructor of AbstractRowTensor.
  ~AbstractRowTensor() override = default;
  MS_DECLARE_PARENT(AbstractRowTensor, AbstractUndetermined)

  /// \brief Get the indices of RowTensor.
  ///
  /// \return A pointer to the abstract tensor.
  const AbstractTensorPtr indices() const { return indices_; }

  /// \brief Set the indices for abstract.
  ///
  /// \param[in] indices The indices.
  void set_indices(const AbstractTensorPtr &indices) { indices_ = indices; }

  /// \brief Get the values.
  ///
  /// \return A pointer to the abstract tensor.
  const AbstractTensorPtr values() const { return values_; }

  /// \brief Set the values.
  ///
  /// \param[in] values The values of tensor.
  void set_values(const AbstractTensorPtr &values) { values_ = values; }

  /// \brief Get the dense shape.
  ///
  /// \return A pointer to the tuple of abstracts.
  const AbstractTuplePtr dense_shape() const { return dense_shape_; }

  /// \brief Set the dense shape.
  ///
  /// \param[in] dense_shape The dense shape of RowTensor.
  void set_dense_shape(const AbstractTuplePtr &dense_shape) { dense_shape_ = dense_shape; }

  TypePtr BuildType() const override;

  AbstractBasePtr Clone() const override;

  AbstractBasePtr Broaden() const override;

  /// \brief Broaden the abstract with the shape not changing.
  ///
  /// \return A pointer to the broadened abstract.
  AbstractBasePtr BroadenWithShape() const;

  std::string ToString() const override;

 private:
  std::shared_ptr<AbstractRowTensor> MakeAbstract(const BaseShapePtr &shp) const;
  AbstractTensorPtr indices_;
  AbstractTensorPtr values_;
  AbstractTuplePtr dense_shape_;
};
using AbstractRowTensorPtr = std::shared_ptr<AbstractRowTensor>;

// COOTensor is a Tuple with fixed number of elements and specific meaning of each position.
class MS_CORE_API AbstractCOOTensor : public AbstractSparseTensor {
 public:
  explicit AbstractCOOTensor(AbstractBasePtrList &&elements,
                             const std::shared_ptr<AnfNodeWeakPtrList> &tuple_nodes = nullptr)
      : AbstractSparseTensor(std::move(elements), tuple_nodes) {}

  explicit AbstractCOOTensor(const AbstractBasePtrList &elements,
                             const std::shared_ptr<AnfNodeWeakPtrList> &tuple_nodes = nullptr)
      : AbstractSparseTensor(elements, tuple_nodes) {}

  ~AbstractCOOTensor() override = default;
  MS_DECLARE_PARENT(AbstractCOOTensor, AbstractSparseTensor)

  const AbstractTensorPtr indices() const;
  const AbstractTensorPtr values() const;

  TypePtr BuildType() const override;
  AbstractBasePtr Clone() const override;
  AbstractBasePtr Broaden() const override;
  AbstractBasePtr PartialBroaden() const override;
  std::string ToString() const override;

  static constexpr size_t kIndicesIdx = 0;
  static constexpr size_t kValuesIdx = 1;
};
using AbstractCOOTensorPtr = std::shared_ptr<AbstractCOOTensor>;

// CSRTensor is a Tuple with fixed number of elements and specific meaning of each position.
class MS_CORE_API AbstractCSRTensor : public AbstractSparseTensor {
 public:
  explicit AbstractCSRTensor(AbstractBasePtrList &&elements,
                             const std::shared_ptr<AnfNodeWeakPtrList> &tuple_nodes = nullptr)
      : AbstractSparseTensor(std::move(elements), tuple_nodes) {}

  explicit AbstractCSRTensor(const AbstractBasePtrList &elements,
                             const std::shared_ptr<AnfNodeWeakPtrList> &tuple_nodes = nullptr)
      : AbstractSparseTensor(elements, tuple_nodes) {}

  ~AbstractCSRTensor() override = default;
  MS_DECLARE_PARENT(AbstractCSRTensor, AbstractSparseTensor)

  const AbstractTensorPtr indptr() const;
  const AbstractTensorPtr indices() const;
  const AbstractTensorPtr values() const;

  TypePtr BuildType() const override;
  AbstractBasePtr Clone() const override;
  AbstractBasePtr Broaden() const override;
  AbstractBasePtr PartialBroaden() const override;
  std::string ToString() const override;

  static constexpr size_t kIndptrIdx = 0;
  static constexpr size_t kIndicesIdx = 1;
  static constexpr size_t kValuesIdx = 2;
};
using AbstractCSRTensorPtr = std::shared_ptr<AbstractCSRTensor>;

class MS_CORE_API AbstractMonad : public AbstractBase {
 public:
  ~AbstractMonad() override = default;
  MS_DECLARE_PARENT(AbstractMonad, AbstractBase)

  std::size_t hash() const override { return hash_combine({tid()}); }
  TypePtr BuildType() const override { return GetTypeTrack(); }
  AbstractBasePtr Broaden() const override { return AbstractBase::Broaden(); }
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

class MS_CORE_API AbstractUMonad final : public AbstractMonad {
 public:
  explicit AbstractUMonad(const ValuePtr &value = kUMonad) : AbstractMonad(value, kUMonadType) {}
  ~AbstractUMonad() override = default;
  MS_DECLARE_PARENT(AbstractUMonad, AbstractMonad)

  AbstractBasePtr Clone() const override { return std::make_shared<AbstractUMonad>(GetValueTrack()); }
  AbstractBasePtr Join(const AbstractBasePtr &other) override;
  bool operator==(const AbstractBase &other) const override;
};
using AbstractUMonadPtr = std::shared_ptr<AbstractUMonad>;

class MS_CORE_API AbstractIOMonad final : public AbstractMonad {
 public:
  explicit AbstractIOMonad(const ValuePtr &value = kIOMonad) : AbstractMonad(value, kIOMonadType) {}
  ~AbstractIOMonad() override = default;
  MS_DECLARE_PARENT(AbstractIOMonad, AbstractMonad)

  AbstractBasePtr Clone() const override { return std::make_shared<AbstractIOMonad>(GetValueTrack()); }
  AbstractBasePtr Join(const AbstractBasePtr &other) override;
  bool operator==(const AbstractBase &other) const override;
};
using AbstractIOMonadPtr = std::shared_ptr<AbstractIOMonad>;
using tensor::MapTensorPtr;
/// \brief Class AbstractMapTensor describes a MapTensor's abstract value.
class MS_CORE_API AbstractMapTensor final : public AbstractBase {
 public:
  explicit AbstractMapTensor(const MapTensorPtr &map_tensor);
  AbstractMapTensor(const MapTensorPtr &map_tensor, const ValuePtr &ref_key_value);
  AbstractMapTensor(const TypePtr &type, const ShapePtr &value_shape, const ValuePtr &value,
                    const ValuePtr &ref_key_value, const ValuePtr &default_value);
  AbstractMapTensor(const AbstractMapTensor &other);
  AbstractMapTensor(const TypePtr &type, const ShapePtr &value_shape, const ValuePtr &value,
                    const ValuePtr &ref_key_value, const ValuePtr &default_value, const ValuePtr &permit_filter_value,
                    const ValuePtr &evict_filter_value);
  ~AbstractMapTensor() override = default;

  MS_DECLARE_PARENT(AbstractMapTensor, AbstractBase)

  MapTensorTypePtr map_tensor_type() const { return dyn_cast<MapTensorType>(GetTypeTrack()); }
  ShapePtr shape() const { return dyn_cast<Shape>(GetShapeTrack()); }
  const ShapePtr &value_shape() const { return value_shape_; }
  const ValuePtr &ref_key_value() const { return ref_key_value_; }
  const ValuePtr &default_value() const { return default_value_; }
  const ValuePtr &permit_filter_value() const { return permit_filter_value_; }
  const ValuePtr &evict_filter_value() const { return evict_filter_value_; }
  TypePtr BuildType() const override { return GetTypeTrack(); }
  BaseShapePtr BuildShape() const override { return GetShapeTrack(); };

  AbstractBasePtr Clone() const override;
  AbstractBasePtr Join(const AbstractBasePtr &other) override;
  bool operator==(const AbstractBase &other) const override;
  std::size_t hash() const override;
  std::string ToString() const override;

 private:
  // The reference key value, can be a string value or kAnyValue.
  ValuePtr ref_key_value_;
  // The default value, a scalar or string with initializer name.
  ValuePtr default_value_;
  // Permission threshold.
  ValuePtr permit_filter_value_;
  // Remove threshold.
  ValuePtr evict_filter_value_;
  // The value shape.
  ShapePtr value_shape_;
};
using AbstractMapTensorPtr = std::shared_ptr<AbstractMapTensor>;

MS_CORE_API std::string ExtractLoggingInfo(const std::string &info);

MS_CORE_API void SynchronizeSequenceElementsUseFlagsRecursively(const AbstractSequencePtr &lhs_sequence,
                                                                const AbstractSequencePtr &rhs_sequence);

MS_CORE_API ValuePtr GetRefKeyValue(const AbstractBasePtr &abs);
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_ABSTRACT_VALUE_H_
