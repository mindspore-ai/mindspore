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
#include <memory>

#include "utils/log_adapter.h"
#include "utils/hashing.h"
#include "utils/any.h"
#include "utils/flags.h"
#include "utils/hash_map.h"
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

  /// \brief Overwrite the operator '==' to compare other abstract.
  ///
  /// \param[in] other The other abstract to be joined.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  virtual bool operator==(const AbstractBase &other) const;

  /// \brief Set the value for the AbstractBase.
  ///
  /// \param[in] value The value of an anf node.
  void set_value(const ValuePtr &value) { value_ = value; }

  /// \brief Set the type for the AbstractBase.
  ///
  /// \param[in] value The type of an anf node.
  void set_type(const TypePtr &type) { type_ = type; }

  /// \brief Set the shape for the AbstractBase.
  ///
  /// \param[in] value The shape of an anf node.
  virtual void set_shape(const BaseShapePtr &shape) { shape_ = shape; }

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
  ValuePtr GetValueTrack() const { return value_; }

  /// \brief Get the abstract type, which is tracked.
  ///
  /// \return A pointer to the Type.
  TypePtr GetTypeTrack() const { return type_; }

  /// \brief Get the abstract shape, which is tracked.
  ///
  /// \return A pointer to the BaseShape.
  BaseShapePtr GetShapeTrack() const { return shape_; }

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
  static void set_trace_node_provider(TraceNodeProvider trace_node_provider) {
    trace_node_provider_ = trace_node_provider;
  }

  inline static TraceNodeProvider trace_node_provider_ = nullptr;

  /// \brief Broaden the abstract. It will upgrade the abstract to a higher level.
  ///
  /// \return A pointer to the broadened abstract.
  virtual AbstractBasePtr Broaden() const;

  /// \brief Combine two abstracts. If two abstracts are different, it will broaden the abstract value.
  ///
  /// \param[in] other The other abstract to be joined.
  ///
  /// \return A pointer to the combined abstract.
  virtual AbstractBasePtr Join(const AbstractBasePtr &) { return shared_from_base<AbstractBase>(); }
  bool IsBroaden() const { return value_ == kAnyValue; }

  /// \brief Write the abstract's string to the std::ostream.
  ///
  /// \param[in] os A std::ostream.
  /// \param[in] a An abstract.
  ///
  /// \return A std::ostream.
  friend std::ostream &operator<<(std::ostream &os, const std::shared_ptr<AbstractBase> &a) {
    os << a->ToString();
    return os;
  }

  /// \brief Broaden abstract with constraints.
  ///
  /// \return A pointer to the broadened abstract.
  virtual AbstractBasePtr PartialBroaden() const;

 protected:
  /// \brief Build a value when value is not set.
  ///
  /// \return A pointer to the Value.
  virtual ValuePtr RealBuildValue() const { return kAnyValue; }

 private:
  ValuePtr value_;
  TypePtr type_;
  BaseShapePtr shape_;
  std::string value_desc_;  // store initial value description for error report
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

  std::size_t hash() const override { return hash_combine({tid(), GetValueTrack()->hash(), GetTypeTrack()->hash()}); }

  TypePtr BuildType() const override { return GetTypeTrack(); }

  AbstractBasePtr Clone() const override {
    return std::make_shared<AbstractScalar>(GetValueTrack(), GetTypeTrack()->Clone());
  }

  AbstractBasePtr Broaden() const override;

  AbstractBasePtr Join(const AbstractBasePtr &other) override;
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
  AbstractError(const StringImmPtr &err, const AnfNodePtr &node) : AbstractBase(err), node_(node) {
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
    return std::make_shared<AbstractError>(GetValueTrack()->cast<StringImmPtr>(), node_);
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

  /// \brief Get the tracking anf node.
  ///
  /// \return A point to the anf node.
  virtual AnfNodePtr tracking_id() const { return nullptr; }

  /// \brief Set a tracking anf node to the abstract.
  virtual void set_tracking_id(AnfNodePtr) {}

  /// \brief Get the context which manages the abstract.
  ///
  /// \return A point to the context.
  virtual AnalysisContextPtr context() const { return nullptr; }
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

  /// \brief Constructor of AbstractScalar.
  ///
  /// \param[in] element_type The type of abstract tensor.
  /// \param[in] shape A vector of the tensor's shape.
  AbstractTensor(const TypePtr &element_type, const ShapeVector &shape) : AbstractUndetermined(element_type, shape) {}

  /// \brief Constructor of AbstractScalar.
  ///
  /// \param[in] tensor The tensor to be abstracted.
  explicit AbstractTensor(const tensor::TensorPtr &tensor) : AbstractUndetermined(tensor->Dtype(), tensor->shape()) {}

  /// \brief Constructor of AbstractScalar.
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
  bool operator==(const AbstractTensor &other) const;

  bool operator==(const AbstractBase &other) const override;

  std::string ToString() const override;

  std::size_t hash() const override {
    auto hash_sum = hash_combine(tid(), element_->hash());
    auto shape = GetShapeTrack();
    if (shape != nullptr) {
      hash_sum = hash_combine(hash_sum, shape->hash());
    }
    return hash_sum;
  }

  AbstractBasePtr PartialBroaden() const override;

 protected:
  bool equal_to(const AbstractTensor &other) const;
  ValuePtr min_value_ = nullptr;
  ValuePtr max_value_ = nullptr;
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
  explicit AbstractSequence(const AbstractBasePtrList &elements, const AnfNodeWeakPtrList &sequence_nodes)
      : elements_(elements), sequence_nodes_(sequence_nodes) {}

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
  std::size_t size() const { return elements_.size(); }

  /// \brief Get the stored elements.
  ///
  /// \return A vector of elements.
  const AbstractBasePtrList &elements() const { return elements_; }

  /// \brief Purify the elements list, and clean unused elements.
  void PurifyElements();

  /// \brief Get the sequence nodes where these 'AbstractSequence' evaluated from.
  ///
  /// \return The nodes of tuple/list, usually are MakeTuple/MakeList CNodes or tuple/list ValueNodes.
  const AnfNodeWeakPtrList &sequence_nodes() const { return sequence_nodes_; }

  /// \brief Set the sequence nodes where these 'AbstractSequence' evaluated from.
  ///
  /// \param[in] sequence_nodes The nodes of tuple/list, usually are MakeTuple/MakeList CNodes or tuple/list ValueNodes.
  void set_sequence_nodes(const AnfNodeWeakPtrList &sequence_nodes) { sequence_nodes_ = sequence_nodes; }

  /// \brief Insert a node into the sequence nodes.
  ///
  /// \param[in] sequence_node The node to intert into sequence nodes.
  void insert_sequence_node(const AnfNodePtr &sequence_node) {
    auto iter =
      std::find_if(sequence_nodes_.begin(), sequence_nodes_.end(),
                   [&sequence_node](const AnfNodeWeakPtr &weak_node) { return sequence_node == weak_node.lock(); });
    if (iter == sequence_nodes_.end()) {
      sequence_nodes_.emplace_back(sequence_node);
    } else {
      MS_LOG(EXCEPTION) << "Fail to insert node \'" << sequence_node->DebugString() << "\' into sequence nodes.";
    }
  }

  /// \brief Update the sequence nodes.
  ///
  /// \param[in] old_sequence_node The old node in sequence nodes.
  /// \param[in] new_sequence_node The new node to replace old node in sequence nodes.
  void update_sequence_node(const AnfNodePtr &old_sequence_node, const AnfNodePtr &new_sequence_node) {
    auto iter = std::find_if(
      sequence_nodes_.begin(), sequence_nodes_.end(),
      [&old_sequence_node](const AnfNodeWeakPtr &weak_node) { return old_sequence_node == weak_node.lock(); });
    if (iter != sequence_nodes_.end()) {
      *iter = new_sequence_node;
      return;
    }
    MS_LOG(EXCEPTION) << "Not found old node \'" << old_sequence_node->DebugString() << "\' in sequence nodes.";
  }

  std::size_t hash() const override;

  std::string ToStringInternal() const;
  std::string ToString() const override;

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
  virtual bool operator==(const AbstractSequence &other) const;

 protected:
  AbstractBasePtrList elements_;
  AnfNodeWeakPtrList sequence_nodes_;
};
using AbstractSequencePtr = std::shared_ptr<AbstractSequence>;

/// \brief Class AbstractTuple describes a tuple.
class MS_CORE_API AbstractTuple final : public AbstractSequence {
 public:
  /// \brief Constructor of AbstractTuple.
  ///
  /// \param[in] elements A list of abstracts.
  /// \param[in] tuple_node The nodes of tuple, usually are MakeTuple CNodes or tuple ValueNodes.
  explicit AbstractTuple(const AbstractBasePtrList &elements, const AnfNodeWeakPtrList &tuple_nodes = {})
      : AbstractSequence(elements, tuple_nodes) {}

  /// \brief Destructor of AbstractTuple.
  ~AbstractTuple() override = default;
  MS_DECLARE_PARENT(AbstractTuple, AbstractSequence)

  TypePtr BuildType() const override { return std::make_shared<Tuple>(ElementsType()); }

  BaseShapePtr BuildShape() const override { return std::make_shared<TupleShape>(ElementsShape()); }

  AbstractBasePtr Clone() const override { return std::make_shared<AbstractTuple>(ElementsClone(), sequence_nodes_); }

  AbstractBasePtr Broaden() const override {
    return std::make_shared<AbstractTuple>(ElementsBroaden(), sequence_nodes_);
  }

  AbstractBasePtr PartialBroaden() const override {
    return std::make_shared<AbstractTuple>(ElementsPartialBroaden(), sequence_nodes_);
  }

  AbstractBasePtr Join(const AbstractBasePtr &other) override {
    auto res = dyn_cast<AbstractSequence>(ElementsJoin<AbstractTuple>(other));
    MS_EXCEPTION_IF_NULL(res);
    res->set_sequence_nodes(SequenceNodesJoin(other));
    return res;
  }

  /// \brief Check whether all elements of the tuple are tensors.
  ///
  /// \return Whether all elements of the tuple are tensors.
  bool ContainsAllBroadenTensors() const;

  /// \brief Overwrite the operator '==' to compare other abstract tuple.
  ///
  /// \param[in] other The other instance of AbstractTuple.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractTuple &other) const;

  bool operator==(const AbstractBase &other) const override;

 protected:
  ValuePtr RealBuildValue() const override { return ElementsBuildValue<ValueTuple>(); }
};
using AbstractTuplePtr = std::shared_ptr<AbstractTuple>;

/// \brief Class AbstractList describes a list.
class MS_CORE_API AbstractList final : public AbstractSequence {
 public:
  /// \brief Constructor of AbstractList.
  ///
  /// \param[in] elements A list of abstracts.
  /// \param[in] list_node The nodes of list, usually are MakeList CNodes or list ValueNodes.
  explicit AbstractList(const AbstractBasePtrList &elements, const AnfNodeWeakPtrList &list_nodes = {})
      : AbstractSequence(elements, list_nodes) {}

  /// \brief Destructor of AbstractList.
  ~AbstractList() override = default;
  MS_DECLARE_PARENT(AbstractList, AbstractSequence)

  TypePtr BuildType() const override { return std::make_shared<List>(ElementsType()); }

  BaseShapePtr BuildShape() const override { return std::make_shared<ListShape>(ElementsShape()); }

  AbstractBasePtr Clone() const override { return std::make_shared<AbstractList>(ElementsClone(), sequence_nodes_); }

  AbstractBasePtr Broaden() const override {
    return std::make_shared<AbstractList>(ElementsBroaden(), sequence_nodes_);
  }

  AbstractBasePtr PartialBroaden() const override {
    return std::make_shared<AbstractList>(ElementsPartialBroaden(), sequence_nodes_);
  }

  AbstractBasePtr Join(const AbstractBasePtr &other) override {
    auto res = dyn_cast<AbstractSequence>(ElementsJoin<AbstractList>(other));
    MS_EXCEPTION_IF_NULL(res);
    res->set_sequence_nodes(SequenceNodesJoin(other));
    return res;
  }

  /// \brief Overwrite the operator '==' to compare other abstract list.
  ///
  /// \param[in] other The other instance of AbstractList.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractList &other) const;

  bool operator==(const AbstractBase &other) const override;

 protected:
  ValuePtr RealBuildValue() const override { return ElementsBuildValue<ValueList>(); }
};
using AbstractListPtr = std::shared_ptr<AbstractList>;

/// \brief Class AbstractClass describes a class node's abstract value.
class MS_CORE_API AbstractClass final : public AbstractBase {
 public:
  /// \brief Constructor of AbstractClass.
  ///
  /// \param[in] tag The name of the class.
  /// \param[in] attributes The abstracts of the attributes of the class.
  /// \param[in] methods The methods of the class.
  AbstractClass(const Named &tag, const std::vector<AbstractAttribute> &attributes,
                const mindspore::HashMap<std::string, ValuePtr> &methods)
      : attributes_(attributes), tag_(tag), methods_(methods) {}

  /// \brief Destructor of AbstractClass.
  ~AbstractClass() override = default;
  MS_DECLARE_PARENT(AbstractClass, AbstractBase)

  TypePtr BuildType() const override;

  /// \brief Overwrite the operator '==' to compare other abstract class.
  ///
  /// \param[in] other The other instance of AbstractClass.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractClass &other) const;

  bool operator==(const AbstractBase &other) const override;

  /// \brief Get the attributes.
  ///
  /// \return A vector of the attributes.
  const std::vector<AbstractAttribute> &attributes() const { return attributes_; }

  /// \brief get the methods of the class.
  ///
  /// \return A map of the method names and methods.
  mindspore::HashMap<std::string, ValuePtr> methods() { return methods_; }

  /// \brief Get a attribute by name.
  ///
  /// \param[in] name The attribute name of the class.
  /// \return A pointer to the abstract.
  AbstractBasePtr GetAttribute(const std::string &name);

  /// \brief Get a method by name.
  ///
  /// \param[in] name The attribute name of the class.
  /// \return A pointer to the value.
  ValuePtr GetMethod(const std::string &name);

  AbstractBasePtr Clone() const override;

  AbstractBasePtr Broaden() const override;

  std::string ToString() const override;

  /// \brief Get the tag in the class.
  ///
  /// \return An instance of Named.
  Named tag() const { return tag_; }

  std::size_t hash() const override;

 protected:
  ValuePtr RealBuildValue() const override;

 private:
  std::vector<AbstractAttribute> attributes_;
  Named tag_;
  mindspore::HashMap<std::string, ValuePtr> methods_;
};
using AbstractClassPtr = std::shared_ptr<AbstractClass>;

/// \brief Class AbstractDictionary describes a dictionary node's abstract value.
class MS_CORE_API AbstractDictionary final : public AbstractBase {
 public:
  /// \brief Constructor of AbstractDictionary.
  ///
  /// \param[in] key_values The vector of AbstractAttribute.
  explicit AbstractDictionary(const std::vector<AbstractAttribute> &key_values) : key_values_(key_values) {}

  /// \brief Destructor of AbstractDictionary.
  ~AbstractDictionary() override = default;
  MS_DECLARE_PARENT(AbstractDictionary, AbstractBase)

  TypePtr BuildType() const override;

  /// \brief Overwrite the operator '==' to compare other abstract dictionary.
  ///
  /// \param[in] other The other instance of AbstractDictionary.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractDictionary &other) const;

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
  /// \return A vector of AbstractAttribute.
  const std::vector<AbstractAttribute> &elements() const { return key_values_; }

  std::vector<AbstractAttribute> key_values_;

 protected:
  ValuePtr RealBuildValue() const override;
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
  bool operator==(const AbstractSlice &other) const;

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
  bool operator==(const AbstractJTagged &other) const;

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
  bool operator==(const AbstractNone &other) const;

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
  bool operator==(const AbstractNull &other) const;

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
  bool operator==(const AbstractTimeOut &other) const;

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
  bool operator==(const AbstractEllipsis &other) const;

  bool operator==(const AbstractBase &other) const override;

  AbstractBasePtr Clone() const override { return std::make_shared<AbstractEllipsis>(); }

  std::string ToString() const override;
};
using AbstractEllipsisPtr = std::shared_ptr<AbstractEllipsis>;

/// \brief Class AbstractRefKey describes a RefKey node's abstract value.
class MS_CORE_API AbstractRefKey final : public AbstractBase {
 public:
  /// \brief Constructor of AbstractRefKey.
  AbstractRefKey() : AbstractBase(), ref_key_value_(nullptr) { set_type(std::make_shared<RefKeyType>()); }

  /// \brief Destructor of AbstractRefKey.
  ~AbstractRefKey() override = default;
  MS_DECLARE_PARENT(AbstractRefKey, AbstractBase)

  TypePtr BuildType() const override { return std::make_shared<RefKeyType>(); }

  /// \brief Overwrite the operator '==' to compare other v.
  ///
  /// \param[in] other The other instance of AbstractTimeOut.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
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

  /// \brief Get the ref key.
  ///
  /// \param[in] The pointer to RefKey.
  RefKeyPtr ref_key_value() const { return ref_key_value_; }

  AbstractBasePtr Join(const AbstractBasePtr &other) override;

  std::string ToString() const override;

 private:
  // cache for ref_key after build value, when value is null, return nullptr.
  RefKeyPtr ref_key_value_{nullptr};
};
using AbstractRefKeyPtr = std::shared_ptr<AbstractRefKey>;

/// \brief Class AbstractRef describes a RefTensor's abstract value.
class MS_CORE_API AbstractRef final : public AbstractTensor {
 public:
  /// \brief Constructor of AbstractRef.
  ///
  /// \param[in] ref_key The ref key of tensor.
  /// \param[in] ref_value The tensor.
  AbstractRef(const AbstractBasePtr &ref_key, const AbstractTensorPtr &ref_value);

  /// \brief Destructor of AbstractEllipsis.
  ~AbstractRef() override = default;
  MS_DECLARE_PARENT(AbstractRef, AbstractTensor)

  TypePtr BuildType() const override;

  /// \brief Overwrite the operator '==' to compare other AbstractRef.
  ///
  /// \param[in] other The other instance of AbstractTimeOut.
  ///
  /// \return A boolean, which indicates whether the other abstract is same.
  bool operator==(const AbstractRef &other) const;

  bool operator==(const AbstractBase &other) const override;

  AbstractBasePtr Clone() const override {
    auto abs_tensor = AbstractTensor::Clone()->cast<AbstractTensorPtr>();
    if (abs_tensor == nullptr) {
      return nullptr;
    }
    return std::make_shared<AbstractRef>(ref_key_->Clone(), abs_tensor);
  }

  /// \brief Use parent's AbstractTensor::Clone() to clone an abstract.
  ///
  /// \return A pointer to the cloned abstract.
  AbstractBasePtr CloneAsTensor() const { return AbstractTensor::Clone(); }

  std::string ToString() const override;

  /// \brief Get the abstract tensor, which is referenced.
  ///
  /// \return A pointer to the abstract tensor.
  inline AbstractTensorPtr ref() { return shared_from_base<AbstractTensor>(); }

  /// \brief Get the ref key.
  ///
  /// \return A pointer to the abstract key.
  inline AbstractBasePtr ref_key() const { return ref_key_; }

  /// \brief Get the ref key value.
  ///
  /// \return A point to the RefKey.
  inline RefKeyPtr ref_key_value() const { return ref_key_value_; }

  AbstractBasePtr Broaden() const override {
    // always broaden for ref
    auto abs_tensor = AbstractTensor::Broaden()->cast<AbstractTensorPtr>();
    if (abs_tensor == nullptr) {
      return nullptr;
    }
    return std::make_shared<AbstractRef>(ref_key_->Broaden(), abs_tensor);
  }

  AbstractBasePtr Join(const AbstractBasePtr &other) override;

  AbstractBasePtr PartialBroaden() const override;

 private:
  AbstractBasePtr ref_key_;
  // cache for ref_key after build value, when value is null, return nullptr.
  RefKeyPtr ref_key_value_;
};
using AbstractRefPtr = std::shared_ptr<AbstractRef>;

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
  AbstractTensorPtr indices_;
  AbstractTensorPtr values_;
  AbstractTuplePtr dense_shape_;
};

/// \brief Class AbstractSparseTensor describes a SparseTensor's abstract value.
class MS_CORE_API AbstractSparseTensor final : public AbstractUndetermined {
 public:
  /// \brief Constructor of AbstractSparseTensor.
  ///
  /// \param[in] element The abstract which is wrapped to be the abstract value of SparseTensor.
  /// \param[in] shape The dimension of the abstract.
  explicit AbstractSparseTensor(const AbstractBasePtr &element, const BaseShapePtr &shape = std::make_shared<Shape>())
      : AbstractUndetermined(element, shape) {}

  /// \brief Constructor of AbstractSparseTensor.
  ///
  /// \param[in] element_type The type of SparseTensor.
  /// \param[in] shape The dimension of SparseTensor.
  AbstractSparseTensor(const TypePtr &element_type, const ShapeVector &shape)
      : AbstractUndetermined(element_type, shape) {}

  /// \brief Destructor of AbstractSparseTensor.
  ~AbstractSparseTensor() override = default;
  MS_DECLARE_PARENT(AbstractSparseTensor, AbstractUndetermined)

  /// \brief Get the indices of SparseTensor.
  ///
  /// \return A pointer to the abstract tensor.
  const AbstractTensorPtr indices() const { return indices_; }

  /// \brief Set the indices for the abstract.
  ///
  /// \param[in] indices The indices.
  void set_indices(const AbstractTensorPtr &indices) { indices_ = indices; }

  /// \brief Get the values.
  ///
  /// \return A pointer to the abstract tensor.
  const AbstractTensorPtr values() const { return values_; }

  /// \brief Set the values.
  ///
  /// \param[in] values The values of SparseTensor.
  void set_values(const AbstractTensorPtr &values) { values_ = values; }

  /// \brief Get the dense shape.
  ///
  /// \return A pointer to the tuple of abstracts.
  const AbstractTuplePtr dense_shape() const { return dense_shape_; }

  /// \brief Set the dense shape.
  ///
  /// \param[in] dense_shape The dense shape of SparseTensor.
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
  AbstractTensorPtr indices_;
  AbstractTensorPtr values_;
  AbstractTuplePtr dense_shape_;
};

// CSRTensor
class MS_CORE_API AbstractCSRTensor : public AbstractUndetermined {
 public:
  explicit AbstractCSRTensor(const AbstractBasePtr &element, const BaseShapePtr &shape = std::make_shared<Shape>())
      : AbstractUndetermined(element, shape) {}
  AbstractCSRTensor(const TypePtr &element_type, const ShapeVector &shape)
      : AbstractUndetermined(element_type, shape) {}
  ~AbstractCSRTensor() override = default;
  MS_DECLARE_PARENT(AbstractCSRTensor, AbstractUndetermined)

  const AbstractTensorPtr indptr() const { return indptr_; }
  void set_indptr(const AbstractTensorPtr &indptr) { indptr_ = indptr; }
  const AbstractTensorPtr indices() const { return indices_; }
  void set_indices(const AbstractTensorPtr &indices) { indices_ = indices; }
  const AbstractTensorPtr values() const { return values_; }
  void set_values(const AbstractTensorPtr &values) { values_ = values; }
  const AbstractTuplePtr dense_shape() const { return dense_shape_; }
  void set_dense_shape(const AbstractTuplePtr &dense_shape) { dense_shape_ = dense_shape; }
  TypePtr BuildType() const override;
  AbstractBasePtr Clone() const override;
  AbstractBasePtr Broaden() const override;
  AbstractBasePtr BroadenWithShape() const;

  std::string ToString() const override;

 private:
  AbstractTensorPtr indptr_;
  AbstractTensorPtr indices_;
  AbstractTensorPtr values_;
  AbstractTuplePtr dense_shape_;
};
using AbstractCSRTensorPtr = std::shared_ptr<AbstractCSRTensor>;

class AbstractMonad : public AbstractBase {
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

class AbstractUMonad final : public AbstractMonad {
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

class AbstractIOMonad final : public AbstractMonad {
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

AnfNodePtr GetTraceNode(const AbstractBasePtr &abs);
std::string ExtractLoggingInfo(const std::string &info);
void SynchronizeSequenceNodesElementsUseFlags(const AnfNodeWeakPtrList &lhs_sequence_nodes,
                                              const AnfNodeWeakPtrList &rhs_sequence_nodes);
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_ABSTRACT_VALUE_H_
