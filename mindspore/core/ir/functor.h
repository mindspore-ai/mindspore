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

#ifndef MINDSPORE_CORE_IR_FUNCTOR_H_
#define MINDSPORE_CORE_IR_FUNCTOR_H_

#include <string>
#include <memory>
#include <vector>

#include "ir/value.h"
#include "utils/hash_set.h"
#include "utils/hash_map.h"
#include "mindapi/base/shape_vector.h"

namespace mindspore {
/// \brief Functor is a Value object to hold the c++ functors that supports exporting and importing mindir.
class MS_CORE_API Functor : public Value {
 public:
  /// \brief Constructor of Functor.
  ///
  /// \param[in] name The name of functor class
  explicit Functor(const std::string &name) : name_(name) {}
  /// \brief Destructor of Functor.
  ~Functor() override = default;
  MS_DECLARE_PARENT(Functor, Value)

  /// \brief  Get the name of functor object
  ///
  /// \return The name of functor object
  const std::string &name() const { return name_; }

  /// \brief Pack member variables to a Value, it's the inverse operation of FromValue.
  ///
  /// \return ValuePtr that packed member variables.
  virtual ValuePtr ToValue() const = 0;

  /// \brief Unpack member variables from Value, it's the inverse operation of ToValue.
  virtual void FromValue(const ValuePtr &) = 0;

  /// \brief The hash value of the Functor object.
  ///
  /// \return The hash value.
  std::size_t hash() const override { return tid(); }

  /// \brief Check whether the input is the current Value object.
  ///
  /// \param[in] rhs The Value object to be compared.
  /// \return Whether the input is the current Value object.
  bool operator==(const Value &rhs) const override { return &rhs == this; }

  /// \brief Get abstract of the Functor object. abstract of Functor is unavailable.
  abstract::AbstractBasePtr ToAbstract() override {
    MS_LOG(INTERNAL_EXCEPTION) << "Functor[" << name() << "] can't be converted to abstract.";
  }

  /// \brief Show the Functor object.
  ///
  /// \return The description of the Functor object.
  std::string ToString() const override {
    auto value = ToValue();
    if (value == nullptr) {
      value = kNone;
    }
    return "Functor[" + name() + "]{" + value->ToString() + "}";
  }

 protected:
  std::string name_;
};
using FunctorPtr = std::shared_ptr<Functor>;

/// \brief ShapeCalcFunctor is the functor of operator ShapeCalc that encapsulate its Infer and Calc functions.
class MS_CORE_API ShapeCalcFunctor : public Functor {
 public:
  /// \brief Constructor of ShapeCalcFunctor.
  explicit ShapeCalcFunctor(const std::string &name) : Functor(name) {}

  /// \brief Destructor of ShapeCalcFunctor.
  ~ShapeCalcFunctor() override = default;
  MS_DECLARE_PARENT(ShapeCalcFunctor, Functor)

  /// \brief Calculate shapes. It's the real calculation of ShapeCalc kernel.
  /// \param[in] inputs The inputs.
  /// \return Result shapes.
  virtual ShapeArray Calc(const ShapeArray &inputs) const = 0;

  /// \brief The InferShape implementation of ShapeCalc primitive.
  /// \param[in] inputs The inputs.
  /// \param[in] unknown_inputs If i exists in 'unknown_inputs', the shape value of inputs[i] is unknown.
  /// \return Length of each shape that returned by Calc.
  virtual std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) const = 0;
};
using ShapeCalcFunctorPtr = std::shared_ptr<ShapeCalcFunctor>;

// common code to declare ShapeCalcFunctor
#define DECLARE_SHAPE_CALC(reg_name, cls) \
  cls() : ShapeCalcFunctor(reg_name) {}   \
  ~cls() override = default;              \
  MS_DECLARE_PARENT(cls, ShapeCalcFunctor)

/// \brief FunctorRegistry is the registry of functors to support importing functor from mindir.
class MS_CORE_API FunctorRegistry {
 public:
  using Creator = std::function<FunctorPtr()>;
  static FunctorRegistry &Instance() {
    static FunctorRegistry ins{};
    return ins;
  }
  Creator GetCreator(const std::string &name) const {
    auto iter = reg.find(name);
    return iter == reg.end() ? nullptr : iter->second;
  }
  class RegCls {
   public:
    RegCls(const std::string &name, const Creator &creator) { FunctorRegistry::Instance().Register(name, creator); }
    ~RegCls() = default;
  };

  void Register(const std::string &name, const Creator &creator) {
    auto ret = reg.insert({name, creator});
    if (!ret.second) {
      MS_LOG(WARNING) << "Duplicated functor is registered. name: " << name;
    } else {
      MS_LOG(DEBUG) << "Register functor: " << name;
    }
  }

 private:
  FunctorRegistry() = default;
  ~FunctorRegistry() = default;
  HashMap<std::string, Creator> reg;
};

#define REG_FUNCTOR(name, cls) \
  static const FunctorRegistry::RegCls g_functor_##cls((name), []() -> FunctorPtr { return std::make_shared<cls>(); })
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_FUNCTOR_H_
