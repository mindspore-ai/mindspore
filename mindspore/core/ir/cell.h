/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_IR_CELL_H_
#define MINDSPORE_CCSRC_IR_CELL_H_

#include <memory>
#include <string>
#include <tuple>

#include "utils/hash_map.h"
#include "abstract/abstract_value.h"
#include "utils/misc.h"

namespace mindspore {
using abstract::AbstractBasePtr;
using abstract::AbstractBasePtrList;
enum MixedPrecisionType { kNotSet = 0, kFP16 = 1, kFP32 = 2, kBF16 = 3, kAutoPromote = 4 };

/// \brief The Cell class of MindSpore is the base class for building all networks and the basic unit of a network.
class MS_CORE_API Cell final : public Named {
 public:
  /// \brief Constructor.
  ///
  /// \param[in] name The name of Cell.
  explicit Cell(const std::string &name);
  MS_DECLARE_PARENT(Cell, Named);

  abstract::AbstractBasePtr ToAbstract() override;

  std::string ToString() const override;

  /// \brief Get the id of this Cell.
  ///
  /// \return The id of this Cell.
  string id() const { return id_; }

  /// \brief Get information about all attributes.
  ///
  /// \return Details of all attributes.
  std::string GetAttrString() const;

  /// \brief Obtain all attributes of Cell.
  ///
  /// \return All attributes of Cell.
  const mindspore::HashMap<std::string, ValuePtr> &attrs() const { return attrs_; }

  /// \brief Set the attributes of Cell.
  ///
  /// \param[in] attributes Attributes.
  void set_attrs(const mindspore::HashMap<std::string, ValuePtr> &attrs_input) { attrs_ = attrs_input; }

  /// \brief Add a new attribute.
  ///
  /// \param[in] name The name of new attribute.
  /// \param[in] attr The value of new attribute.
  void AddAttr(const std::string &name, const ValuePtr &attr) { attrs_[name] = attr; }

  /// \brief Delete an attribute.
  ///
  /// \param[in] name The name of the attribute to be deleted.
  void DelAttr(const std::string &name);

  /// \brief Obtain the attribute corresponding to the name.
  ///
  /// \param[in] attr_name The name of the attribute.
  /// \return The value of the attribute corresponding to the name.
  ValuePtr GetAttr(const std::string &attr_name) const {
    auto iter = attrs_.find(attr_name);
    return iter == attrs_.cend() ? nullptr : iter->second;
  }

  /// \brief Determine whether the Cell has the attribute corresponding to the name.
  ///
  /// \param[in] attr_name The name of the attribute.
  /// \return True if the Cell has this attribute, otherwise False.
  bool HasAttr(const std::string &attr_name) const {
    auto iter = attrs_.find(attr_name);
    return !(iter == attrs_.cend());
  }

  /// \brief Get mixed precision type.
  ///
  /// \return The mixed precision type.
  MixedPrecisionType GetMixedPrecisionType() const { return mixed_type_; }

  /// \brief Set mixed precision type.
  ///
  /// \param[in] mixed_type The type of mixed precision, float16 or float32.
  void SetMixedPrecisionType(enum MixedPrecisionType mixed_type) { mixed_type_ = mixed_type; }

  bool operator==(const Value &other) const override;

  /// \brief Determine whether Cell is the same as other.
  ///
  /// \param[in] other Another Cell.
  /// \return True if the same, otherwise False.
  bool operator==(const Cell &other) const;

  /// \brief Destructor.
  ~Cell() override = default;

 private:
  string id_;
  mindspore::HashMap<std::string, ValuePtr> attrs_;
  enum MixedPrecisionType mixed_type_ { kNotSet };
};

using CellPtr = std::shared_ptr<Cell>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_IR_CELL_H_
