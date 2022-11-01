/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_QUANTIZATION_PARAM_H_
#define MINDSPORE_CORE_IR_QUANTIZATION_PARAM_H_

#include <string>
#include <memory>
#include "ir/named.h"
#include "ir/primal_attr.h"

namespace mindspore {
/// \brief QuantizationParam defines tensor quantization param of MindSpore.
class MS_CORE_API QuantizationParam : public Value {
 public:
  explicit QuantizationParam(const std::string &quant_algo_name) : quant_algo_name_(quant_algo_name) {}
  ~QuantizationParam() = default;

  /// \brief Add attribute to QuantizationParam attribute map.
  ///
  /// \param[in] name The name of attribute.
  /// \param[in] attr The value of attribute.
  /// \return The QuantizationParam to which attribute has been added.
  QuantizationParam &AddAttr(const std::string &name, const ValuePtr &attr) {
    attrs_[name] = attr;
    return *this;
  }

  /// \brief Delete the attribute.
  ///
  /// \param[in] name The name of attribute to be delete.
  /// \return The QuantizationParam to which attribute has been added.
  QuantizationParam &DelAttr(const std::string &name) {
    (void)attrs_.erase(name);
    return *this;
  }

  /// \brief Set attribute to the quant param attribute map.
  void SetAttr(const std::string &attrName, const ValuePtr &attr) { attrs_[attrName] = attr; }
  /// \brief Get QuantizationParam's attribute.
  ///
  /// \param[in] attrName QuantizationParam attribute name.
  /// \return The value of attribute in QuantizationParam attribute map, if the map is not
  ValuePtr GetAttr(const std::string &attrName) const {
    auto iter = attrs_.find(attrName);
    return iter == attrs_.cend() ? nullptr : iter->second;
  }

  /// \brief Use add attribute by using a map,all elements of the map will be added in the QuantizationParam's attribute
  /// map.
  ///
  /// \param[in] attrs The attribute map needs to be added in the QuantizationParam attribute.
  /// \return The QuantizationParam to which attribute has been added.
  QuantizationParam &set_attrs(const mindspore::HashMap<std::string, ValuePtr> &attrs) {
    for (auto &attr : attrs) {
      attrs_[attr.first] = attr.second;
    }
    return *this;
  }

  /// \brief Get QuantizationParam's all attributes.
  ///
  /// \return The QuantizationParam's all attribute.
  const mindspore::HashMap<std::string, ValuePtr> &attrs() const { return attrs_; }

  /// \brief Get QuantizationParam's algorithm name.
  ///
  /// \return The QuantizationParam's algorithm name.
  std::string quant_algo_name() const { return quant_algo_name_; }

  /// \brief Set quantization algorithm name.
  ///
  /// \param[in] quant_algo_name The QuantizationParam's algorithm name.
  /// \return The QuantizationParam to which algorithm name has been added.
  QuantizationParam &set_quant_algo_name(const std::string &quant_algo_name) {
    this->quant_algo_name_ = quant_algo_name;
    return *this;
  }
  MS_DECLARE_PARENT(QuantizationParam, Value);

  bool operator==(const Value &other) const override;
  /// \brief To compare whether two Primitive objects are equal.
  ///
  /// \param[in] other The other QuantizationParam be compared with.
  /// \return return true if the name and attributes of primitives are the same,otherwise return false.
  bool operator==(const QuantizationParam &other) const;

 private:
  std::string quant_algo_name_;
  mindspore::HashMap<std::string, ValuePtr> attrs_;
};
using QuantizationParamPtr = std::shared_ptr<QuantizationParam>;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_QUANT_PARAM_H
