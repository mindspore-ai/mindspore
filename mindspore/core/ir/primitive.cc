/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "ir/primitive.h"

#include <utility>
#include "abstract/abstract_function.h"
#include "utils/ms_utils.h"
#include "utils/flags.h"
#include "ops/op_def.h"

namespace mindspore {
static uint64_t MakeId() {
  // Use atomic to make id generator thread safe.
  static std::atomic<uint64_t> last_id{1};
  return last_id.fetch_add(1, std::memory_order_relaxed);
}

void Primitive::SetSideEffectFlag(const std::string &name, bool inplace_prim) {
  const auto &op_def = mindspore::ops::GetOpDef(name);
  const auto &graph_view_prim = op_def != nullptr ? op_def->is_graph_view_ : false;
  graph_view_prim_ = graph_view_prim;
  if (graph_view_prim_ || inplace_prim) {
    set_attr(GRAPH_FLAG_SIDE_EFFECT_MEM, MakeValue(true));
  }
}

std::vector<int64_t> Primitive::GetInplaceIndexes() {
  auto op_def = mindspore::ops::GetOpDef(name());
  std::vector<int64_t> indexes{};
  if (op_def != nullptr) {
    // Get inplace_indexes for a primtive defined by yaml.
    size_t output_size = op_def->returns_.size();
    for (size_t index = 0; index < output_size; ++index) {
      auto inplace_index = op_def->returns_[index].inplace_input_index_;
      (void)indexes.emplace_back(inplace_index);
    }
    MS_LOG(DEBUG) << "For Primitive '" << name() << "', the inplace_input_indexes is " << indexes;
    return indexes;
  }
  // Try to get inplace_indexes for a Python primitive.
  auto input_names = GetAttr("input_names");
  auto output_names = GetAttr("output_names");
  if (input_names == nullptr || output_names == nullptr) {
    return indexes;
  }
  const auto &input_name_list = GetValue<std::vector<std::string>>(input_names);
  std::vector<std::string> output_name_list{};
  if (output_names->isa<StringImm>()) {
    (void)output_name_list.emplace_back(GetValue<std::string>(output_names));
  } else {
    output_name_list = GetValue<std::vector<std::string>>(output_names);
  }
  for (const auto &output : output_name_list) {
    const auto &rw_write_indexes = rw_write_input_indexes();
    auto iter = std::find(input_name_list.begin(), input_name_list.end(), output);
    auto index = std::distance(input_name_list.begin(), iter);
    // Record the ref index when output's name is one of inputs' names and this input is rw_write.
    bool is_ref = (iter != input_name_list.end()) &&
                  (std::find(rw_write_indexes.begin(), rw_write_indexes.end(), index) != rw_write_indexes.end());
    auto inplace_index = is_ref ? index : -1;
    (void)indexes.emplace_back(inplace_index);
  }
  MS_LOG(DEBUG) << "For Primitive '" << name() << "', the inplace_input_indexes is " << indexes;
  return indexes;
}

Primitive::Primitive(const std::string &name, bool is_base, const PrimType prim_type, bool inplace_prim)
    : Named(name),
      prim_type_(prim_type),
      is_base_(is_base),
      has_signature_(false),
      record_evaluate_add_attr_(false),
      const_prim_(false),
      inplace_prim_(inplace_prim),
      id_(MakeId()) {
  SetSideEffectFlag(name, inplace_prim);
  SetSignaturesFromOpDef();
}

Primitive::Primitive(const std::string &name, const mindspore::HashMap<std::string, ValuePtr> &attrs, bool inplace_prim)
    : Named(name),
      attrs_(attrs),
      prim_type_(kPrimTypeBuiltIn),
      is_base_(true),
      has_signature_(false),
      record_evaluate_add_attr_(false),
      const_prim_(false),
      inplace_prim_(inplace_prim),
      id_(MakeId()) {
  SetSideEffectFlag(name, inplace_prim);
  SetSignaturesFromOpDef();
}

Primitive::Primitive(const Primitive &prim)
    : Named(prim),
      attrs_(prim.attrs_),
      evaluate_added_attrs_(prim.evaluate_added_attrs_),
      instance_name_(prim.instance_name_),
      prim_type_(prim.prim_type_),
      is_base_(prim.is_base_),
      has_signature_(prim.has_signature_),
      signatures_(prim.signatures()),
      record_evaluate_add_attr_(false),
      const_prim_(false),
      inplace_prim_(prim.inplace_prim_),
      const_input_indexes_(prim.const_input_indexes_),
      rw_write_input_indexes_(prim.rw_write_input_indexes_),
      inplace_input_indexes_(prim.inplace_input_indexes_),
      id_(prim.id_) {
  SetSideEffectFlag(prim.name(), prim.inplace_prim_);
}

Primitive &Primitive::operator=(const Primitive &other) {
  if (this == &other) {
    return *this;
  }
  Named::operator=(other);
  attrs_ = other.attrs_;
  evaluate_added_attrs_ = other.evaluate_added_attrs_;
  instance_name_ = other.instance_name_;
  is_base_ = other.is_base_;
  has_signature_ = other.has_signature_;
  prim_type_ = other.prim_type_;
  record_evaluate_add_attr_ = false;
  const_prim_ = false;
  inplace_prim_ = other.inplace_prim_;
  id_ = other.id_;
  const_input_indexes_ = other.const_input_indexes_;
  rw_write_input_indexes_ = other.rw_write_input_indexes_;
  inplace_input_indexes_ = other.inplace_input_indexes_;
  return *this;
}

abstract::AbstractBasePtr Primitive::ToAbstract() {
  return std::make_shared<abstract::PrimitiveAbstractClosure>(shared_from_base<Primitive>(), nullptr);
}

bool Primitive::operator==(const Value &other) const {
  if (other.isa<Primitive>()) {
    auto &other_prim = static_cast<const Primitive &>(other);
    return *this == other_prim;
  } else {
    return false;
  }
}

bool Primitive::operator==(const Primitive &other) const {
  if (name() != other.name()) {
    return false;
  }
  return common::IsAttrsEqual(attrs_, other.attrs_);
}

std::string Primitive::GetAttrsText() const {
  if (attrs_.empty()) {
    return "";
  }

  std::ostringstream oss;
  oss << "[";
  bool is_first = true;
  for (auto &attr : attrs_) {
    if (is_first) {
      is_first = false;
    } else {
      oss << ", ";
    }
    const std::string value = attr.second == nullptr ? "" : attr.second->DumpText();
    oss << attr.first << ": " << value;
  }
  oss << "]";

  return oss.str();
}

void Primitive::set_signatures(const std::vector<Signature> &signatures) {
  signatures_ = signatures;
  set_has_signature(!signatures.empty());
  rw_write_input_indexes_.clear();
  for (size_t i = 0; i < signatures.size(); ++i) {
    if (signatures[i].rw == SignatureEnumRW::kRWWrite) {
      (void)rw_write_input_indexes_.emplace_back(i);
    }
  }
  auto inplace_input_indexes = GetInplaceIndexes();
  set_inplace_input_indexes(inplace_input_indexes);
}

void Primitive::SetSignaturesFromOpDef() {
  auto op_def = mindspore::ops::GetOpDef(name());
  if (op_def == nullptr) {
    return;
  }
  const auto &signatures = op_def->signatures_;
  set_signatures(signatures);
}

std::string Primitive::ToString() const {
  if (mindspore::ops::IsPrimitiveFunction(name())) {
    return "PrimFunc_" + name();
  }
  return name();
}
}  // namespace mindspore
