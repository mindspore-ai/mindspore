/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "coder/opcoders/op_coder_register.h"
#include <utility>
#include <string>
#include "coder/utils/type_cast.h"
namespace mindspore::lite::micro {

bool CoderKey::operator<(const CoderKey rhs) const {
  return std::tie(this->target_, this->data_type_, this->op_type_) <
         std::tie(rhs.target_, rhs.data_type_, rhs.op_type_);
}

std::string CoderKey::ToString() const {
  std::ostringstream code;
  code << "target: " << EnumNameTarget(target_) << "\t"
       << "data_type_: " << data_type_ << "\t"
       << "op_type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_type_));
  return code.str();
}

OpCoderFactory *OpCoderFactory::GetInstance() {
  static OpCoderFactory reg;
  return &reg;
}

int OpCoderFactory::RegistOpCoder(Target target, TypeId data_type, schema::PrimitiveType operator_type,
                                  const CoderCreatorFunc &creator_func) {
  // check key
  CoderKey key(target, data_type, operator_type);
  // insert pair to registry
  if (this->opcoder_sets_.find(key) != this->opcoder_sets_.end()) {
    MS_LOG(ERROR) << "coder already exist: " << key.ToString();
    return RET_ERROR;
  }
  this->opcoder_sets_.insert(std::pair<CoderKey, CoderCreatorFunc>(key, creator_func));
  return RET_OK;
}

CoderCreatorFunc OpCoderFactory::FindOpCoder(const CoderKey &key) {
  auto iterator = this->opcoder_sets_.find(key);
  if (iterator != this->opcoder_sets_.end()) {
    return iterator->second;
  }
  // matching kAllTargets
  iterator = this->opcoder_sets_.find(key.AllKey());
  if (iterator != this->opcoder_sets_.end()) {
    return iterator->second;
  }
  return nullptr;
}

OpCoderRegister::OpCoderRegister(Target target, TypeId data_type, schema::PrimitiveType operator_type,
                                 const CoderCreatorFunc &creatorFunc) {
  OpCoderFactory::GetInstance()->RegistOpCoder(target, data_type, operator_type, creatorFunc);
}

}  // namespace mindspore::lite::micro
