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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_OP_CODER_REGISTER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_OP_CODER_REGISTER_H_

#include <map>
#include <vector>
#include <memory>
#include <string>
#include "src/lite_kernel.h"
#include "include/model.h"
#include "coder/config.h"
namespace mindspore::lite::micro {
class OperatorCoder;
using CoderCreatorFunc = std::function<std::unique_ptr<OperatorCoder>(
  const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors, const Model::Node *node,
  size_t node_index, Target target)>;

class CoderKey {
 public:
  CoderKey() = delete;

  CoderKey(Target target, TypeId data_type, int op_type) : target_(target), data_type_(data_type), op_type_(op_type) {}

  CoderKey AllKey() const {
    CoderKey key(kAllTargets, data_type_, op_type_);
    return key;
  }

  bool operator<(CoderKey rhs) const;
  std::string ToString() const;

  ~CoderKey() = default;

 private:
  Target target_ = kTargetUnknown;
  TypeId data_type_ = kTypeUnknown;
  int op_type_ = schema::PrimitiveType_NONE;
};

class OpCoderFactory {
 public:
  OpCoderFactory() = default;

  static OpCoderFactory *GetInstance();

  int RegistOpCoder(Target target, TypeId data_type, schema::PrimitiveType operator_type,
                    const CoderCreatorFunc &creator_func);

  CoderCreatorFunc FindOpCoder(const CoderKey &key);

  ~OpCoderFactory() { opcoder_sets_.clear(); }

 private:
  // target || data type || primitive type
  std::map<CoderKey, CoderCreatorFunc> opcoder_sets_;
};

class OpCoderRegister {
 public:
  OpCoderRegister() = delete;

  OpCoderRegister(Target target, TypeId data_type, schema::PrimitiveType operator_type,
                  const CoderCreatorFunc &creator_func);

  ~OpCoderRegister() = default;
};
#define REG_OPERATOR_CODER(target, data_type, operator_type, creator_func) \
  static OpCoderRegister g_##target##data_type##operator_type##Creator(target, data_type, operator_type, creator_func);
}  // namespace mindspore::lite::micro

#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_OP_CODER_REGISTER_H_
