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

#ifndef MINDSPORE_LITE_TOOLS_OPERATOR_INFO_REGISTER_H
#define MINDSPORE_LITE_TOOLS_OPERATOR_INFO_REGISTER_H

#include <map>
#include <vector>
#include <string>
#include <memory>
#include "tools/optimizer/parallel/operator_info.h"

namespace mindspore {
namespace opt {
using OperatorInfoCreatorFunc =
  std::function<std::unique_ptr<opt::OperatorInfo>(const std::string &name, const SplitStrategy &strategy)>;

class SplitOpKey {
 public:
  SplitOpKey() = delete;

  SplitOpKey(int op_type, TypeId data_type, bool is_depth_wise)
      : op_type_(op_type), data_type_(data_type), is_depth_wise_(is_depth_wise) {}

  bool operator<(const SplitOpKey &key) const;

  std::string ToString() const;

  ~SplitOpKey() = default;

 private:
  int op_type_{schema::PrimitiveType_NONE};
  TypeId data_type_{kTypeUnknown};
  // Conv && DepthwiseCon has same schema_id, so need this flags
  bool is_depth_wise_{false};
};

class OperatorInfoFactory {
 public:
  static OperatorInfoFactory *GeInstance();

  OperatorInfoFactory(const OperatorInfoFactory &) = delete;

  OperatorInfoFactory &operator=(const OperatorInfoFactory &) = delete;

  void RegisterOperatorInfo(schema::PrimitiveType operator_type, TypeId type_id, bool is_depth_wise,
                            const OperatorInfoCreatorFunc &creator_func);

  OperatorInfoCreatorFunc FindOperatorInfo(const SplitOpKey &split_op_key);

 private:
  OperatorInfoFactory() = default;

  virtual ~OperatorInfoFactory() = default;

 private:
  // key: op_type -->data_type-->-->is_depth_wise-->name
  std::map<SplitOpKey, OperatorInfoCreatorFunc> operator_info_map_;
};

class OperatorInfoRegister {
 public:
  OperatorInfoRegister() = delete;

  OperatorInfoRegister(schema::PrimitiveType operator_type, TypeId type_id, bool is_depth_wise,
                       const OperatorInfoCreatorFunc &creator_func);

  ~OperatorInfoRegister() = default;
};

#define OPERATOR_INFO_REGISTER(operator_type, type_id, is_depth_wise, creator_func)                          \
  static OperatorInfoRegister g_name##operator_type##type_id##is_depth_wise##Creator(operator_type, type_id, \
                                                                                     is_depth_wise, creator_func);
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPERATOR_INFO_REGISTER_H
