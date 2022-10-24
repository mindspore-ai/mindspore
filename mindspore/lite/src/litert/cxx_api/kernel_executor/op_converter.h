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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_CXX_API_KERNEL_EXECUTOR_OP_CONVERTER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_CXX_API_KERNEL_EXECUTOR_OP_CONVERTER_H_

#include <string>
#include <memory>
#include <map>
#include "ops/base_operator.h"
#include "src/common/log_adapter.h"
#include "include/api/types.h"

namespace mindspore {
namespace lite {
typedef std::shared_ptr<mindspore::ops::BaseOperator> (*OpsConverterCreator)(
  const std::shared_ptr<mindspore::ops::BaseOperator> &op);

class OpsConverterRegistry {
 public:
  static OpsConverterRegistry *GetInstance() {
    static OpsConverterRegistry registry;
    return &registry;
  }
  void InsertOpsMap(const std::string &name, OpsConverterCreator creator) { ops_converter_creators_[name] = creator; }
  OpsConverterCreator GetOpsConverterCreator(const std::string &name) {
    if (ops_converter_creators_.find(name) != ops_converter_creators_.end()) {
      return ops_converter_creators_[name];
    } else {
      return nullptr;
    }
  }

 protected:
  std::map<std::string, OpsConverterCreator> ops_converter_creators_ = {};
};

class RegistryOpsConverter {
 public:
  RegistryOpsConverter(const std::string &name, OpsConverterCreator creator) noexcept {
    OpsConverterRegistry::GetInstance()->InsertOpsMap(name, creator);
  }
  ~RegistryOpsConverter() = default;
};
}  // namespace lite
}  // namespace mindspore
#endif
