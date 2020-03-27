/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef PREDICT_SRC_OP_REGISTRY_H_
#define PREDICT_SRC_OP_REGISTRY_H_

#include <map>
#include <string>
#include <unordered_map>
#include "common/mslog.h"
#include "common/module_registry.h"
#include "src/op.h"

#define MSPREDICT_API __attribute__((visibility("default")))

namespace mindspore {
namespace predict {
class MSPREDICT_API OpRegistry {
 public:
  OpRegistry();
  virtual ~OpRegistry();

  static OpRegistry *GetInstance();
  virtual OpCreator GetOpCreator(const OpDesc &desc);

  const std::map<OpDesc, OpCreator> &GetOpCreators();

  void RegOp(OpDesc desc, OpCreator creator);
  void RegOp(OP_ARCH arch, OpT type, OpCreator creator);
  static bool Merge(const std::unordered_map<OpDesc, OpCreator> &newCreators);

 protected:
  std::map<OpDesc, OpCreator> creators;
};

template <>
class Module<OpRegistry> : public ModuleBase {
 public:
  virtual OpRegistry *GetInstance() = 0;
};

const char MODULE_REG_NAME_OP_REGISTRY[] = "op_registry";

class OpRegistrar {
 public:
  OpRegistrar(const OpDesc &desc, OpCreator creator) { OpRegistry::GetInstance()->RegOp(desc, creator); }

  OpRegistrar(const OP_ARCH arch, const OpT type, OpCreator creator) {
    MS_ASSERT(OpRegistry::GetInstance() != nullptr);
    OpRegistry::GetInstance()->RegOp(arch, type, creator);
  }
};

#define REG_OP(arch, type, opCreater) static OpRegistrar g_##arch##type##OpReg(arch, type, opCreater);
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_SRC_OP_REGISTRY_H_
