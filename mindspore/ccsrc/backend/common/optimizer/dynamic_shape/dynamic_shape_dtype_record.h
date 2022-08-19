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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_DYNAMIC_SHAPE_DTYPE_RECORD_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_DYNAMIC_SHAPE_DTYPE_RECORD_H_

#include <map>
#include <vector>
#include "ir/anf.h"
#include "backend/common/optimizer/optimizer.h"

namespace mindspore::opt::dynamic_shape {
class DynamicShapeDtypeManager {
 public:
  static DynamicShapeDtypeManager &GetInstance();
  void Register(const AnfNodePtr &node, const std::vector<TypePtr> &device_abs);
  bool CheckDeviceType(const AnfNodePtr &node) const;
  TypePtrList GetDeviceType(const AnfNodePtr &node);

 private:
  DynamicShapeDtypeManager() = default;
  ~DynamicShapeDtypeManager() = default;
  DISABLE_COPY_AND_ASSIGN(DynamicShapeDtypeManager);

  std::map<AnfNodePtr, TypePtrList> device_type_recorder_;
};

// If the data type of abstract is not same with the one of device, it will replace with device data type.
class DynamicShapeDtypeRecord : public Pass {
 public:
  DynamicShapeDtypeRecord() : Pass("dynamic_shape_dtype_record") {}
  ~DynamicShapeDtypeRecord() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
}  // namespace mindspore::opt::dynamic_shape
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_DYNAMIC_SHAPE_DTYPE_RECORD_H_
