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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_KERNEL_REGISTER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_KERNEL_REGISTER_H_

#include <map>
#include <vector>
#include <string>
#include <set>
#include "coder/config.h"
#include "coder/opcoders/op_coder_register.h"
#include "ir/dtype/type_id.h"
#include "schema/ops_generated.h"

namespace mindspore::lite::micro {

constexpr char kCustomKernelName[] = "CustomKernel";
constexpr char kCustomKernelParam[] = "CustomParameter";

class KernelRegistry {
 public:
  KernelRegistry() = default;

  static KernelRegistry *GetInstance();

  void RegisterKernel(schema::PrimitiveType op);

  bool CheckRegistered(schema::PrimitiveType op);

  bool HasKernelRegistered();

  std::string GenKernelInterface(const char *func, const char *param);

  ~KernelRegistry() { registry_.clear(); }

 private:
  std::set<schema::PrimitiveType> registry_;
};
}  // namespace mindspore::lite::micro

#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_KERNEL_REGISTER_H_
