/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_KERNEL_REGISTRY_H_
#define MINDSPORE_LITE_SRC_KERNEL_REGISTRY_H_

#include <map>
#include <string>
#include <unordered_map>
#include "src/lite_kernel.h"
#include "schema/model_generated.h"

namespace mindspore::lite {
class KernelRegistry {
 public:
  KernelRegistry();
  virtual ~KernelRegistry();

  static KernelRegistry *GetInstance();
  int Init();
  void FreeCreatorArray();
  virtual kernel::KernelCreator GetCreator(const kernel::KernelKey &desc);
  const kernel::KernelCreator *GetCreatorArrays();
  int GetCreatorFuncIndex(const kernel::KernelKey desc);
  void RegKernel(const kernel::KernelKey desc, kernel::KernelCreator creator);
  void RegKernel(const kernel::KERNEL_ARCH arch, const TypeId data_type, const schema::PrimitiveType type,
                 kernel::KernelCreator creator);
  bool Merge(const std::unordered_map<kernel::KernelKey, kernel::KernelCreator> &newCreators);

 protected:
  kernel::KernelCreator *creator_arrays_ = nullptr;
  int device_type_length_;
  int data_type_length_;
  int op_type_length_;
};

class KernelRegistrar {
 public:
  KernelRegistrar(const kernel::KernelKey &desc, kernel::KernelCreator creator) {
    KernelRegistry::GetInstance()->RegKernel(desc, creator);
  }

  KernelRegistrar(const kernel::KERNEL_ARCH arch, const TypeId data_type, const schema::PrimitiveType op_type,
                  kernel::KernelCreator creator) {
    KernelRegistry::GetInstance()->RegKernel(arch, data_type, op_type, creator);
  }
};

#define REG_KERNEL(arch, data_type, op_type, kernelCreater) \
  static KernelRegistrar g_##arch##data_type##op_type##kernelReg(arch, data_type, op_type, kernelCreater);
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_KERNEL_REGISTRY_H_
