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

#include <string>
#include <unordered_map>
#include <vector>
#include <set>
#include "src/lite_kernel.h"
#include "src/register_kernel.h"
#include "schema/model_generated.h"

using mindspore::kernel::kKernelArch_MAX;
using mindspore::kernel::kKernelArch_MIN;
using mindspore::schema::PrimitiveType_MAX;
using mindspore::schema::PrimitiveType_MIN;

namespace mindspore::lite {
class KernelRegistry {
 public:
  KernelRegistry() = default;
  virtual ~KernelRegistry();

  static KernelRegistry *GetInstance();
  static int Init();
  virtual kernel::KernelCreator GetCreator(const kernel::KernelKey &desc);
  virtual kernel::CreateKernel GetDelegateCreator(const kernel::KernelKey &desc);
  int GetCreatorFuncIndex(kernel::KernelKey desc);
  int GetFuncIndex(const kernel::KernelKey &desc);
  void RegKernel(kernel::KernelKey desc, kernel::KernelCreator creator);
  void RegKernel(kernel::KERNEL_ARCH arch, TypeId data_type, int type, kernel::KernelCreator creator);
  int RegKernel(const std::string &arch, const std::string &vendor, const TypeId data_type, const int type,
                kernel::CreateKernel creator);
  bool Merge(const std::unordered_map<kernel::KernelKey, kernel::KernelCreator> &newCreators);
  bool SupportKernel(const kernel::KernelKey &key);
  int GetKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                const InnerContext *ctx, const kernel::KernelKey &key, OpParameter *op_parameter,
                kernel::LiteKernel **kernel, const void *primitive = nullptr);

 protected:
  static const int device_type_length_{kKernelArch_MAX - kKernelArch_MIN + 1};
  static const int data_type_length_{kNumberTypeEnd - kNumberTypeBegin + 1};
  static const int op_type_length_{PrimitiveType_MAX - PrimitiveType_MIN + 1};
  static const int array_size_{device_type_length_ * data_type_length_ * op_type_length_};
  kernel::KernelCreator *creator_arrays_ = nullptr;
  std::unordered_map<std::size_t, std::unordered_map<std::size_t, kernel::CreateKernel *>> kernel_creators_;
  std::set<std::string> all_vendors_;

 private:
  std::mutex lock_;
};

class KernelRegistrar {
 public:
  KernelRegistrar(const kernel::KernelKey &desc, kernel::KernelCreator creator) {
    KernelRegistry::GetInstance()->RegKernel(desc, creator);
  }
  ~KernelRegistrar() = default;

  KernelRegistrar(const kernel::KERNEL_ARCH arch, const TypeId data_type, const int op_type,
                  kernel::KernelCreator creator) {
    KernelRegistry::GetInstance()->RegKernel(arch, data_type, op_type, creator);
  }
};

#define REG_KERNEL(arch, data_type, op_type, kernelCreater) \
  static KernelRegistrar g_##arch##data_type##op_type##kernelReg(arch, data_type, op_type, kernelCreater);
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_KERNEL_REGISTRY_H_
