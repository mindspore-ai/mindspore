/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_REGISTRY_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_REGISTRY_H_

#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <set>
#include "src/litert/kernel_exec.h"
#include "schema/model_generated.h"

using mindspore::kernel::kKernelArch_MAX;
using mindspore::kernel::kKernelArch_MIN;
using mindspore::schema::PrimitiveType_MAX;
using mindspore::schema::PrimitiveType_MIN;

namespace mindspore::lite {
class MS_API KernelRegistry {
 public:
  KernelRegistry() = default;
  virtual ~KernelRegistry();

  static KernelRegistry *GetInstance();
  virtual kernel::KernelCreator GetCreator(const kernel::KernelKey &desc);
  int GetCreatorFuncIndex(kernel::KernelKey desc);
  void RegKernel(kernel::KernelKey desc, kernel::KernelCreator creator);
  void RegKernel(kernel::KERNEL_ARCH arch, TypeId data_type, int type, kernel::KernelCreator creator);
  bool SupportKernel(const kernel::KernelKey &key);
  int GetKernelExec(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                    const InnerContext *ctx, const mindspore::Context *ms_ctx, const kernel::KernelKey &key,
                    OpParameter *op_parameter, kernel::KernelExec **kernel, const void *primitive = nullptr);
  int ReplaceKernelExec(kernel::KernelExec *kernel, const kernel::KernelKey &key);
  kernel::LiteKernel *GetLiteKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                    const InnerContext *ctx, const kernel::KernelKey &key, OpParameter *parameter);

 protected:
  int GetCustomKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                      const mindspore::Context *ctx, const kernel::KernelKey &key, kernel::KernelExec **kernel,
                      const void *primitive = nullptr);
  static const int device_type_length_{kKernelArch_MAX - kKernelArch_MIN + 1};
  static const int data_type_length_{kNumberTypeEnd - kNumberTypeBegin + 1};
  static const int op_type_length_{PrimitiveType_MAX - PrimitiveType_MIN + 1};
  static const int array_size_{device_type_length_ * data_type_length_ * op_type_length_};
  kernel::KernelCreator *creator_arrays_ = nullptr;
  static const int inner_op_type_length_{PrimType_InnerOpMax - PrimType_InnerOpMin};
  static const int inner_op_array_size_{device_type_length_ * data_type_length_ * inner_op_type_length_};
  kernel::KernelCreator *inner_op_creator_arrays_ = nullptr;

 private:
  void CreatorArraysInit();

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

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_REGISTRY_H_
