/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_GET_KEYS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_GET_KEYS_H_

#include <vector>
#include <string>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "fl/worker/fl_worker.h"
#include "fl/armour/secure_protocol/key_agreement.h"

namespace mindspore {
namespace kernel {
class GetKeysKernelMod : public NativeCpuKernelMod {
 public:
  GetKeysKernelMod() = default;
  ~GetKeysKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override;

  void Init(const CNodePtr &kernel_node) override;

  void InitKernel(const CNodePtr &kernel_node) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {KernelAttr().AddOutputAttr(kNumberTypeFloat32)};
    return support_list;
  }

 private:
  void BuildGetKeysReq(const std::shared_ptr<fl::FBBuilder> &fbb);
  bool SavePublicKeyList(
    const flatbuffers::Vector<flatbuffers::Offset<mindspore::schema::ClientPublicKeys>> *remote_public_key);

  uint32_t rank_id_;
  uint32_t server_num_;
  uint32_t target_server_rank_;
  std::string fl_id_;
  std::shared_ptr<fl::FBBuilder> fbb_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FL_GET_KEYS_H_
