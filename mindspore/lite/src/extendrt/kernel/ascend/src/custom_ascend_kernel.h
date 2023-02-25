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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_CUSTOM_ASCEND_KERNEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_CUSTOM_ASCEND_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <map>
#include "extendrt/kernel/ascend/options/acl_model_options.h"
#include "extendrt/kernel/ascend/model/model_infer.h"
#include "include/api/types.h"
#include "include/api/context.h"
#include "kernel/kernel.h"
#include "kernel/common_utils.h"
#include "include/errorcode.h"

namespace mindspore::kernel {
namespace acl {
class CustomAscendKernelMod : public kernel::KernelMod {
 public:
  CustomAscendKernelMod();
  ~CustomAscendKernelMod() override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  std::vector<KernelTensorPtr> RetrieveOutputShape() override;
  std::vector<KernelAttr> GetOpSupport() override { return {}; }
  std::vector<KernelTensorPtr> GetInputKernelTensor() override;

 private:
  void RecordInputDataIndex(const std::vector<KernelTensorPtr> &inputs);
  AclModelOptionsPtr GenAclOptions(const BaseOperatorPtr &base_operator);
  void UpdateOutputAddr(const std::vector<AddressPtr> &outputs);
  void UpdateInputKernelTensorInfo();

  bool ResetInputOutputShapes();
  bool OnNewInputShapes(const std::vector<KernelTensorPtr> &new_shapes);

  bool load_model_;
  std::vector<KernelTensorPtr> inputs_;
  std::vector<KernelTensorPtr> outputs_;
  AclModelOptionsPtr acl_options_;
  ModelInferPtr model_infer_;
  size_t input_data_idx_;
};
}  // namespace acl
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_CUSTOM_ASCEND_KERNEL_H_
