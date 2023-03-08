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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAP_TENSOR_GET_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAP_TENSOR_GET_CPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include <utility>
#include "mindspore/core/ops/map_tensor_get.h"
#include "plugin/device/cpu/kernel/map_tensor/map_tensor_cpu_kernel.h"

namespace mindspore {
namespace kernel {
constexpr size_t kMapTensorGetInputNum = 2;
constexpr size_t kMapTensorGetOutputNum = 1;

class MapTensorGetCpuKernelMod : public MapTensorCpuKernelMod {
 public:
  MapTensorGetCpuKernelMod() = default;
  ~MapTensorGetCpuKernelMod() override = default;

  std::vector<KernelAttr> GetOpSupport() override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_launch_func_(this, inputs, workspace, outputs);
  }

 private:
  template <typename KeyType, typename ValueType>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

  void InitSizeLists(const ShapeVector &keys_shape, const ShapeVector &output_shape);

  size_t input_key_type_size_{0};
  size_t output_type_size_{0};

  using MapTensorGetLaunchFunc = std::function<bool(MapTensorGetCpuKernelMod *, const std::vector<AddressPtr> &,
                                                    const std::vector<AddressPtr> &, const std::vector<AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, MapTensorGetLaunchFunc>> map_tensor_get_func_list_;
  MapTensorGetLaunchFunc kernel_launch_func_;
  bool insert_default_value_{true};

  // This flag indicates whether the embedding storage capability is enabled, which supports hot data caching and
  // persistent storage of non-hotspot data for hash table, which is generally used in very large hash table
  // scenarios.
  bool enable_embedding_storage_{false};
  // The global unique parameter key of hash table, used to get the embedding storage instance.
  int32_t parameter_key_{-1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAP_TENSOR_GET_CPU_KERNEL_H_
