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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_STRIDESLICE_V2_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_STRIDESLICE_V2_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "nnacl/fp32/strided_slice_fp32.h"

namespace mindspore {
namespace kernel {
constexpr auto kStridedSliceV2 = "StridedSliceV2";
class StridedSliceV2CpuKernelMod : public NativeCpuKernelMod {
 public:
  StridedSliceV2CpuKernelMod() = default;
  ~StridedSliceV2CpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  enum ParallelStrategy { kOnSplitAxis, kOnOuter };

  template <typename T>
  void StridedSliceV2LaunchDynamicType(const std::vector<kernel::AddressPtr> &inputs);

  template <typename T>
  void InitSliceParam(const BaseOperatorPtr &base_operator, std::vector<T> *begin, std::vector<T> *end,
                      std::vector<T> *stride);
  bool MatchParallelPattern();
  void InitParallelParam();
  void ParallelRun(const uint8_t *input_addr, uint8_t *output_addr, int thread_num);

  void StridedSliceV2LaunchCal(const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> &outputs);
  common::Status RunTaskOnOuter(const uint8_t *input_addr, uint8_t *output_addr, int start_pos);
  common::Status RunTaskOnSplitAxis(const uint8_t *input_addr, uint8_t *output_addr, int start_pos);

  TypeId dtype_;
  TypeId dtype_attr_;
  int data_size_{4};
  int split_axis_{-1};
  int inner_{1};
  int outer_{1};
  int cal_num_per_thread_{1};
  bool parallel_{false};
  BaseOperatorPtr base_operator_;
  size_t inputs_num_;
  size_t shape_dim_input;
  size_t slice_len;
  ParallelStrategy parallel_strategy_{kOnSplitAxis};
  ShapeVector input_shape_;
  ShapeVector output_shape_;
  ShapeVector begin_shape_;
  ShapeVector end_shape_;
  ShapeVector stride_shape_;
  StridedSliceParameter slice_param_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_STRIDESLICE_V2_CPU_KERNEL_H_
