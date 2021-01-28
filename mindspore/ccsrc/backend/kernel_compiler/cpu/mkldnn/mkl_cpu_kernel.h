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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MKL_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MKL_CPU_KERNEL_H_

#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include "dnnl.hpp"
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class MKLCPUKernel : public CPUKernel {
 public:
  MKLCPUKernel() = default;
  ~MKLCPUKernel() override = default;

 protected:
  bool BinaryBroadCast(std::vector<size_t> *src0_shape, std::vector<size_t> *src1_shape,
                       std::vector<size_t> *dst_shape);
  void GetPadding(const CNodePtr &kernel_node, const std::string &pad_mode, const std::vector<size_t> &src_shape,
                  const std::vector<size_t> &kernel_size, const std::vector<int> &stride, std::vector<int> *padding_l,
                  std::vector<int> *padding_r, const std::vector<int> &dilation);
  void AddArgument(int arg_key, const dnnl::memory::desc &mem_desc, bool alloc = false);
  void SetArgumentHandle(int arg_key, void *ptr);
  dnnl::memory::format_tag GetDefaultFormatTag(const dnnl::memory::dims &dims) const;
  dnnl::memory::desc GetDefaultMemDesc(const std::vector<size_t> &shape);
  void ExecutePrimitive();
  std::unordered_map<int, dnnl::memory> arguments_;
  std::shared_ptr<dnnl::primitive> primitive_{nullptr};
  inline dnnl::memory::desc formatted_md(const dnnl::memory::dims &dimensions, dnnl::memory::format_tag layout) {
    return dnnl::memory::desc{{dimensions}, dnnl::memory::data_type::f32, layout};
  }
  void Reorder(dnnl::memory *src_mem, dnnl::memory *dst_mem);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MKL_CPU_KERNEL_H_
