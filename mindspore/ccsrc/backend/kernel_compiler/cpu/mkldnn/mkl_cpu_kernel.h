/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <utility>
#include "dnnl.hpp"
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <class T, class... Args>
auto CreateDesc(Args &&... args) {
  MS_LOG(DEBUG) << "begin to invoke constructor of " << demangle(typeid(T).name());
  auto desc = T(std::forward<Args>(args)...);
  MS_LOG(DEBUG) << "end to invoke constructor of " << demangle(typeid(T).name());
  return desc;
}

template <class T, class... Args>
auto CreatePrimitive(Args &&... args) {
  MS_LOG(DEBUG) << "begin to invoke constructor of " << demangle(typeid(T).name());
  auto prim = std::make_shared<T>(std::forward<Args>(args)...);
  MS_LOG(DEBUG) << "end to invoke constructor of " << demangle(typeid(T).name());
  return prim;
}

template <class T>
auto GetWorkspaceDesc(const T &prim_desc) {
  MS_LOG(DEBUG) << "begin to invoke " << demangle(typeid(T).name()) << "::workspace_desc()";
  auto desc = prim_desc.workspace_desc();
  MS_LOG(DEBUG) << "end to invoke " << demangle(typeid(T).name()) << "::workspace_desc()";
  return desc;
}

template <class T>
auto GetMeanDesc(const T &prim_desc) {
  MS_LOG(DEBUG) << "begin to invoke " << demangle(typeid(T).name()) << "::mean_desc()";
  auto desc = prim_desc.mean_desc();
  MS_LOG(DEBUG) << "end to invoke " << demangle(typeid(T).name()) << "::mean_desc()";
  return desc;
}

template <class T>
auto GetVarianceDesc(const T &prim_desc) {
  MS_LOG(DEBUG) << "begin to invoke " << demangle(typeid(T).name()) << "::variance_desc()";
  auto desc = prim_desc.variance_desc();
  MS_LOG(DEBUG) << "end to invoke " << demangle(typeid(T).name()) << "::variance_desc()";
  return desc;
}

template <class T>
auto GetWeightsLayerDesc(const T &prim_desc) {
  MS_LOG(DEBUG) << "begin to invoke " << demangle(typeid(T).name()) << "::weights_layer_desc()";
  auto desc = prim_desc.weights_layer_desc();
  MS_LOG(DEBUG) << "end to invoke " << demangle(typeid(T).name()) << "::weights_layer_desc()";
  return desc;
}

template <class T>
auto GetWeightsIterDesc(const T &prim_desc) {
  MS_LOG(DEBUG) << "begin to invoke " << demangle(typeid(T).name()) << "::weights_iter_desc()";
  auto desc = prim_desc.weights_iter_desc();
  MS_LOG(DEBUG) << "end to invoke " << demangle(typeid(T).name()) << "::weights_iter_desc()";
  return desc;
}

template <class T>
auto GetBiasDesc(const T &prim_desc) {
  MS_LOG(DEBUG) << "begin to invoke " << demangle(typeid(T).name()) << "::bias_desc()";
  auto desc = prim_desc.bias_desc();
  MS_LOG(DEBUG) << "end to invoke " << demangle(typeid(T).name()) << "::bias_desc()";
  return desc;
}

template <class T>
auto GetDiffWeightsLayerDesc(const T &prim_desc) {
  MS_LOG(DEBUG) << "begin to invoke " << demangle(typeid(T).name()) << "::diff_weights_layer_desc()";
  auto desc = prim_desc.diff_weights_layer_desc();
  MS_LOG(DEBUG) << "end to invoke " << demangle(typeid(T).name()) << "::diff_weights_layer_desc()";
  return desc;
}

template <class T>
auto GetDiffWeightsIterDesc(const T &prim_desc) {
  MS_LOG(DEBUG) << "begin to invoke " << demangle(typeid(T).name()) << "::diff_weights_iter_desc()";
  auto desc = prim_desc.diff_weights_iter_desc();
  MS_LOG(DEBUG) << "end to invoke " << demangle(typeid(T).name()) << "::diff_weights_iter_desc()";
  return desc;
}

template <class T>
auto GetDiffBiasDesc(const T &prim_desc) {
  MS_LOG(DEBUG) << "begin to invoke " << demangle(typeid(T).name()) << "::diff_bias_desc()";
  auto desc = prim_desc.diff_bias_desc();
  MS_LOG(DEBUG) << "end to invoke " << demangle(typeid(T).name()) << "::diff_bias_desc()";
  return desc;
}

template <class T>
auto GetMemDesc(const T &prim_desc) {
  MS_LOG(DEBUG) << "begin to invoke " << demangle(typeid(T).name()) << "::get_desc()";
  auto desc = prim_desc.get_desc();
  MS_LOG(DEBUG) << "end to invoke " << demangle(typeid(T).name()) << "::get_desc()";
  return desc;
}

class MKLCPUKernel : public CPUKernel {
 public:
  MKLCPUKernel() = default;
  ~MKLCPUKernel() override = default;

 protected:
  bool BinaryBroadCast(std::vector<size_t> *src0_shape, std::vector<size_t> *src1_shape,
                       std::vector<size_t> *dst_shape) const;
  void GetPadding(const CNodePtr &kernel_node, const std::string &pad_mode, const std::vector<size_t> &src_shape,
                  const std::vector<size_t> &kernel_size, const std::vector<int> &stride, std::vector<int> *padding_l,
                  std::vector<int> *padding_r, const std::vector<int> &dilation) const;
  void AddArgument(int arg_key, const dnnl::memory::desc &mem_desc, bool alloc = false);
  void SetArgumentHandle(int arg_key, void *ptr);
  dnnl::memory::format_tag GetDefaultFormatTag(const dnnl::memory::dims &dims) const;
  dnnl::memory::desc GetDefaultMemDesc(const std::vector<size_t> &shape) const;
  void ExecutePrimitive();
  inline dnnl::memory::desc formatted_md(const dnnl::memory::dims &dimensions, dnnl::memory::format_tag layout) const {
    MS_LOG(DEBUG) << "begin to invoke constructor of dnnl::memory::desc";
    auto desc = dnnl::memory::desc{{dimensions}, dnnl::memory::data_type::f32, layout};
    MS_LOG(DEBUG) << "end to invoke constructor of dnnl::memory::desc";
    return desc;
  }
  void Reorder(dnnl::memory *src_mem, dnnl::memory *dst_mem);

  size_t GetSize(const dnnl::memory::desc &desc) const;
  void SetDataHandle(dnnl::memory mem, void *ptr);
  void *GetDataHandle(const dnnl::memory &mem) const;

  std::unordered_map<int, dnnl::memory> arguments_;
  std::shared_ptr<dnnl::primitive> primitive_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MKL_CPU_KERNEL_H_
