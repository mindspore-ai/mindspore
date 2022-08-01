/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <memory>
#include <vector>
#include <utility>
#include "dnnl.hpp"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#ifdef USE_MS_THREADPOOL_FOR_DNNL
#include "dnnl_threadpool.hpp"
#include "dnnl_threadpool_iface.hpp"
#endif

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

#ifdef USE_MS_THREADPOOL_FOR_DNNL
class mkl_threadpool : public dnnl::threadpool_interop::threadpool_iface {
 private:
  ThreadPool *tp_;
  int thread_num_{8};
  bool first_parallel{true};

 public:
  explicit mkl_threadpool(ThreadPool *tp) : tp_(tp) {}
  void set_num_threads(int num) { thread_num_ = num; }
  int get_num_threads() const override { return std::min(SizeToInt(tp_->GetKernelThreadNum()), thread_num_); }
  bool get_in_parallel() const override { return !first_parallel; }
  uint64_t get_flags() const override { return 0; }
  void parallel_for(int n, const std::function<void(int, int)> &fn) override {
    bool need_change_flag = first_parallel ? true : false;
    if (need_change_flag) {
      first_parallel = false;
    }
    int nthr = get_num_threads();
    int n_jobs = std::min(n, nthr);
    auto func = [&, n_jobs](void *, int i, float, float) {
      fn(i, n_jobs);
      return 0;
    };
    (void)tp_->ParallelLaunch(func, nullptr, n_jobs);
    if (need_change_flag) {
      first_parallel = true;
    }
  }
};
#endif

struct PaddingInfo {
  const std::string &pad_mode;
  const dnnl::memory::dims &kernel_size;
  const dnnl::memory::dims &stride;
  const dnnl::memory::dims &dilation;
  dnnl::memory::dims *padding_l{nullptr};
  dnnl::memory::dims *padding_r{nullptr};
  std::vector<int64_t> *padding_invalid{nullptr};
  bool ceil_mode{false};
};

class DeprecatedMKLCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
#ifdef USE_MS_THREADPOOL_FOR_DNNL
  DeprecatedMKLCpuKernelMod() : engine_(dnnl::engine::kind::cpu, 0) {
    auto thread_pool = GetActorMgrInnerThreadPool();
    mkl_threadpool_ = std::make_shared<mkl_threadpool>(thread_pool);
    MS_LOG(DEBUG) << "begin to invoke dnnl::threadpool_interop::make_stream";
    stream_ = dnnl::threadpool_interop::make_stream(engine_, mkl_threadpool_.get());
    MS_LOG(DEBUG) << "end to invoke dnnl::threadpool_interop::make_stream";
  }
#else
  DeprecatedMKLCpuKernelMod() : engine_(dnnl::engine::kind::cpu, 0), stream_(engine_) {}
#endif
  ~DeprecatedMKLCpuKernelMod() override = default;

 protected:
  bool BinaryBroadCast(std::vector<size_t> *src0_shape, std::vector<size_t> *src1_shape,
                       std::vector<size_t> *dst_shape) const;
  void GetPadding(const CNodePtr &kernel_node, const std::vector<int64_t> &src_shape,
                  const PaddingInfo &padding_info) const;
  void AddArgument(int arg_key, const dnnl::memory::desc &mem_desc, bool alloc = false);
  void SetArgumentHandle(int arg_key, void *ptr);
  dnnl::memory::format_tag GetDefaultFormatTag(const dnnl::memory::dims &dims) const;
  dnnl::memory::desc GetDefaultMemDesc(const std::vector<int64_t> &shape) const;
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
  dnnl::engine engine_;
  dnnl::stream stream_;
#ifdef USE_MS_THREADPOOL_FOR_DNNL
  std::shared_ptr<dnnl::threadpool_interop::threadpool_iface> mkl_threadpool_{nullptr};
#endif
};

class MKLCpuKernelMod : public NativeCpuKernelMod {
 public:
#ifdef USE_MS_THREADPOOL_FOR_DNNL
  MKLCpuKernelMod() : engine_(dnnl::engine::kind::cpu, 0) {
    auto thread_pool = pool_ == nullptr ? GetActorMgrInnerThreadPool() : pool_;
    mkl_threadpool_ = std::make_shared<mkl_threadpool>(thread_pool);
    MS_LOG(DEBUG) << "begin to invoke dnnl::threadpool_interop::make_stream";
    stream_ = dnnl::threadpool_interop::make_stream(engine_, mkl_threadpool_.get());
    MS_LOG(DEBUG) << "end to invoke dnnl::threadpool_interop::make_stream";
  }
#else
  MKLCpuKernelMod() : engine_(dnnl::engine::kind::cpu, 0), stream_(engine_) {}
#endif
  ~MKLCpuKernelMod() override = default;

 protected:
  bool BinaryBroadCast(std::vector<size_t> *src0_shape, std::vector<size_t> *src1_shape,
                       std::vector<size_t> *dst_shape) const;
  void GetPadding(const BaseOperatorPtr &base_operator, const std::vector<int64_t> &src_shape,
                  const PaddingInfo &padding_info) const;
  void AddArgument(int arg_key, const dnnl::memory::desc &mem_desc, bool alloc = false);
  void SetArgumentHandle(int arg_key, void *ptr);
  dnnl::memory::format_tag GetDefaultFormatTag(const dnnl::memory::dims &dims) const;

  dnnl::memory::desc GetExactMemDesc(const std::vector<size_t> &shape,
                                     dnnl::memory::data_type type = dnnl::memory::data_type::f32) const;
  dnnl::memory::desc GetExactMemDesc(const std::vector<int64_t> &shape,
                                     dnnl::memory::data_type type = dnnl::memory::data_type::f32) const {
    return GetExactMemDesc(LongVecToSizeVec(shape), type);
  }
  dnnl::memory::desc GetDefaultMemDesc(const std::vector<size_t> &shape) const {
    return GetExactMemDesc(shape, dnnl::memory::data_type::f32);
  }
  dnnl::memory::desc GetDefaultMemDesc(const std::vector<int64_t> &shape) const {
    return GetExactMemDesc(LongVecToSizeVec(shape), dnnl::memory::data_type::f32);
  }
  void ExecutePrimitive();
  inline dnnl::memory::desc formatted_md(const dnnl::memory::dims &dimensions, dnnl::memory::format_tag layout) const {
    MS_LOG(DEBUG) << "begin to invoke constructor of dnnl::memory::desc";
    auto desc = dnnl::memory::desc{{dimensions}, dnnl::memory::data_type::f32, layout};
    MS_LOG(DEBUG) << "end to invoke constructor of dnnl::memory::desc";
    return desc;
  }
  void Reorder(dnnl::memory *src_mem, dnnl::memory *dst_mem);
  size_t GetSize(const dnnl::memory::desc &desc) const;
  dnnl::memory::data_type GetDnnlDataType(TypeId ms_type_id) const;
  void SetDataHandle(dnnl::memory mem, void *ptr);
  void *GetDataHandle(const dnnl::memory &mem) const;
  std::unordered_map<int, dnnl::memory> arguments_;
  std::shared_ptr<dnnl::primitive> primitive_{nullptr};
  dnnl::engine engine_;
  dnnl::stream stream_;
#ifdef USE_MS_THREADPOOL_FOR_DNNL
  std::shared_ptr<dnnl::threadpool_interop::threadpool_iface> mkl_threadpool_{nullptr};
#endif
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MKL_CPU_KERNEL_H_
