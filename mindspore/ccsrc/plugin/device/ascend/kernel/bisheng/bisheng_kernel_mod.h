/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ASCEND_BISHENG_KERNEL_MOD_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ASCEND_BISHENG_KERNEL_MOD_H

#include <algorithm>
#include <utility>
#include <vector>
#include <string>
#include <memory>
#include "kernel/common_utils.h"
#include "plugin/device/ascend/kernel/bisheng/bisheng_op_info.h"
#include "plugin/device/ascend/kernel/ascend_kernel_mod.h"

namespace mindspore {
namespace kernel {
struct BiShengKernelArgs {
  std::vector<ShapeVector> input_shapes;
  std::vector<ShapeVector> output_shapes;
};

class BACKEND_EXPORT BiShengKernelMod : public AscendKernelMod {
 public:
  BiShengKernelMod();
  ~BiShengKernelMod() override;
  std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
                                   const std::vector<AddressPtr> &outputs, uint32_t stream_id) override;
  using TilingFunc = std::function<int(const BiShengKernelArgs &args, std::vector<uint8_t> *tiling_data)>;
  using WorkSpaceFunc = std::function<size_t(const BiShengKernelArgs &args)>;

  std::string FunctionName() const { return func_name_; }
  virtual size_t BlockDim();
  virtual std::string GetOpName() = 0;
  virtual TilingFunc GetTilingFunc() = 0;
  virtual std::vector<WorkSpaceFunc> GetWorkspaceFunc() { return {}; }
  virtual std::string GetBinary() {
    constexpr auto kBishengKernelImplRelativePath = "libbisheng_kernels_impl.so";
    return kBishengKernelImplRelativePath;
  }

 protected:
  std::string func_name_{};

 private:
  void DoTiling(std::vector<void *> *workspace_addrs);
  void *tiling_addr_{nullptr};
};

class TilingPacking {
 public:
  template <typename T>
  inline static void PackTiling(std::vector<uint8_t> *dst, const T &data) {
    auto src = reinterpret_cast<const uint8_t *>(&data);
    (void)dst->insert(dst->end(), src, src + sizeof(T));
  }

  template <typename T>
  inline static void PackTiling(std::vector<uint8_t> *dst, const std::vector<T> &data) {
    auto src = reinterpret_cast<const uint8_t *>(data.data());
    (void)dst->insert(dst->end(), src, src + data.size() * sizeof(T));
  }
};

#define KernelFunc(Clazz)                                                                                         \
 public:                                                                                                          \
  using Func =                                                                                                    \
    std::function<bool(Clazz *, const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, \
                       const std::vector<kernel::AddressPtr> &, void *stream)>;                                   \
  using FuncList = std::vector<std::pair<KernelAttr, Clazz::Func>>;                                               \
  std::vector<KernelAttr> GetOpSupport() override {                                                               \
    std::vector<KernelAttr> support_list;                                                                         \
    (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),                  \
                         [](const std::pair<KernelAttr, Clazz::Func> &pair) { return pair.first; });              \
    return support_list;                                                                                          \
  }                                                                                                               \
                                                                                                                  \
 private:                                                                                                         \
  friend class BishengOpInfoRegister<Clazz>;                                                                      \
  inline static FuncList func_list_ = {};                                                                         \
  static const BishengOpInfoRegister<Clazz> reg_;                                                                 \
  inline static std::string bisheng_name_ = {};                                                                   \
  inline static TilingFunc tiling_func_ = nullptr;                                                                \
  inline static std::vector<std::string> func_name_list_ = {};                                                    \
  inline static std::vector<WorkSpaceFunc> workspace_func_list_ = {};                                             \
  Func kernel_func_;
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ASCEND_BISHENG_KERNEL_MOD_H
