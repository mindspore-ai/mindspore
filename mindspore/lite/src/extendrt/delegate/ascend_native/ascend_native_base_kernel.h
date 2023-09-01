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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_BASE_KERNEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_BASE_KERNEL_H_

#include <string>
#include <utility>
#include <vector>
#include <memory>
#include "extendrt/delegate/type.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "extendrt/kernel/base_kernel.h"
#include "ops/base_operator.h"
#include "ops/op_name.h"

namespace mindspore {
namespace common {
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;
}  // namespace common
struct KernelWithIndexAndTensor {
  KernelWithIndexAndTensor() = default;
  KernelWithIndexAndTensor(common::KernelWithIndex kernel_index, kernel::InferTensor *tensor_info)
      : kernel_index(kernel_index), tensor_info(tensor_info) {}

  common::KernelWithIndex kernel_index;
  kernel::InferTensor *tensor_info{nullptr};
};
namespace kernel {
class AscendNativeBaseKernel : public BaseKernel {
 public:
  AscendNativeBaseKernel() = delete;

  AscendNativeBaseKernel(const AscendNativeBaseKernel &kernel) = delete;

  AscendNativeBaseKernel(AscendNativeBaseKernel &&other) = delete;

  AscendNativeBaseKernel &operator=(const AscendNativeBaseKernel &kernel) = delete;

  AscendNativeBaseKernel &operator=(AscendNativeBaseKernel &&src) = delete;

  AscendNativeBaseKernel(InferPrimitive prim, const InferContext *ctx, const void *stream, std::string name)
      : BaseKernel(prim, ctx), stream_(stream), name_(name) {}

  AscendNativeBaseKernel(const std::vector<InferTensor *> &inputs, const std::vector<InferTensor *> &outputs,
                         InferPrimitive prim, const InferContext *ctx, const void *stream, std::string name)
      : BaseKernel(prim, inputs, outputs, ctx), stream_(stream), name_(name) {}

  template <class OpsT>
  std::shared_ptr<OpsT> AsOps() {
    return std::make_shared<OpsT>(primitive_.base_operator->GetPrim());
  }

  void set_stream(const void *stream) { stream_ = stream; }
  const void *get_stream() { return stream_; }
  const std::string get_name() const { return name_; }
  void set_name(std::string name) { name_ = name; }
  bool InferShapeDone() const override { return true; }
  int InferShape() override { return mindspore::lite::RET_OK; }
  int PreProcess() override { return mindspore::lite::RET_OK; }
  int PostProcess() override { return mindspore::lite::RET_OK; }
  virtual bool IsWeightInputHanledInner() const { return false; }
  virtual bool isFormatAndTypeSupport(int index, TypeId type, Format fmt) { return true; }
  virtual size_t get_workspace_size() const { return 0; }

  void *get_workspace() const { return ws_ptr_; }
  void set_workspace(void *ws_ptr) { ws_ptr_ = ws_ptr; }

 protected:
  const void *stream_ = nullptr;
  std::string name_;
  FuncGraphPtr func_graph_;

 private:
  void *ws_ptr_ = nullptr;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_NATIVE_BASE_KERNEL_H_
