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
#ifndef MINDSPORE_LITE_EXTENDRT_SESSION_ASCEND_NATIVE_SESSION_H_
#define MINDSPORE_LITE_EXTENDRT_SESSION_ASCEND_NATIVE_SESSION_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <map>

#include "extendrt/infer_session.h"
#include "extendrt/delegate/type.h"
#include "extendrt/delegate/ascend_native/delegate.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"
#include "extendrt/delegate/factory.h"
#include "infer/kernel.h"
#include "infer/tensor.h"
#include "infer/context.h"

namespace mindspore {
class AscendNativeSession : public InferSession {
 public:
  AscendNativeSession() = default;
  explicit AscendNativeSession(std::shared_ptr<mindspore::AscendNativeDelegate> delegate)
      : delegate_(std::move(delegate)) {}
  ~AscendNativeSession() override = default;

  Status Init(const std::shared_ptr<Context> &context, const ConfigInfos &config_info = {}) override;
  Status CompileGraph(FuncGraphPtr graph, const void *data = nullptr, size_t size = 0,
                      uint32_t *graph_id = nullptr) override;
  Status RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs,
                  const MSKernelCallBack &before, const MSKernelCallBack &after) override;
  Status RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                  std::vector<tensor::Tensor> *outputs) override;
  Status Resize(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                const std::vector<std::vector<int64_t>> &dims) override;
  std::vector<MutableTensorImplPtr> GetOutputs(uint32_t graph_id) override;
  std::vector<MutableTensorImplPtr> GetInputs(uint32_t graph_id) override;
  std::vector<std::string> GetOutputNames(uint32_t graph_id) override;
  std::vector<std::string> GetInputNames(uint32_t graph_id) override;
  MutableTensorImplPtr GetOutputByTensorName(uint32_t graph_id, const std::string &tensorName) override;
  MutableTensorImplPtr GetInputByTensorName(uint32_t graph_id, const std::string &name) override;

 private:
  Status FindGraphInputs(const std::vector<AnfNodePtr> &node_list, const std::vector<AnfNodePtr> &graph_inputs,
                         const std::vector<std::shared_ptr<kernel::BaseKernel>> &kernels);
  Status FindGraphOutputs(const std::vector<AnfNodePtr> &node_list, const AnfNodePtr &graph_output,
                          const std::vector<std::shared_ptr<kernel::BaseKernel>> &kernels);
  Status MoveDataFromHostToDevice(void *sd, bool s_fp16, void *dd, bool d_fp16, size_t elem_num);
  Status MoveDataFromDeviceToHost(void *sd, bool s_fp16, void *dd, bool d_fp16, size_t elem_num);
  void *MallocDevice(size_t size);
  void FreeDevice(void *ptr);

  Status RefDataFromOuter(const std::vector<tensor::Tensor> &outer_tensors);
  void ResetTensorData(const std::vector<void *> &old_data, const std::vector<lite::Tensor *> &tensors);
  std::vector<mindspore::tensor::Tensor> LiteTensorToTensor();
  void InitializeTensorRefrenceCnt();
  Status AllocTensors();
  Status AllocateGraphTensors();
  std::shared_ptr<AscendDeviceInfo> GetDeviceInfo(const std::shared_ptr<Context> &context);

  std::shared_ptr<mindspore::AscendNativeDelegate> delegate_;
  std::vector<std::shared_ptr<kernel::BaseKernel>> kernels_;
  std::vector<infer::abstract::Tensor *> inputs_;
  std::vector<infer::abstract::Tensor *> outputs_;
  std::shared_ptr<infer::abstract::Context> context_;
  size_t mem_size_ = 0;
  void *memory_base_addr_ = nullptr;
  void *ascend_native_stream_ = nullptr;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_EXTENDRT_SESSION_ASCEND_NATIVE_SESSION_H_
