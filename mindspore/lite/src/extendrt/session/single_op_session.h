/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_EXTENDRT_SINGLE_OP_SESSION_H_
#define MINDSPORE_LITE_EXTENDRT_SINGLE_OP_SESSION_H_

#include <string>
#include <memory>
#include <vector>
#include <tuple>
#include <unordered_map>
#include "src/extendrt/infer_session.h"
#include "mindspore/ccsrc/kernel/framework_utils.h"

namespace mindspore {
/// \brief Single Op Session implementation, used in Ascend Device Context.
class SingleOpInferSession : public InferSession {
 public:
  SingleOpInferSession() = default;
  ~SingleOpInferSession() override = default;
  Status Init(const std::shared_ptr<Context> &context, const ConfigInfos &config_info = {}) override;
  Status AscendInit(const std::shared_ptr<Context> &context);
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
  void SetConfigInfo(const ConfigInfos &config_infos) { config_infos_ = config_infos; }
  void SetCustomAscendOpAttrs(const kernel::BaseOperatorPtr &op);

 protected:
  Status OnNewInputShapes(const std::vector<ShapeVector> &new_shapes);
  Status BuildCustomAscendKernel(const CNodePtr &node);
  std::tuple<kernel::KernelModPtr, kernel::KernelArgs> BuildCustomAscendKernelImpl(const CNodePtr &node);
  Status InitInputOutputInfos(const FuncGraphPtr &graph);
  void SetBackOutputIfDynamic(std::vector<tensor::Tensor> *outputs);
  Status InitInputOutputData(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs);

  std::vector<MutableTensorImplPtr> inputs_;
  std::vector<std::string> input_names_;
  std::vector<MutableTensorImplPtr> outputs_;
  std::vector<std::string> output_names_;
  uint32_t device_id_ = 0;
  std::vector<bool> dyn_outshape_;

  kernel::KernelModPtr kernel_mod_ = nullptr;
  kernel::KernelArgs kernel_args_;
  ConfigInfos config_infos_;
  bool is_multi_model_sharing_mem_prepare_ = false;

  std::unordered_map<MutableTensorImplPtr, size_t> malloced_data_size_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_EXTENDRT_SINGLE_OP_SESSION_H_
