/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_EXECUTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <map>

#include "include/api/context.h"
#include "include/model.h"
#include "include/transform/graph_ir/types.h"
#include "extendrt/session/lite_graph_executor.h"
#include "common/config_infos.h"
#include "include/transform/graph_ir/utils.h"

namespace mindspore {
class GeGraphExecutor : public LiteGraphExecutor {
 public:
  GeGraphExecutor(const std::shared_ptr<mindspore::Context> &context, const ConfigInfos &config_infos)
      : context_(context), config_infos_(config_infos) {}
  ~GeGraphExecutor();

  bool CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options,
                    uint32_t *graph_id) override;

  bool RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs,
                const std::map<string, string> &compile_options) override;

  bool Resize(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
              const std::vector<ShapeVector> &dims) override {
    return true;
  }

  std::vector<tensor::Tensor> GetInputInfos(uint32_t graph_id) override;
  std::vector<tensor::Tensor> GetOutputInfos(uint32_t graph_id) override;

 private:
  const std::shared_ptr<mindspore::Context> context_;
  ConfigInfos config_infos_;
  std::shared_ptr<ge::Session> ge_session_ = nullptr;
  std::vector<uint32_t> init_graph_id_list_;
  std::vector<uint32_t> compute_graph_id_list_;

  std::shared_ptr<AscendDeviceInfo> GetAscendDeviceInfo();
  void GetGeGraphOptions(const FuncGraphPtr &anf_graph, std::map<std::string, std::string> *ge_options);
  void GetGeSessionOptions(std::map<std::string, std::string> *ge_options);
  bool CreateSession();

  bool AddGraph(const transform::DfGraphPtr &graph, const std::map<std::string, std::string> &options,
                uint32_t *graph_id);
  bool RunGeInitGraph(uint32_t init_graph_id, const std::vector<tensor::TensorPtr> &init_tensors);
  tensor::TensorPtr ConvertGeTensorNoCopy(::ge::Tensor *ge_tensor_ptr);

  static std::atomic_uint32_t global_graph_idx_;
  static uint32_t GetNextGraphIdx();

  bool is_data_flow_graph_ = false;
  bool RunGeGraphAsync(uint32_t graph_id, const std::vector<::ge::Tensor> &inputs, std::vector<::ge::Tensor> *outputs);
  bool RunDataFlowGraphAsync(uint32_t graph_id, const std::vector<::ge::Tensor> &inputs,
                             std::vector<::ge::Tensor> *outputs);
  std::map<uint32_t, std::vector<tensor::Tensor>> graph_inputs_;
  std::map<uint32_t, std::vector<tensor::Tensor>> graph_outputs_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_GRAPH_EXECUTOR_H_
