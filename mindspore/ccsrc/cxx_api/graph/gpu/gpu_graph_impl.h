/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_CXX_API_GRAPH_MS_GPU_GRAPH_IMPL_H
#define MINDSPORE_CCSRC_CXX_API_GRAPH_MS_GPU_GRAPH_IMPL_H
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include "include/api/status.h"
#include "include/api/graph.h"
#include "cxx_api/graph/graph_impl.h"
#include "backend/session/session_basic.h"
#include "ir/anf.h"
#include "cxx_api/model/model_impl.h"
#include "cxx_api/graph/graph_utils.h"

namespace mindspore::api {
class GPUGraphImpl : public GraphCell::GraphImpl {
 public:
  GPUGraphImpl();
  ~GPUGraphImpl() override = default;

  Status Run(const std::vector<Buffer> &inputs, std::vector<Buffer> *outputs) override;
  Status Load() override;
  Status GetInputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                       std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) override;
  Status GetOutputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                        std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) override;

 private:
  Status InitEnv();
  Status FinalizeEnv();
  Status CompileGraph(const std::shared_ptr<FuncGraph> &funcGraphPtr);
  Status CheckModelInputs(const std::vector<tensor::TensorPtr> &inputs) const;
  std::vector<tensor::TensorPtr> RunGraph(const std::vector<tensor::TensorPtr> &inputs);
  Status ExecuteModel(const std::vector<Buffer> &inputs, std::vector<Buffer> *outputs);

  std::shared_ptr<session::SessionBasic> session_impl_;
  uint32_t graph_id_;
  std::string device_type_;
  uint32_t device_id_;
  std::vector<tensor::TensorPtr> inputs_;
  std::vector<tensor::TensorPtr> outputs_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  bool init_flag_;
  bool load_flag_;

  // tensor-rt
  uint32_t batch_size_;
  uint32_t workspace_size_;
};
}  // namespace mindspore::api
#endif  // MINDSPORE_CCSRC_CXX_API_GRAPH_MS_GPU_GRAPH_IMPL_H
