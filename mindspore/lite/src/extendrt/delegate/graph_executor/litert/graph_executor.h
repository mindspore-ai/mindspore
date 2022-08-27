/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_LITERT_GRAPH_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_LITERT_GRAPH_EXECUTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <map>

#include "runtime/hardware/device_context.h"
#include "include/api/context.h"
#include "include/model.h"
#include "src/litert/lite_session.h"

namespace mindspore {
using GraphExecutor = mindspore::device::GraphExecutor;
class LiteRTGraphExecutor : public GraphExecutor {
 public:
  LiteRTGraphExecutor() = default;
  explicit LiteRTGraphExecutor(const std::shared_ptr<mindspore::Context> &context);
  ~LiteRTGraphExecutor() = default;
  bool CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options) override;
  bool RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                std::vector<tensor::Tensor> *outputs, const std::map<string, string> &compile_options) override;
  bool Resize(const std::vector<tensor::Tensor> &inputs, const std::vector<std::vector<int64_t>> &dims);
  std::vector<tensor::TensorPtr> GetInputs();
  std::shared_ptr<lite::LiteSession> CreateLiteSession(lite::InnerContext *context);
  std::vector<MSTensor> GetLiteSessionInputs();
  std::vector<MSTensor> GetLiteSessionOutputs();
  void ResetTensorData(std::vector<void *> old_data, const std::vector<lite::Tensor *> &tensors);
  std::vector<int32_t> TruncateShape(const std::vector<int64_t> &shape, enum TypeId type, size_t data_len,
                                     bool verify_size);

 private:
  const std::shared_ptr<mindspore::Context> context_;
  lite::LiteGraph lite_graph_;
  std::shared_ptr<lite::LiteSession> lite_session_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_GRAPH_EXECUTOR_LITERT_GRAPH_EXECUTOR_H_
