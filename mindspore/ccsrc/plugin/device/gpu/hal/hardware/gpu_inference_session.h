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
#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_GPU_INFERENCE_SESSION_H
#define MINDSPORE_CCSRC_BACKEND_SESSION_GPU_INFERENCE_SESSION_H
#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <stack>
#include <map>
#include <tuple>
#include <set>
#include "utils/hash_map.h"
#include "plugin/device/gpu/hal/hardware/gpu_session.h"
#include "include/backend/kernel_graph.h"
#include "kernel/kernel.h"
#include "backend/common/session/session_factory.h"

namespace mindspore {
namespace session {
class GpuInferenceSession : public gpu::GPUSession {
 public:
  GpuInferenceSession() = default;
  ~GpuInferenceSession() = default;
  void LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                     const std::vector<tensor::TensorPtr> &inputs_const) const;
  bool CheckModelInputs(uint32_t graph_id, const std::vector<tensor::TensorPtr> &inputs,
                        std::string *error_msg) const override;
  bool CompareInput(const tensor::TensorPtr &input, const ParameterPtr &parameter) const;
  template <typename T>
  std::string PrintInputShape(std::vector<T> shape) const;
  std::string InputsInfo(const std::vector<ParameterPtr> &paras, const std::vector<tensor::TensorPtr> &inputs) const;

 protected:
  GraphId CompileGraphImpl(NotNull<FuncGraphPtr> func_graph) override;
};
MS_REG_SESSION(kGpuInferenceDevice, GpuInferenceSession);
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_GPU_INFERENCE_SESSION_H
