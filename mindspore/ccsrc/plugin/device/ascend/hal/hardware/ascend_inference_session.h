/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_INFERENCE_SESSION_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_INFERENCE_SESSION_H_
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <tuple>
#include <set>
#include "utils/hash_map.h"
#include "plugin/device/ascend/hal/hardware/ascend_session.h"
#include "include/backend/kernel_graph.h"
#include "kernel/kernel.h"
#include "backend/common/session/session_factory.h"

namespace mindspore::session {
class AscendInferenceSession : public AscendSession {
 public:
  AscendInferenceSession() = default;
  ~AscendInferenceSession() override = default;
  void LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                     const std::vector<tensor::TensorPtr> &inputs_const) const override;
  bool CheckModelInputs(uint32_t graph_id, const std::vector<tensor::TensorPtr> &inputs,
                        std::string *error_msg) const override;
  bool CompareInput(const tensor::TensorPtr &input, const ParameterPtr &parameter) const;
  template <typename T>
  std::string PrintInputShape(std::vector<T> shape) const;
  std::string InputsInfo(const std::vector<ParameterPtr> &paras, const std::vector<tensor::TensorPtr> &inputs) const;

 protected:
  GraphId CompileGraphImpl(NotNull<FuncGraphPtr> func_graph) override;
};
MS_REG_SESSION(kDavinciInferenceDevice, AscendInferenceSession);
}  // namespace mindspore::session
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_INFERENCE_SESSION_H_
