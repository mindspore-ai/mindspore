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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_GPU_INFER_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_GPU_INFER_EXECUTOR_H_
#include <string>
#include <vector>
#include <utility>
#include <memory>
#include "include/api/status.h"
#include "include/api/graph.h"
#include "ir/anf.h"
#include "extendrt/graph_executor.h"
namespace mindspore {
class GPUInferExecutor : public GraphExecutor {
 public:
  GPUInferSession();
  ~GPUInferSession() override = default;
  Status Execute(const ExecutePlan &plan, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) override;

 protected:
  bool CheckDeviceSupport(mindspore::DeviceType device_type) override;
  Status Load(uint32_t device_id);
  Status InitEnv();
  Status FinalizeEnv();
  Status CheckModelInputs(const std::vector<tensor::TensorPtr> &inputs) const;

 private:
  uint32_t graph_id_;
  std::string device_type_;
  uint32_t device_id_;
  std::vector<tensor::TensorPtr> inputs_info_;
  std::vector<tensor::TensorPtr> outputs_info_;
  std::vector<tensor::TensorPtr> last_inputs_;
  std::vector<tensor::TensorPtr> last_outputs_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  bool init_flag_;
  bool load_flag_;
  bool set_device_id_flag_;

  // tensor-rt
  uint32_t batch_size_;
  uint32_t workspace_size_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_GPU_INFER_EXECUTOR_H_
