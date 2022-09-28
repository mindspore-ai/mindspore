/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_SESSION_SESSION_H
#define MINDSPORE_CCSRC_SESSION_SESSION_H

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <map>

#include "backend/common/session/session_basic.h"
#include "ir/anf.h"
#include "include/api/status.h"
#include "cxx_api/model/model_impl.h"

namespace mindspore {
class MsModel : public ModelImpl {
 public:
  MsModel() {}
  ~MsModel() = default;

  Status Build() override;
  Status Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) override;

  std::vector<MSTensor> GetInputs() override;
  std::vector<MSTensor> GetOutputs() override;

  bool CheckDeviceSupport(mindspore::DeviceType device_type) override;
  bool CheckModelSupport(enum ModelType model_type) override;

 private:
  std::shared_ptr<GraphCell> GenerateGraphCell(const std::vector<std::vector<int64_t>> &dims);
  uint32_t GetDeviceID() const;

  std::map<std::string, std::shared_ptr<GraphCell>> dynamic_size_graph_map_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_SESSION_SESSION_BASIC_H
