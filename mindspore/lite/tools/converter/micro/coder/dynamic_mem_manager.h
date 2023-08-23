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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_DYNAMIC_MEM_MANAGER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_DYNAMIC_MEM_MANAGER_H_

#include <map>
#include <vector>
#include <memory>
#include <string>
#include "src/tensor.h"
#include "tools/converter/micro/coder/shape_info_container.h"

namespace mindspore::lite::micro {
class DynamicMemManager {
 public:
  DynamicMemManager() = default;
  virtual ~DynamicMemManager() = default;
  int AllocDynamicMem(const std::vector<std::unique_ptr<OperatorCoder>> &nodes,
                      const std::vector<Tensor *> &graph_inputs, const std::vector<Tensor *> &graph_outputs,
                      const ShapeInfoContainer *shape_info_container);

  std::string GetVarTensorAddr(const Tensor *tensor) const;
  std::string AllocWorkSpace(size_t size, int index);
  const std::map<int, std::vector<size_t>> &GetOffsetAllScenes() { return offsets_all_scenes_; }
  const std::vector<size_t> GetBufferSizes() { return buffer_sizes_; }
  const std::vector<size_t> GetWorkSpaces() { return workspaces_; }

 private:
  int AllocDynamicMemCore(const std::vector<std::unique_ptr<OperatorCoder>> &nodes,
                          const std::vector<Tensor *> &graph_outputs, int scene_index);
  std::map<int, std::vector<size_t>> offsets_all_scenes_;
  std::map<const Tensor *, int> offset_index_;
  std::map<const Tensor *, std::string> graph_inputs_;
  std::vector<size_t> buffer_sizes_;
  std::vector<size_t> workspaces_;
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_DYNAMIC_MEM_MANAGER_H_
