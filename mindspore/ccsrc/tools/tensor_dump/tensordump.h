/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_TOOLS_TENSOR_DUMP_TENSORDUMP_H_
#define MINDSPORE_CCSRC_TOOLS_TENSOR_DUMP_TENSORDUMP_H_

#include <array>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "ir/tensor.h"
#include "ir/tensor_new.h"
#include "tools/visible.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace datadump {

inline constexpr int kCallFromCXX = 0;
inline constexpr int kCallFromPython = 1;

class TOOLS_EXPORT TensorDumpManager {
 public:
  enum class task_type { dump = 0, skip = 1, update_step = 2, silentdetect = 3, unknown = 100 };
  static TensorDumpManager &GetInstance() {
    static TensorDumpManager instance;
    return instance;
  }
  ~TensorDumpManager() = default;
  void SetDumpStep(const std::vector<size_t> &);
  std::string ProcessFileName(const std::string &, const std::string &, const int = kCallFromCXX);
  task_type GetTaskType(const std::string &tensor_name, const int mode);
  void Exec(const std::string &, tensor::TensorPtr, const int = kCallFromCXX);
  void ExecTask(task_type, const std::string &, tensor::TensorPtr, const int);
  void SetAclDumpCallbackReg(void *);

 private:
  TensorDumpManager() = default;
  DISABLE_COPY_AND_ASSIGN(TensorDumpManager);
  void UpdateStep(const int);
  size_t GetStep(const int) const;
  bool NeedDump(const int) const;
  std::string TensorNameToArrayName(std::string, std::string, const int);
  size_t FetchAddID();
  std::atomic<size_t> id_;
  std::array<size_t, 2> step_ = {0, 0};
  std::set<size_t> valid_steps_;
  std::mutex mtx_;
};

}  // namespace datadump
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TOOLS_TENSOR_DUMP_TENSORDUMP_H_
