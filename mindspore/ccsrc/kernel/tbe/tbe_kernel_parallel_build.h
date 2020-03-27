/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_TBE_TBE_KERNEL_PARALLEL_BUILD_H_
#define MINDSPORE_CCSRC_KERNEL_TBE_TBE_KERNEL_PARALLEL_BUILD_H_

#include <utility>
#include <string>
#include <map>
#include <vector>
#include "kernel/kernel.h"
#include "pybind11/stl.h"
#include <nlohmann/json.hpp>
namespace mindspore {
namespace kernel {
bool TbeOpParallelBuild(std::vector<AnfNodePtr> anf_nodes);

struct KernelBuildTaskInfo {
  AnfNode *node;
  std::string processor;
  std::string json_name;
  std::vector<size_t> input_size_list;
  std::vector<size_t> output_size_list;
  int32_t scope_id;
};

class ParallelBuildManager {
 public:
  ParallelBuildManager();
  ~ParallelBuildManager() = default;
  int32_t StartCompileOp(const nlohmann::json &kernel_json) const;
  void SaveTaskInfo(int32_t task_id, const AnfNodePtr &anf_node, const std::string &json_name,
                    const std::vector<size_t> &input_size_list, const std::vector<size_t> &output_size_list,
                    int32_t scope_id = 0);
  void SaveSameOpInfo(const AnfNodePtr &anf_node, const std::string &json_name,
                      const std::vector<size_t> &input_size_list, const std::vector<size_t> &output_size_list);
  bool GenSameOpKernelMod() const;
  bool SearchInCache(const std::string &json_name, const std::string &processor,
                     const std::vector<size_t> &input_size_list, const std::vector<size_t> &output_size_list,
                     AnfNode *node) const;

  bool WaitOne(int *task_id, char **task_result) const;
  bool IsAllTaskFinish() const;
  std::pair<int32_t, KernelModPtr> TaskFinishProcess(int32_t task_id, bool set_kernel_mod = true);
  KernelModPtr GenKernelMod(const string &json_name, const string &processor, const vector<size_t> &input_size_list,
                            const vector<size_t> &output_size_list, const KernelPackPtr &kernel_pack) const;

 private:
  PyObject *tbe_parallel_compiler_;
  std::map<int32_t, KernelBuildTaskInfo> task_map_;
  std::vector<KernelBuildTaskInfo> same_op_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_TBE_TBE_KERNEL_PARALLEL_BUILD_H_
