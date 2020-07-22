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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_UTIL_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_UTIL_H_

#include <map>
#include <string>
#include <unordered_map>
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace parallel {
namespace ps {
class Util {
 public:
  static bool IsParamServerMode();
  static bool IsRoleOfWorker();
  static bool IsRoleOfPServer();
  static bool IsRoleOfScheduler();
  static void SetInternalEnvVar();
  static int optimizer_id(std::string name);
  static std::string optimizer_name(int id);
  static std::string optimizer_node_name(int id);
  static bool is_optimizer(std::string name);
  static int LocalShard(int first_dim, int rank_id, int server_num);

 private:
  static std::unordered_map<std::string, int> optimizer_to_ids;
  static std::unordered_map<int, std::string> id_to_optimizers;
  static std::unordered_map<int, std::string> id_to_optimizer_nodes;
};
}  // namespace ps
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PS_UTIL_H_
