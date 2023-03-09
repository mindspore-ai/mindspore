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

#ifndef MINDSPORE_LITE_TEST_UT_TOOLS_OPTIMIZER_FUSION_FUSION_INOUT_TEST_FUSION_INOUT_TEST_H_
#define MINDSPORE_LITE_TEST_UT_TOOLS_OPTIMIZER_FUSION_FUSION_INOUT_TEST_FUSION_INOUT_TEST_H_

#include <string>
#include <vector>
#include "common/common_test.h"
#include "ir/anf.h"
#include "include/backend/optimizer/pass.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pass_manager.h"

namespace mindspore {
class FusionInoutTest : public mindspore::CommonTest {
 public:
  FusionInoutTest() = default;

  bool DoTest();

 protected:
  FuncGraphPtr Fuse();

  std::vector<std::string> GetInputNames();

  size_t GetOutputNumber();

  virtual void InitPass() = 0;

  virtual void InitGraph() = 0;

  static ParameterPtr AddParameter(const FuncGraphPtr &graph, size_t data_size, const std::vector<int64_t> &shape,
                                   TypeId data_type, const std::string &name);

  static CNodePtr AddReturn(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &return_inputs);

 protected:
  opt::PassPtr pass_ = nullptr;
  FuncGraphPtr graph_ = nullptr;
};
}  // namespace mindspore
#endif
