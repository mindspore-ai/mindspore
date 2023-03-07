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
#ifndef TESTS_UT_CPP_COMMON_UT_BACKEND_COMMON_H_
#define TESTS_UT_CPP_COMMON_UT_BACKEND_COMMON_H_
#include "common/common_test.h"
#include "include/backend/kernel_graph.h"

namespace mindspore {
class BackendCommon : public UT::Common {
 public:
  BackendCommon() = default;
  ~BackendCommon() override = default;
  void PrintGraphNodeList(const FuncGraphPtr &func_graph);
  virtual bool CheckEqualGraph(const FuncGraphPtr &a, const FuncGraphPtr &b);
  virtual std::shared_ptr<session::KernelGraph> GetKernelGraph(const FuncGraphPtr &func_graph,
                                                               const AbstractBasePtrList &args_spec_list,
                                                               bool need_infer = true);
  virtual FuncGraphPtr GetFuncGraph(const FuncGraphPtr &func_graph, const AbstractBasePtrList &args_spec_list);

  virtual std::shared_ptr<session::KernelGraph> Compile(const FuncGraphPtr &func_graph);
};
}  // namespace mindspore
#endif  // TESTS_UT_CPP_COMMON_UT_BACKEND_COMMON_H_
