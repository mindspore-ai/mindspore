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

#include <vector>
#include <map>
#include "common/common_test.h"
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "runtime/device/gsm/mem_usage_analyzer.h"

namespace mindspore::device {
class TestMemUsageAnalyzer : public BackendCommon {
 public:
  TestMemUsageAnalyzer() : get_py_func_("gtest_input.runtime.device.gsm.mem_usage_analyzer_test", true) {}

  UT::PyFuncGraphFetcher get_py_func_;
};

/// Feature: MemUsageAnalyzer
/// Description: Test MemUsageAnalyzer interface
/// Expectation: Pass all interface test
TEST_F(TestMemUsageAnalyzer, test_mem_usage_analyzer) {
  auto net = get_py_func_("add_net");
  EXPECT_NE(net, nullptr);
  std::vector<int64_t> shp_x{1, 2, 2, 2};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, x_abstract, x_abstract, x_abstract};

  auto func_graph = GetFuncGraph(net, args_spec_list);
  auto kernel_graph = Compile(func_graph);

  auto analyzer = std::make_shared<MemUsageAnalyzer>();
  analyzer->Analyze(kernel_graph);
  auto kernel_infos = analyzer->GetMemUsageKernelInfos();
  auto tensor_infos = analyzer->GetMemUsageTensorInfos();

  ASSERT_EQ(5, kernel_infos.size());
  ASSERT_EQ(15, tensor_infos.size());
  for (size_t i = 0; i < kernel_infos.size(); ++i) {
    ASSERT_NE(nullptr, analyzer->GetMemUsageKernelInfo(i));
  }

  for (size_t i = 0; i < tensor_infos.size(); ++i) {
    ASSERT_NE(nullptr, analyzer->GetMemUsageTensorInfo(i));
  }

  ASSERT_EQ(100, analyzer->LeastMemNeeded());
}

/// Feature: MemUsageAnalyzer
/// Description: Test MemUsageAnalyzer interface with allreduce node
/// Expectation: Pass all interface test
TEST_F(TestMemUsageAnalyzer, test_mem_usage_analyzer_fused_tesnor) {
  auto net = get_py_func_("all_reduce_net");
  EXPECT_NE(net, nullptr);
  std::vector<int64_t> shp_x{2, 2, 2, 2};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, x_abstract};

  auto func_graph = GetFuncGraph(net, args_spec_list);
  auto kernel_graph = Compile(func_graph);

  auto analyzer = std::make_shared<MemUsageAnalyzer>();
  analyzer->Analyze(kernel_graph);
  auto kernel_infos = analyzer->GetMemUsageKernelInfos();
  auto tensor_infos = analyzer->GetMemUsageTensorInfos();

  ASSERT_EQ(4, kernel_infos.size());
  ASSERT_EQ(14, tensor_infos.size());
  ASSERT_EQ(260, analyzer->LeastMemNeeded());

  size_t comm_kernel_num = 0;
  for (size_t i = 0; i < kernel_infos.size(); ++i) {
    auto kernel_info = analyzer->GetMemUsageKernelInfo(i);
    ASSERT_NE(nullptr, kernel_info);
    if (kernel_info->is_comm_) {
      ++comm_kernel_num;
    }
  }
  ASSERT_EQ(1, comm_kernel_num);

  size_t tensor_max_used = 0;
  size_t fused_tensor_size = 0;
  for (size_t i = 0; i < tensor_infos.size(); ++i) {
    auto tensor_info = analyzer->GetMemUsageTensorInfo(i);
    ASSERT_NE(nullptr, tensor_info);
    auto used_size = tensor_info->used_by_kernels_.size();
    tensor_max_used = tensor_max_used < used_size ? used_size : tensor_max_used;
    if (tensor_info->fused_tensor_ids_.size() > 0) {
      ++fused_tensor_size;
    }
  }
  ASSERT_EQ(2, tensor_max_used);
  ASSERT_EQ(2, fused_tensor_size);
}
}  // namespace mindspore::device