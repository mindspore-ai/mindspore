/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "common/backend_common_test.h"
#include "mindspore/core/ops/nn_ops.h"
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "utils/log_adapter.h"
#include "ir/graph_utils.h"
#include "pipeline/jit/ps/resource.h"
#include "include/common/debug/draw.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/optimizer.h"
#include "utils/ms_context.h"
#include "plugin/device/ascend/optimizer/mindir/ascend_vm_op_adapter.h"

#define private public
#define protected public
#include "plugin/device/ascend/optimizer/format_type/rectify_do_mask_kernel_info.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;

class TestHWRectifyDoMask : public BackendCommon {
 public:
  TestHWRectifyDoMask() : getPyFun_("gtest_input.pre_activate.rectify_do_mask_test", true) {}
  ~TestHWRectifyDoMask() override = default;
  UT::PyFuncGraphFetcher getPyFun_;

  void SetFormat(const FuncGraphPtr &g) {
    auto node_list = TopoSort(g->get_return());
    int cnt = 0;
    for (auto &node : node_list) {
      if (IsPrimitiveCNode(node, prim::kPrimDropoutDoMask)) {
        auto cnode = node->cast<CNodePtr>();
        cnt++;
        KernelBuildInfoBuilder builder;
        if (cnt == 1) {
          builder.SetInputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT});
          builder.SetOutputsFormat({kOpFormat_DEFAULT});
          node->set_kernel_info(std::make_shared<device::KernelInfo>());
          AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), node.get());
          cnode->AddPrimalAttr(kPrimalAttrUniqueId, MakeValue("1"));
        } else {
          builder.SetInputsFormat({kOpFormat_FRAC_NZ, kOpFormat_DEFAULT, kOpFormat_DEFAULT});
          builder.SetOutputsFormat({kOpFormat_FRAC_NZ});
          node->set_kernel_info(std::make_shared<device::KernelInfo>());
          AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), node.get());
          cnode->AddPrimalAttr(kPrimalAttrForwardUniqueId, MakeValue("1"));
        }
      }
    }
  }

  void CheckNQ(const FuncGraphPtr &g) {
    std::string format;
    auto node_list = TopoSort(g->get_return());
    for (auto &node : node_list) {
      if (IsPrimitiveCNode(node, prim::kPrimDropoutDoMask)) {
        if (format.empty()) {
          format = AnfAlgo::GetInputFormat(node, 0);
        } else {
          EXPECT_NE(format, AnfAlgo::GetInputFormat(node, 0));
        }
      }
    }
    ASSERT_FALSE(format.empty());
  }

  void CheckEQ(const FuncGraphPtr &g) {
    std::string format;
    auto node_list = TopoSort(g->get_return());
    for (auto &node : node_list) {
      if (IsPrimitiveCNode(node, prim::kPrimDropOutDoMask)) {
        if (format.empty()) {
          format = AnfAlgo::GetInputFormat(node, 0);
        } else {
          EXPECT_EQ(format, AnfAlgo::GetInputFormat(node, 0));
        }
      }
    }
    ASSERT_FALSE(format.empty());
  }
};

/// Feature: Test RectifyDoMaskKernelInfo pass
/// Description: Test RectifyDoMaskKernelInfo pass
/// Expectation: The forward and backward DropOutDoMask should select same format.
TEST_F(TestHWRectifyDoMask, test_rectify_dropout_do_mask) {
  /*
   * def test_rectify_dropout_do_mask(x):
   *     mask = gen_mask(shape, 0.9)
   *     res_x = do_mask(x, mask, 0.9)
   *     res_y = do_mask(y, mask, 0.9)
   *     return res_x, res_y
   *
   */
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  FuncGraphPtr fg = getPyFun_.CallAndParseRet("test_rectify_dropout_do_mask", "f");
  SetFormat(fg);
  CheckNQ(fg);
  auto graph_optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pass_manager = std::make_shared<opt::PassManager>();
  pass_manager->AddPass(std::make_shared<opt::AscendVmOpAdapter>());
  pass_manager->AddPass(std::make_shared<opt::RectifyDoMaskKernelInfo>());
  graph_optimizer->AddPassManager(pass_manager);
  auto new_g = graph_optimizer->Optimize(fg);
  CheckEQ(new_g);
}
}  // namespace opt
}  // namespace mindspore
