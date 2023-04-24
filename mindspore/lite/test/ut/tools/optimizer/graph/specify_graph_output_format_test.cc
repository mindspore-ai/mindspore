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

#define USE_DEPRECATED_API
#include <memory>
#include <algorithm>
#include "common/common_test.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "tools/optimizer/graph/specify_graph_output_format.h"
#include "test/ut/utils/build_func_graph.h"
#include "tools/optimizer/common/format_utils.h"
#include "base/base.h"
#include "ir/dtype/number.h"
#include "mindspore/lite/tools/common/graph_util.h"
#include "ops/fusion/conv2d_fusion.h"

namespace mindspore {
namespace lite {
class SpecifyGraphOutputFormatTest : public mindspore::CommonTest {
 public:
  SpecifyGraphOutputFormatTest() = default;
  CNodePtr AddConv(const FuncGraphPtr &graph, const ShapeVector &shape, const Format &format) {
    auto prim = std::make_unique<ops::Conv2DFusion>();
    auto value_node = NewValueNode(prim->GetPrim());
    auto conv = graph->NewCNode({value_node, graph->add_parameter(), graph->add_parameter(), graph->add_parameter()});
    SetOutputFormat(conv, {format});
    auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shape);
    conv->set_abstract(abstract);
    return conv;
  }
  static void SetOutputFormat(const CNodePtr &cnode, const std::vector<Format> &formats) {
    MS_ASSERT(cnode != nullptr);
    auto prim = GetCNodePrimitive(cnode);
    std::vector<int64_t> int_formats;
    for (const auto &format : formats) {
      int_formats.emplace_back(format);
    }
    prim->AddAttr(opt::kOutputsFormat, MakeValue(int_formats));
  }

  std::vector<std::pair<AnfNodePtr, int64_t>> outputs_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<int64_t>> output_dims_;
};

TEST_F(SpecifyGraphOutputFormatTest, TestNHWCOutputFormat) {
  auto func_graph = std::make_shared<FuncGraph>();
  auto conv = AddConv(func_graph, {1, 4, 9, 9}, NCHW);
  auto ret = lite::AddReturn(func_graph, {conv});

  auto pass = opt::SpecifyGraphOutputFormat(NHWC);
  auto status = pass.Run(func_graph);
  ASSERT_TRUE(status);
  GetFuncGraphOutputsInfo(func_graph, &outputs_, &output_names_, &output_dims_);
  std::vector<int64_t> expected_shape{1, 9, 9, 4};
  ASSERT_TRUE(output_dims_[0] == expected_shape);
}

TEST_F(SpecifyGraphOutputFormatTest, TestNCHWOutputFormat) {
  auto func_graph = std::make_shared<FuncGraph>();
  auto conv = AddConv(func_graph, {1, 9, 9, 4}, NHWC);
  auto ret = lite::AddReturn(func_graph, {conv});

  auto pass = opt::SpecifyGraphOutputFormat(NCHW);
  auto status = pass.Run(func_graph);
  ASSERT_TRUE(status);
  GetFuncGraphOutputsInfo(func_graph, &outputs_, &output_names_, &output_dims_);
  std::vector<int64_t> expected_shape{1, 4, 9, 9};
  ASSERT_TRUE(output_dims_[0] == expected_shape);
}

TEST_F(SpecifyGraphOutputFormatTest, TestTwoNCHWOutputFormat) {
  auto func_graph = std::make_shared<FuncGraph>();
  auto conv0 = AddConv(func_graph, {1, 9, 9, 4}, NHWC);
  auto conv1 = AddConv(func_graph, {1, 9, 9, 4}, NHWC);
  auto ret = lite::AddReturn(func_graph, {conv0, conv1});

  auto pass = opt::SpecifyGraphOutputFormat(NCHW);
  auto status = pass.Run(func_graph);
  ASSERT_TRUE(status);
  GetFuncGraphOutputsInfo(func_graph, &outputs_, &output_names_, &output_dims_);
  ASSERT_EQ(output_dims_.size(), 2);
  std::vector<int64_t> expected_shape{1, 4, 9, 9};
  ASSERT_TRUE(output_dims_[0] == expected_shape);
  ASSERT_TRUE(output_dims_[1] == expected_shape);
}

TEST_F(SpecifyGraphOutputFormatTest, TestNotNCHWOrNHWCOutputFormat) {
  auto func_graph = std::make_shared<FuncGraph>();
  auto conv = AddConv(func_graph, {1, 9, 9, 4}, KHWC);
  auto ret = lite::AddReturn(func_graph, {conv});

  auto pass = opt::SpecifyGraphOutputFormat(NCHW);
  auto status = pass.Run(func_graph);
  ASSERT_TRUE(status);
  GetFuncGraphOutputsInfo(func_graph, &outputs_, &output_names_, &output_dims_);
  std::vector<int64_t> expected_shape{1, 9, 9, 4};
  ASSERT_TRUE(output_dims_[0] == expected_shape);
}
}  // namespace lite
}  // namespace mindspore
