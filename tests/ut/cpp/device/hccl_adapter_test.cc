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
#include <memory>
#include "common/common_test.h"
#include "plugin/device/ascend/hal/hccl_adapter/all_to_all_v_calc_param.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ir/dtype/type_id.h"

namespace mindspore::hccl {
class TestHcclAdapter : public UT::Common {
 public:
  TestHcclAdapter() {}

 protected:
  CNodePtr CreateAllToAllvNode(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> inputs,
                               const std::vector<int64_t> &send_rank_ids, const std::vector<int64_t> &recv_rank_ids) {
    MS_EXCEPTION_IF_NULL(graph);
    std::vector<AnfNodePtr> all_to_all_v_input = {NewValueNode(std::make_shared<Primitive>(kAllToAllvOpName))};
    all_to_all_v_input.insert(all_to_all_v_input.end(), inputs.begin(), inputs.end());
    auto all_to_all_v = graph->NewCNode(all_to_all_v_input);
    MS_EXCEPTION_IF_NULL(all_to_all_v);
    common::AnfAlgo::SetNodeAttr(kAttrSendRankIds, MakeValue<std::vector<int64_t>>(send_rank_ids), all_to_all_v);
    common::AnfAlgo::SetNodeAttr(kAttrRecvRankIds, MakeValue<std::vector<int64_t>>(recv_rank_ids), all_to_all_v);
    common::AnfAlgo::SetNodeAttr(kAttrGroup, MakeValue<std::string>("default_group"), all_to_all_v);
    return all_to_all_v;
  }

  void SetOutputs(const CNodePtr &cnode, const std::vector<ShapeVector> &shape, const std::vector<TypeId> &data_type) {
    common::AnfAlgo::SetOutputInferTypeAndShape(data_type, shape, cnode.get());
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    builder.SetFusionType(kernel::kPatternOpaque);
    builder.SetProcessor(kernel::Processor::AICORE);
    builder.SetKernelType(TBE_KERNEL);
    builder.SetInputsFormat(std::vector<std::string>(cnode->size() - 1, format_));
    builder.SetOutputsFormat(std::vector<std::string>(shape.size(), format_));
    builder.SetInputsDeviceType(std::vector<TypeId>(cnode->size() - 1, type_));
    builder.SetOutputsDeviceType(std::vector<TypeId>(shape.size(), type_));
    cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cnode.get());
  }

  std::vector<AnfNodePtr> CreateInputs(const FuncGraphPtr &graph, const std::vector<ShapeVector> &shape,
                                       const std::vector<TypeId> &data_type) {
    MS_EXCEPTION_IF_NULL(graph);
    if (shape.size() != data_type.size()) {
      return {};
    }
    std::vector<AnfNodePtr> res;
    for (size_t i = 0; i < shape.size(); ++i) {
      auto node = graph->NewCNode(std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>("AnyNameOp"))});
      common::AnfAlgo::SetOutputInferTypeAndShape(std::vector<TypeId>{data_type[i]}, std::vector<ShapeVector>{shape[i]},
                                                  node.get());
      kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
      builder.SetFusionType(kernel::kPatternOpaque);
      builder.SetProcessor(kernel::Processor::AICORE);
      builder.SetKernelType(TBE_KERNEL);
      builder.SetInputsFormat({format_});
      builder.SetOutputsFormat({format_});
      builder.SetInputsDeviceType({type_});
      builder.SetOutputsDeviceType({type_});
      node->set_kernel_info(std::make_shared<device::KernelInfo>());
      AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), node.get());
      res.emplace_back(node);
    }
    return res;
  }

  TypeId type_ = TypeId::kNumberTypeInt32;
  std::string format_ = "NCHW";
};

/// Feature: AllToAllvCalcParam
/// Description: on 2p, send to rank 1, and recv nothing
/// Expectation: send count 0 1
///             send offset 0 0
///              recv count 0 0
///             recv offset 0 0
TEST_F(TestHcclAdapter, test_all_to_all_v_calc_param_2p_only_send) {
  auto graph = std::make_shared<FuncGraph>();
  ASSERT_TRUE(graph != nullptr);
  uint32_t rank_size = 2;
  std::vector<int64_t> send_rank_ids = {1};
  std::vector<int64_t> recv_rank_ids = {};
  auto alltoall = CreateAllToAllvNode(graph, CreateInputs(graph, {{1}}, {type_}), send_rank_ids, recv_rank_ids);
  ASSERT_TRUE(alltoall != nullptr);
  ASSERT_NO_THROW(SetOutputs(alltoall, {}, {}));
  AllToAllvCalcParam calc(alltoall, rank_size);
  ASSERT_NO_THROW(calc.CalcOpParam());
  EXPECT_EQ(calc.GetSendCounts(), std::vector<int64_t>({0, 1}));
  EXPECT_EQ(calc.GetSendDispls(), std::vector<int64_t>({0, 0}));
  EXPECT_EQ(calc.GetRecvCounts(), std::vector<int64_t>({0, 0}));
  EXPECT_EQ(calc.GetRecvDispls(), std::vector<int64_t>({0, 0}));
}

/// Feature: AllToAllvCalcParam
/// Description: on 2p, send nothing, and recv from rank 0 and rank 1
/// Expectation: send count 0 0
///             send offset 0 0
///              recv count 1 1
///             recv offset 0 128
TEST_F(TestHcclAdapter, test_all_to_all_v_calc_param_2p_only_recv) {
  auto graph = std::make_shared<FuncGraph>();
  ASSERT_TRUE(graph != nullptr);
  uint32_t rank_size = 2;
  std::vector<int64_t> send_rank_ids = {};
  std::vector<int64_t> recv_rank_ids = {0, 1};
  auto alltoall = CreateAllToAllvNode(graph, CreateInputs(graph, {}, {}), send_rank_ids, recv_rank_ids);
  ASSERT_TRUE(alltoall != nullptr);
  ASSERT_NO_THROW(SetOutputs(alltoall, {{1}, {1}}, {type_, type_}));
  AllToAllvCalcParam calc(alltoall, rank_size);
  ASSERT_NO_THROW(calc.CalcOpParam());
  EXPECT_EQ(calc.GetSendCounts(), std::vector<int64_t>({0, 0}));
  EXPECT_EQ(calc.GetSendDispls(), std::vector<int64_t>({0, 0}));
  EXPECT_EQ(calc.GetRecvCounts(), std::vector<int64_t>({1, 1}));
  EXPECT_EQ(calc.GetRecvDispls(), std::vector<int64_t>({0, 128}));
}

/// Feature: AllToAllvCalcParam
/// Description: on 4p, send to rank1,2,3, and recv nothing
/// Expectation: send count 0 1 1 1
///             send offset 0 0 128 256
///              recv count 0 0 0 0
///             recv offset 0 0 0 0
TEST_F(TestHcclAdapter, test_all_to_all_v_calc_param_4p_only_send) {
  auto graph = std::make_shared<FuncGraph>();
  ASSERT_TRUE(graph != nullptr);
  uint32_t rank_size = 4;
  std::vector<int64_t> send_rank_ids = {1, 2, 3};
  std::vector<int64_t> recv_rank_ids = {};
  auto alltoall = CreateAllToAllvNode(graph, CreateInputs(graph, {{1}, {1}, {1}}, {type_, type_, type_}), send_rank_ids,
                                      recv_rank_ids);
  ASSERT_TRUE(alltoall != nullptr);
  ASSERT_NO_THROW(SetOutputs(alltoall, {}, {}));
  AllToAllvCalcParam calc(alltoall, rank_size);
  ASSERT_NO_THROW(calc.CalcOpParam());
  EXPECT_EQ(calc.GetSendCounts(), std::vector<int64_t>({0, 1, 1, 1}));
  EXPECT_EQ(calc.GetSendDispls(), std::vector<int64_t>({0, 0, 128, 256}));
  EXPECT_EQ(calc.GetRecvCounts(), std::vector<int64_t>({0, 0, 0, 0}));
  EXPECT_EQ(calc.GetRecvDispls(), std::vector<int64_t>({0, 0, 0, 0}));
}

/// Feature: AllToAllvCalcParam
/// Description: on 4p, send to rank1,3, and recv nothing
/// Expectation: send count 0 1 0 1
///             send offset 0 0 128 128
///              recv count 0 0 0 0
///             recv offset 0 0 0 0
TEST_F(TestHcclAdapter, test_all_to_all_v_calc_param_4p_only_send_2) {
  auto graph = std::make_shared<FuncGraph>();
  ASSERT_TRUE(graph != nullptr);
  uint32_t rank_size = 4;
  std::vector<int64_t> send_rank_ids = {1, 3};
  std::vector<int64_t> recv_rank_ids = {};
  auto alltoall =
    CreateAllToAllvNode(graph, CreateInputs(graph, {{1}, {1}}, {type_, type_}), send_rank_ids, recv_rank_ids);
  ASSERT_TRUE(alltoall != nullptr);
  ASSERT_NO_THROW(SetOutputs(alltoall, {}, {}));
  AllToAllvCalcParam calc(alltoall, rank_size);
  ASSERT_NO_THROW(calc.CalcOpParam());
  EXPECT_EQ(calc.GetSendCounts(), std::vector<int64_t>({0, 1, 0, 1}));
  EXPECT_EQ(calc.GetSendDispls(), std::vector<int64_t>({0, 0, 128, 128}));
  EXPECT_EQ(calc.GetRecvCounts(), std::vector<int64_t>({0, 0, 0, 0}));
  EXPECT_EQ(calc.GetRecvDispls(), std::vector<int64_t>({0, 0, 0, 0}));
}

/// Feature: AllToAllvCalcParam
/// Description: on 2p, send to rank1, and recv from rank1
/// Expectation: send count 0 1
///             send offset 0 0
///              recv count 0 1
///             recv offset 0 0
TEST_F(TestHcclAdapter, test_all_to_all_v_calc_param_2p_exchange) {
  auto graph = std::make_shared<FuncGraph>();
  ASSERT_TRUE(graph != nullptr);
  uint32_t rank_size = 2;
  std::vector<int64_t> send_rank_ids = {1};
  std::vector<int64_t> recv_rank_ids = {1};
  auto alltoall = CreateAllToAllvNode(graph, CreateInputs(graph, {{1}}, {type_}), send_rank_ids, recv_rank_ids);
  ASSERT_TRUE(alltoall != nullptr);
  ASSERT_NO_THROW(SetOutputs(alltoall, {{1}}, {type_}));
  AllToAllvCalcParam calc(alltoall, rank_size);
  ASSERT_NO_THROW(calc.CalcOpParam());
  EXPECT_EQ(calc.GetSendCounts(), std::vector<int64_t>({0, 1}));
  EXPECT_EQ(calc.GetSendDispls(), std::vector<int64_t>({0, 0}));
  EXPECT_EQ(calc.GetRecvCounts(), std::vector<int64_t>({0, 1}));
  EXPECT_EQ(calc.GetRecvDispls(), std::vector<int64_t>({0, 0}));
}

/// Feature: AllToAllvCalcParam
/// Description: on 2p, send to rank0, and recv from rank0
/// Expectation: send count 1 0
///             send offset 0 128
///              recv count 1 0
///             recv offset 0 128
TEST_F(TestHcclAdapter, test_all_to_all_v_calc_param_2p_send_to_self) {
  auto graph = std::make_shared<FuncGraph>();
  ASSERT_TRUE(graph != nullptr);
  uint32_t rank_size = 2;
  std::vector<int64_t> send_rank_ids = {0};
  std::vector<int64_t> recv_rank_ids = {0};
  auto alltoall = CreateAllToAllvNode(graph, CreateInputs(graph, {{1}}, {type_}), send_rank_ids, recv_rank_ids);
  ASSERT_TRUE(alltoall != nullptr);
  ASSERT_NO_THROW(SetOutputs(alltoall, {{1}}, {type_}));
  AllToAllvCalcParam calc(alltoall, rank_size);
  ASSERT_NO_THROW(calc.CalcOpParam());
  EXPECT_EQ(calc.GetSendCounts(), std::vector<int64_t>({1, 0}));
  EXPECT_EQ(calc.GetSendDispls(), std::vector<int64_t>({0, 128}));
  EXPECT_EQ(calc.GetRecvCounts(), std::vector<int64_t>({1, 0}));
  EXPECT_EQ(calc.GetRecvDispls(), std::vector<int64_t>({0, 128}));
}

/// Feature: AllToAllvCalcParam
/// Description: on 4p, send to rank0123, and recv from rank0123
/// Expectation: send count 1 1 1 1
///             send offset 0 128 256 384
///              recv count 1 1 1 1
///             recv offset 0 128 256 384
TEST_F(TestHcclAdapter, test_all_to_all_v_calc_param_4p_all_to_all) {
  auto graph = std::make_shared<FuncGraph>();
  ASSERT_TRUE(graph != nullptr);
  uint32_t rank_size = 4;
  std::vector<int64_t> send_rank_ids = {0, 1, 2, 3};
  std::vector<int64_t> recv_rank_ids = {0, 1, 2, 3};
  auto alltoall = CreateAllToAllvNode(graph, CreateInputs(graph, {{1}, {1}, {1}, {1}}, {type_, type_, type_, type_}),
                                      send_rank_ids, recv_rank_ids);
  ASSERT_TRUE(alltoall != nullptr);
  ASSERT_NO_THROW(SetOutputs(alltoall, {{1}, {1}, {1}, {1}}, {type_, type_, type_, type_}));
  AllToAllvCalcParam calc(alltoall, rank_size);
  ASSERT_NO_THROW(calc.CalcOpParam());
  EXPECT_EQ(calc.GetSendCounts(), std::vector<int64_t>({1, 1, 1, 1}));
  EXPECT_EQ(calc.GetSendDispls(), std::vector<int64_t>({0, 128, 256, 384}));
  EXPECT_EQ(calc.GetRecvCounts(), std::vector<int64_t>({1, 1, 1, 1}));
  EXPECT_EQ(calc.GetRecvDispls(), std::vector<int64_t>({0, 128, 256, 384}));
}

/// Feature: AllToAllvCalcParam
/// Description: on 4p, send to rank0123, and recv from rank0123, but recv order is wrong
/// Expectation: send count 1 1 1 1
///             send offset 0 128 256 384
///              recv count 1 1 1 1
///             recv offset 256 128 384 0
TEST_F(TestHcclAdapter, test_all_to_all_v_calc_param_4p_all_in_all_in_wrong_order) {
  auto graph = std::make_shared<FuncGraph>();
  ASSERT_TRUE(graph != nullptr);
  uint32_t rank_size = 4;
  std::vector<int64_t> send_rank_ids = {0, 1, 2, 3};
  std::vector<int64_t> recv_rank_ids = {3, 1, 0, 2};
  auto alltoall = CreateAllToAllvNode(graph, CreateInputs(graph, {{1}, {1}, {1}, {1}}, {type_, type_, type_, type_}),
                                      send_rank_ids, recv_rank_ids);
  ASSERT_TRUE(alltoall != nullptr);
  ASSERT_NO_THROW(SetOutputs(alltoall, {{1}, {1}, {1}, {1}}, {type_, type_, type_, type_}));
  AllToAllvCalcParam calc(alltoall, rank_size);
  ASSERT_NO_THROW(calc.CalcOpParam());
  EXPECT_EQ(calc.GetSendCounts(), std::vector<int64_t>({1, 1, 1, 1}));
  EXPECT_EQ(calc.GetSendDispls(), std::vector<int64_t>({0, 128, 256, 384}));
  EXPECT_EQ(calc.GetRecvCounts(), std::vector<int64_t>({1, 1, 1, 1}));
  EXPECT_EQ(calc.GetRecvDispls(), std::vector<int64_t>({256, 128, 384, 0}));
}

/// Feature: AllToAllvCalcParam
/// Description: on 4p, send to rank123, and recv from nothing, but send order is wrong
/// Expectation: send count 0 1 1 1
///             send offset 0 128 256 0
///              recv count 0 0 0 0
///             recv offset 0 0 0 0
TEST_F(TestHcclAdapter, test_all_to_all_v_calc_param_4p_only_send_in_wrong_order) {
  auto graph = std::make_shared<FuncGraph>();
  ASSERT_TRUE(graph != nullptr);
  uint32_t rank_size = 4;
  std::vector<int64_t> send_rank_ids = {3, 1, 2};
  std::vector<int64_t> recv_rank_ids = {};
  auto alltoall = CreateAllToAllvNode(graph, CreateInputs(graph, {{1}, {1}, {1}}, {type_, type_, type_}), send_rank_ids,
                                      recv_rank_ids);
  ASSERT_TRUE(alltoall != nullptr);
  ASSERT_NO_THROW(SetOutputs(alltoall, {}, {}));
  AllToAllvCalcParam calc(alltoall, rank_size);
  ASSERT_NO_THROW(calc.CalcOpParam());
  EXPECT_EQ(calc.GetSendCounts(), std::vector<int64_t>({0, 1, 1, 1}));
  EXPECT_EQ(calc.GetSendDispls(), std::vector<int64_t>({0, 128, 256, 0}));
  EXPECT_EQ(calc.GetRecvCounts(), std::vector<int64_t>({0, 0, 0, 0}));
  EXPECT_EQ(calc.GetRecvDispls(), std::vector<int64_t>({0, 0, 0, 0}));
}

/// Feature: AllToAllvCalcParam
/// Description: on 2p, rank id over valid range
/// Expectation: throw exception
TEST_F(TestHcclAdapter, test_all_to_all_v_calc_param_2p_invalid_rank_id) {
  auto graph = std::make_shared<FuncGraph>();
  ASSERT_TRUE(graph != nullptr);
  uint32_t rank_size = 2;
  std::vector<int64_t> send_rank_ids = {};
  std::vector<int64_t> recv_rank_ids = {0, 2};
  auto alltoall = CreateAllToAllvNode(graph, CreateInputs(graph, {}, {}), send_rank_ids, recv_rank_ids);
  ASSERT_TRUE(alltoall != nullptr);
  ASSERT_NO_THROW(SetOutputs(alltoall, {{1}, {1}}, {type_, type_}));
  AllToAllvCalcParam calc(alltoall, rank_size);
  ASSERT_ANY_THROW(calc.CalcOpParam());
}

/// Feature: AllToAllvCalcParam
/// Description: on 2p, has 2 outputs but only 1 recv_rank_ids is set
/// Expectation: throw exception
TEST_F(TestHcclAdapter, test_all_to_all_v_calc_param_2p_invalid_rank_id_2) {
  auto graph = std::make_shared<FuncGraph>();
  ASSERT_TRUE(graph != nullptr);
  uint32_t rank_size = 2;
  std::vector<int64_t> send_rank_ids = {};
  std::vector<int64_t> recv_rank_ids = {0};
  auto alltoall = CreateAllToAllvNode(graph, CreateInputs(graph, {}, {}), send_rank_ids, recv_rank_ids);
  ASSERT_TRUE(alltoall != nullptr);
  ASSERT_NO_THROW(SetOutputs(alltoall, {{1}, {1}}, {type_, type_}));
  AllToAllvCalcParam calc(alltoall, rank_size);
  ASSERT_ANY_THROW(calc.CalcOpParam());
}

/// Feature: AllToAllvCalcParam
/// Description: on 2p, rank id over valid range
/// Expectation: throw exception
TEST_F(TestHcclAdapter, test_all_to_all_v_calc_param_2p_wrong_order_and_invalid_rank_id) {
  auto graph = std::make_shared<FuncGraph>();
  ASSERT_TRUE(graph != nullptr);
  uint32_t rank_size = 2;
  std::vector<int64_t> send_rank_ids = {};
  std::vector<int64_t> recv_rank_ids = {2, 0};
  auto alltoall = CreateAllToAllvNode(graph, CreateInputs(graph, {}, {}), send_rank_ids, recv_rank_ids);
  ASSERT_TRUE(alltoall != nullptr);
  ASSERT_NO_THROW(SetOutputs(alltoall, {{1}, {1}}, {type_, type_}));
  AllToAllvCalcParam calc(alltoall, rank_size);
  ASSERT_ANY_THROW(calc.CalcOpParam());
}
}  // namespace mindspore::hccl
