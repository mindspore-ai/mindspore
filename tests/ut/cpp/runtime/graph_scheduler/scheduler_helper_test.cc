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

#include "common/common_test.h"
#include "abstract/abstract_function.h"
#include "runtime/graph_scheduler/scheduler_helper.h"

namespace mindspore {
namespace runtime {
class SchedulerHelperTest : public UT::Common {
 public:
  SchedulerHelperTest() {}
};

/// Feature: Add fusion actor.
/// Description: Test the common interface.
/// Expectation: As expected.
TEST_F(SchedulerHelperTest, AddDependency) {
  auto memory_manager_actor = std::make_shared<MemoryManagerActor>();
  MS_EXCEPTION_IF_NULL(memory_manager_actor);
  auto kernel_graph = std::make_shared<KernelGraph>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimLess)};
  auto backend_node1 = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(backend_node1);
  auto backend_node2 = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(backend_node2);
  std::set<size_t> ref_input_indexes;
  std::set<size_t> ref_output_indexes;

  auto from_actor =
    std::make_shared<KernelActor>("from_actor", backend_node1, nullptr, memory_manager_actor->GetAID(), nullptr,
                                  nullptr, GraphExecutionStrategy::kPipeline, ref_input_indexes, ref_output_indexes);
  auto to_actor =
    std::make_shared<KernelActor>("to_actor", backend_node2, nullptr, memory_manager_actor->GetAID(), nullptr, nullptr,
                                  GraphExecutionStrategy::kPipeline, ref_input_indexes, ref_output_indexes);
  SchedulerHelper::AddDependency(from_actor.get(), to_actor.get());
  ASSERT_EQ(1, from_actor->dependent_actors().size());

  auto fusion_actor = SchedulerHelper::BuildFusionActor({from_actor, to_actor});
  ASSERT_EQ(2, fusion_actor->sub_actors().size());

  SchedulerHelper::AddArrowForFusionActor(fusion_actor.get());
  ASSERT_EQ(0, fusion_actor->input_data_arrow_aids().size());

  SchedulerHelper::FuseDataArrowsToBatchDataArrow(fusion_actor.get());
  ASSERT_EQ(0, fusion_actor->batch_output_data_arrows().size());
}

/// Feature: Integration of dynamic and static memory.
/// Description: Test the common interface of AddSomasInfo.
/// Expectation: As expected.
TEST_F(SchedulerHelperTest, AddSomasInfo) {
  auto memory_manager_actor = std::make_shared<MemoryManagerActor>();
  MS_EXCEPTION_IF_NULL(memory_manager_actor);
  auto kernel_graph = std::make_shared<KernelGraph>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimLess)};
  auto backend_node = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(backend_node);
  std::set<size_t> ref_input_indexes;
  std::set<size_t> ref_output_indexes;
  auto kernel_actor =
    std::make_shared<KernelActor>("kernel_actor", backend_node, nullptr, memory_manager_actor->GetAID(), nullptr,
                                  nullptr, GraphExecutionStrategy::kPipeline, ref_input_indexes, ref_output_indexes);

  SchedulerHelper::AddSomasInfo(kernel_actor.get());
  ASSERT_EQ(kernel_actor->somas_info(), nullptr);

  // Enable somas.
  auto somas_info = kernel_graph->MutableSomasInfo();
  ASSERT_NE(somas_info, nullptr);
  somas_info->whole_block_size_ = 1;
  SchedulerHelper::AddSomasInfo(kernel_actor.get());
  ASSERT_NE(kernel_actor->somas_info(), nullptr);
  ASSERT_EQ(kernel_actor->somas_info(), somas_info);
}

/// Feature: Integration of dynamic and static memory.
/// Description: Test the common interface of AddMemoryAllocSign.
/// Expectation: As expected.
TEST_F(SchedulerHelperTest, AddMemoryAllocSign) {
  auto memory_manager_actor = std::make_shared<MemoryManagerActor>();
  MS_EXCEPTION_IF_NULL(memory_manager_actor);
  auto to_graph = std::make_shared<KernelGraph>();
  MS_EXCEPTION_IF_NULL(to_graph);
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimLess)};
  auto backend_node1 = to_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(backend_node1);
  auto backend_node2 = to_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(backend_node2);
  std::set<size_t> ref_input_indexes;
  std::set<size_t> ref_output_indexes;

  auto from_actor =
    std::make_shared<KernelActor>("from_actor", backend_node1, nullptr, memory_manager_actor->GetAID(), nullptr,
                                  nullptr, GraphExecutionStrategy::kPipeline, ref_input_indexes, ref_output_indexes);
  auto to_actor =
    std::make_shared<KernelActor>("to_actor", backend_node2, nullptr, memory_manager_actor->GetAID(), nullptr, nullptr,
                                  GraphExecutionStrategy::kPipeline, ref_input_indexes, ref_output_indexes);

  SchedulerHelper::AddMemoryAllocSign(from_actor.get(), to_actor.get(), to_graph);
  ASSERT_EQ(to_actor->memory_alloc_insert_position(), nullptr);

  // Enable somas.
  auto somas_info = to_graph->MutableSomasInfo();
  ASSERT_NE(somas_info, nullptr);
  somas_info->whole_block_size_ = 1;
  SchedulerHelper::AddMemoryAllocSign(from_actor.get(), to_actor.get(), to_graph);
  ASSERT_NE(to_actor->memory_alloc_insert_position(), nullptr);
  ASSERT_EQ(to_actor->memory_alloc_insert_position(), from_actor.get());
}

/// Feature: Integration of dynamic and static memory.
/// Description: Test the common interface of AddMemoryFreeSign.
/// Expectation: As expected.
TEST_F(SchedulerHelperTest, AddMemoryFreeSign) {
  auto memory_manager_actor = std::make_shared<MemoryManagerActor>();
  MS_EXCEPTION_IF_NULL(memory_manager_actor);
  auto from_graph = std::make_shared<KernelGraph>();
  MS_EXCEPTION_IF_NULL(from_graph);
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimLess)};
  auto backend_node1 = from_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(backend_node1);
  auto backend_node2 = from_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(backend_node2);
  std::set<size_t> ref_input_indexes;
  std::set<size_t> ref_output_indexes;

  auto from_actor =
    std::make_shared<KernelActor>("from_actor", backend_node1, nullptr, memory_manager_actor->GetAID(), nullptr,
                                  nullptr, GraphExecutionStrategy::kPipeline, ref_input_indexes, ref_output_indexes);
  auto to_actor =
    std::make_shared<KernelActor>("to_actor", backend_node2, nullptr, memory_manager_actor->GetAID(), nullptr, nullptr,
                                  GraphExecutionStrategy::kPipeline, ref_input_indexes, ref_output_indexes);

  SchedulerHelper::AddMemoryFreeSign(from_actor.get(), to_actor.get(), from_graph);
  ASSERT_EQ(from_actor->memory_free_insert_position(), nullptr);

  // Enable somas.
  auto somas_info = from_graph->MutableSomasInfo();
  ASSERT_NE(somas_info, nullptr);
  somas_info->whole_block_size_ = 1;
  SchedulerHelper::AddMemoryFreeSign(from_actor.get(), to_actor.get(), from_graph);
  ASSERT_NE(from_actor->memory_free_insert_position(), nullptr);
  ASSERT_EQ(from_actor->memory_free_insert_position(), to_actor.get());
}
}  // namespace runtime
}  // namespace mindspore
