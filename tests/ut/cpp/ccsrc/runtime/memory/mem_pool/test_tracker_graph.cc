/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "runtime/memory/mem_pool/tracker_graph.h"
#include <gtest/gtest.h>

namespace mindspore {
namespace device {
namespace tracker {
namespace graph {

class TestTrackerGraph : public UT::Common {};

/// Feature: tracker graph unit tests.
/// Description: test need skip.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestTrackerGraph, test_need_skip) {
  // Positive cases.
  EXPECT_TRUE(NeedSkipRaceCheck(nullptr));

  TaskInfoPtr task_info = std::make_shared<TaskInfo>();
  task_info->node_name = "Reshape";
  EXPECT_TRUE(NeedSkipRaceCheck(task_info));

  task_info->node_name = "Default/Reshape-op0";
  EXPECT_TRUE(NeedSkipRaceCheck(task_info));

  task_info->node_name = "Default/ReshapeCell/Reshape-op0";
  EXPECT_TRUE(NeedSkipRaceCheck(task_info));

  task_info->node_name =
    "Default/network-MFPipelineWithLossScaleCell/newwork-_VirtualDatasetCell/_backbone-PipelineCell/"
    "network-DataOrderWrapperCell/network-LlamaForCausalLM/model-LlamaModel/casual_mask-LowerTriangularMaskWithDynamic/"
    "Reshape-op0";
  EXPECT_TRUE(NeedSkipRaceCheck(task_info));

  // Negative cases.
  task_info->node_name = "";
  EXPECT_FALSE(NeedSkipRaceCheck(task_info));

  task_info->node_name = "Default/Reshape-op";
  EXPECT_FALSE(NeedSkipRaceCheck(task_info));

  task_info->node_name = "Default/Reshape-opa";
  EXPECT_FALSE(NeedSkipRaceCheck(task_info));

  task_info->node_name = "MatMul";
  EXPECT_FALSE(NeedSkipRaceCheck(task_info));

  task_info->node_name =
    "Reshape/Default/network-MFPipelineWithLossScaleCell/newwork-_VirtualDatasetCell/_backbone-PipelineCell/"
    "network-DataOrderWrapperCell/network-LlamaForCausalLM/model-LlamaModel/casual_mask-LowerTriangularMaskWithDynamic/"
    "Add-op0";
  EXPECT_FALSE(NeedSkipRaceCheck(task_info));

  task_info->node_name =
    "Default/network-MFPipelineWithLossScaleCell/newwork-_VirtualDatasetCell/_backbone-PipelineCell/"
    "network-DataOrderWrapperCell/network-LlamaForCausalLM/model-LlamaModel/layers-CellList/0-LlamaDecodeLayer/"
    "attention-LLamaAttention/wv-Liner/Add-Reshapeop0";
  EXPECT_FALSE(NeedSkipRaceCheck(task_info));

  task_info->node_name =
    "Default/network-MFPipelineWithLossScaleCell/newwork-_VirtualDatasetCell/_backbone-PipelineCell/"
    "network-DataOrderWrapperCell/network-LlamaForCausalLM/model-LlamaModel/layers-CellList/0-LlamaDecodeLayer/"
    "attention-LLamaAttention/wv-Liner/Add-Reshape-op0";
  EXPECT_FALSE(NeedSkipRaceCheck(task_info));

  task_info->node_name =
    "Default/Reshape-op0/network-MFPipelineWithLossScaleCell/newwork-_VirtualDatasetCell/_backbone-PipelineCell/"
    "network-DataOrderWrapperCell/network-LlamaForCausalLM/model-LlamaModel/layers-CellList/0-LlamaDecodeLayer/"
    "attention-LLamaAttention/Add-op0";
  EXPECT_FALSE(NeedSkipRaceCheck(task_info));
}

/// Feature: tracker graph unit tests.
/// Description: test is event.
/// Expectation: all interface work normally and can not throw exception.
TEST_F(TestTrackerGraph, test_is_event) {
  // Positive cases.
  TaskInfoPtr task_info = std::make_shared<TaskInfo>();
  task_info->node_name = "RecordEvent";
  EXPECT_TRUE(IsEvent(task_info, "RecordEvent"));
  task_info->node_name = "WaitEvent";
  EXPECT_TRUE(IsEvent(task_info, "WaitEvent"));

  task_info->node_name = "Default/StreamSend-op0";
  EXPECT_TRUE(IsEvent(task_info, "RecordEvent"));
  task_info->node_name = "Default/StreamRecv-op0";
  EXPECT_TRUE(IsEvent(task_info, "WaitEvent"));

  task_info->node_name = "Default/StreamSendCell/StreamSend-op0";
  EXPECT_TRUE(IsEvent(task_info, "RecordEvent"));
  task_info->node_name = "Default/StreamRecvCell/StreamRecv-op0";
  EXPECT_TRUE(IsEvent(task_info, "WaitEvent"));

  task_info->node_name =
    "Default/network-MFPipelineWithLossScaleCell/newwork-_VirtualDatasetCell/_backbone-PipelineCell/"
    "network-DataOrderWrapperCell/network-LlamaForCausalLM/model-LlamaModel/casual_mask-LowerTriangularMaskWithDynamic/"
    "StreamSend-op0";
  EXPECT_TRUE(IsEvent(task_info, "RecordEvent"));
  task_info->node_name =
    "Default/network-MFPipelineWithLossScaleCell/newwork-_VirtualDatasetCell/_backbone-PipelineCell/"
    "network-DataOrderWrapperCell/network-LlamaForCausalLM/model-LlamaModel/casual_mask-LowerTriangularMaskWithDynamic/"
    "StreamRecv-op0";
  EXPECT_TRUE(IsEvent(task_info, "WaitEvent"));

  // Negative cases.
  task_info->node_name = "";
  EXPECT_FALSE(IsEvent(task_info, "RecordEvent"));
  EXPECT_FALSE(IsEvent(task_info, "WaitEvent"));

  task_info->node_name = "RecordEvent";
  EXPECT_FALSE(IsEvent(task_info, "WaitEvent"));
  task_info->node_name = "WaitEvent";
  EXPECT_FALSE(IsEvent(task_info, "RecordEvent"));

  task_info->node_name = "Default/StreamSend-op0";
  EXPECT_FALSE(IsEvent(task_info, "WaitEvent"));
  task_info->node_name = "Default/StreamRecv-op0";
  EXPECT_FALSE(IsEvent(task_info, "RecordEvent"));

  task_info->node_name = "Default/StreamSend-op";
  EXPECT_FALSE(IsEvent(task_info, "RecordEvent"));
  task_info->node_name = "Default/StreamRecv-op";
  EXPECT_FALSE(IsEvent(task_info, "WaitEvent"));

  task_info->node_name = "Default/StreamSend-opa";
  EXPECT_FALSE(IsEvent(task_info, "RecordEvent"));
  task_info->node_name = "Default/StreamRecv-opa";
  EXPECT_FALSE(IsEvent(task_info, "WaitEvent"));

  task_info->node_name = "MatMul";
  EXPECT_FALSE(IsEvent(task_info, "RecordEvent"));
  EXPECT_FALSE(IsEvent(task_info, "WaitEvent"));

  task_info->node_name =
    "StreamSend/Default/network-MFPipelineWithLossScaleCell/newwork-_VirtualDatasetCell/_backbone-PipelineCell/"
    "network-DataOrderWrapperCell/network-LlamaForCausalLM/model-LlamaModel/casual_mask-LowerTriangularMaskWithDynamic/"
    "Add-op0";
  EXPECT_FALSE(IsEvent(task_info, "RecordEvent"));
  
  task_info->node_name =
    "Default/network-MFPipelineWithLossScaleCell/newwork-_VirtualDatasetCell/_backbone-PipelineCell/"
    "network-DataOrderWrapperCell/network-LlamaForCausalLM/model-LlamaModel/layers-CellList/0-LlamaDecodeLayer/"
    "attention-LLamaAttention/wv-Liner/Add-StreamSendop0";
  EXPECT_FALSE(IsEvent(task_info, "RecordEvent"));

  task_info->node_name =
    "Default/network-MFPipelineWithLossScaleCell/newwork-_VirtualDatasetCell/_backbone-PipelineCell/"
    "network-DataOrderWrapperCell/network-LlamaForCausalLM/model-LlamaModel/layers-CellList/0-LlamaDecodeLayer/"
    "attention-LLamaAttention/wv-Liner/Add-StreamSend-op0";
  EXPECT_FALSE(IsEvent(task_info, "RecordEvent"));

  task_info->node_name =
    "Default/StreamSend-op0/network-MFPipelineWithLossScaleCell/newwork-_VirtualDatasetCell/_backbone-PipelineCell/"
    "network-DataOrderWrapperCell/network-LlamaForCausalLM/model-LlamaModel/layers-CellList/0-LlamaDecodeLayer/"
    "attention-LLamaAttention/Add-op0";
  EXPECT_FALSE(IsEvent(task_info, "RecordEvent"));
}
}  // namespace graph
}  // namespace tracker
}  // namespace device
}  // namespace mindspore
