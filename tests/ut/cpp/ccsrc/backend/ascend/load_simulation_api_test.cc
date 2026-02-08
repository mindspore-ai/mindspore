/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "utils/ms_context.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_base_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_tdt_symbol.h"
#define private public
#define protected public
#include "plugin/ascend/res_manager/mem_manager/ascend_memory_adapter.h"
#undef private
#undef protected
using namespace mindspore::device::ascend;
namespace mindspore {
namespace device {
namespace ascend {
class TestLoadSimulationAPI : public UT::Common {
 public:
  TestLoadSimulationAPI() = default;
  virtual ~TestLoadSimulationAPI() = default;

  void SetUp() override {
    common::SetEnv("MS_SIMULATION_LEVEL", "1");
    MsContext::GetInstance()->set_param<std::string>(MS_CTX_DEVICE_TARGET, "Ascend");
    MsContext::GetInstance()->set_param<int>(MS_CTX_EXECUTION_MODE, 1);
  }
  void TearDown() override {}
};

/// Feature: test load simulation api.
/// Description: load simulation api.
/// Expectation: load simulation api and can not throw exception.
TEST_F(TestLoadSimulationAPI, test_load_simulation_api) {
  LoadSimulationApiSymbols();
  // simulation acl base api
  EXPECT_NE(aclCreateDataBuffer_, nullptr);
  EXPECT_NE(aclCreateTensorDesc_, nullptr);
  EXPECT_NE(aclDataTypeSize_, nullptr);
  EXPECT_NE(aclDestroyDataBuffer_, nullptr);
  EXPECT_NE(aclDestroyTensorDesc_, nullptr);
  EXPECT_NE(aclGetTensorDescDimV2_, nullptr);
  EXPECT_NE(aclGetTensorDescNumDims_, nullptr);
  EXPECT_NE(aclSetTensorConst_, nullptr);
  EXPECT_NE(aclSetTensorDescName_, nullptr);
  EXPECT_NE(aclSetTensorFormat_, nullptr);
  EXPECT_NE(aclSetTensorPlaceMent_, nullptr);
  EXPECT_NE(aclSetTensorShape_, nullptr);

  // simulation rt api
  EXPECT_NE(aclrtCreateContext_, nullptr);
  EXPECT_NE(aclrtCreateEventWithFlag_, nullptr);
  EXPECT_NE(aclrtCreateStreamWithConfig_, nullptr);
  EXPECT_NE(aclrtDestroyContext_, nullptr);
  EXPECT_NE(aclrtDestroyEvent_, nullptr);
  EXPECT_NE(aclrtDestroyStream_, nullptr);
  EXPECT_NE(aclrtDestroyStreamForce_, nullptr);
  EXPECT_NE(aclrtEventElapsedTime_, nullptr);
  EXPECT_NE(aclrtFree_, nullptr);
  EXPECT_NE(aclrtFreeHost_, nullptr);
  EXPECT_NE(aclrtGetCurrentContext_, nullptr);
  EXPECT_NE(aclrtGetDevice_, nullptr);
  EXPECT_NE(aclrtGetDeviceCount_, nullptr);
  EXPECT_NE(aclrtGetDeviceIdFromExceptionInfo_, nullptr);
  EXPECT_NE(aclrtGetErrorCodeFromExceptionInfo_, nullptr);
  EXPECT_NE(aclrtGetMemInfo_, nullptr);
  EXPECT_NE(aclrtGetRunMode_, nullptr);
  EXPECT_NE(aclrtGetStreamIdFromExceptionInfo_, nullptr);
  EXPECT_NE(aclrtGetTaskIdFromExceptionInfo_, nullptr);
  EXPECT_NE(aclrtGetThreadIdFromExceptionInfo_, nullptr);
  EXPECT_NE(aclrtLaunchCallback_, nullptr);
  EXPECT_NE(aclrtMalloc_, nullptr);
  EXPECT_NE(aclrtMallocHost_, nullptr);
  EXPECT_NE(aclrtMemcpy_, nullptr);
  EXPECT_NE(aclrtMemcpyAsync_, nullptr);
  EXPECT_NE(aclrtMemset_, nullptr);
  EXPECT_NE(aclrtProcessReport_, nullptr);
  EXPECT_NE(aclrtQueryEventStatus_, nullptr);
  EXPECT_NE(aclrtRecordEvent_, nullptr);
  EXPECT_NE(aclrtResetDevice_, nullptr);
  EXPECT_NE(aclrtResetEvent_, nullptr);
  EXPECT_NE(aclrtSetCurrentContext_, nullptr);
  EXPECT_NE(aclrtSetDevice_, nullptr);
  EXPECT_NE(aclrtSetDeviceSatMode_, nullptr);
  EXPECT_NE(aclrtSetExceptionInfoCallback_, nullptr);
  EXPECT_NE(aclrtSetOpExecuteTimeOut_, nullptr);
  EXPECT_NE(aclrtSetOpWaitTimeout_, nullptr);
  EXPECT_NE(aclrtSetStreamFailureMode_, nullptr);
  EXPECT_NE(aclrtStreamQuery_, nullptr);
  EXPECT_NE(aclrtStreamWaitEvent_, nullptr);
  EXPECT_NE(aclrtSubscribeReport_, nullptr);
  EXPECT_NE(aclrtSynchronizeEvent_, nullptr);
  EXPECT_NE(aclrtSynchronizeStream_, nullptr);
  EXPECT_NE(aclrtSynchronizeStreamWithTimeout_, nullptr);
  EXPECT_NE(aclrtUnmapMem_, nullptr);
  EXPECT_NE(aclrtReserveMemAddress_, nullptr);
  EXPECT_NE(aclrtMallocPhysical_, nullptr);
  EXPECT_NE(aclrtMapMem_, nullptr);
  EXPECT_NE(aclrtFreePhysical_, nullptr);
  EXPECT_NE(aclrtReleaseMemAddress_, nullptr);
  EXPECT_NE(aclrtCtxSetSysParamOpt_, nullptr);

  // simulation tdt api
  EXPECT_NE(acltdtAddDataItem_, nullptr);
  EXPECT_NE(acltdtCreateChannel_, nullptr);
  EXPECT_NE(acltdtCreateChannelWithCapacity_, nullptr);
  EXPECT_NE(acltdtCreateDataItem_, nullptr);
  EXPECT_NE(acltdtCreateDataset_, nullptr);
  EXPECT_NE(acltdtDestroyChannel_, nullptr);
  EXPECT_NE(acltdtDestroyDataItem_, nullptr);
  EXPECT_NE(acltdtDestroyDataset_, nullptr);
  EXPECT_NE(acltdtGetDataAddrFromItem_, nullptr);
  EXPECT_NE(acltdtGetDataItem_, nullptr);
  EXPECT_NE(acltdtGetDatasetName_, nullptr);
  EXPECT_NE(acltdtGetDatasetSize_, nullptr);
  EXPECT_NE(acltdtGetDataSizeFromItem_, nullptr);
  EXPECT_NE(acltdtGetDataTypeFromItem_, nullptr);
  EXPECT_NE(acltdtGetDimNumFromItem_, nullptr);
  EXPECT_NE(acltdtGetDimsFromItem_, nullptr);
  EXPECT_NE(acltdtGetTensorTypeFromItem_, nullptr);
  EXPECT_NE(acltdtGetSliceInfoFromItem_, nullptr);
  EXPECT_NE(acltdtQueryChannelSize_, nullptr);
  EXPECT_NE(acltdtReceiveTensor_, nullptr);
  EXPECT_NE(acltdtSendTensor_, nullptr);
  EXPECT_NE(acltdtStopChannel_, nullptr);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
