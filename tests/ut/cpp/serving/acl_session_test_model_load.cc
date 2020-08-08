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

#include "acl_session_test_common.h"

using namespace std;

namespace mindspore {
namespace serving {

class MockFailAclDeviceContextStream : public AclDeviceContextStream {
 public:
  aclError aclrtSetDevice(int32_t deviceId) override {
    if (set_device_fail_list_.empty()) {
      return AclDeviceContextStream::aclrtSetDevice(deviceId);
    }
    auto val = set_device_fail_list_.front();
    set_device_fail_list_.erase(set_device_fail_list_.begin());
    if (val) {
      return AclDeviceContextStream::aclrtSetDevice(deviceId);
    }
    return 1;
  }

  aclError aclrtResetDevice(int32_t deviceId) override {
    auto ret = AclDeviceContextStream::aclrtResetDevice(deviceId);
    if (ret != ACL_ERROR_NONE) {
      return ret;
    }
    if (reset_device_fail_list_.empty()) {
      return ret;
    }
    auto val = reset_device_fail_list_.front();
    reset_device_fail_list_.erase(reset_device_fail_list_.begin());
    return val ? ACL_ERROR_NONE : 1;
  }

  aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId) override {
    if (create_context_fail_list_.empty()) {
      return AclDeviceContextStream::aclrtCreateContext(context, deviceId);
    }
    auto val = create_context_fail_list_.front();
    create_context_fail_list_.erase(create_context_fail_list_.begin());
    if (val) {
      return AclDeviceContextStream::aclrtCreateContext(context, deviceId);
    }
    return 1;
  }

  aclError aclrtDestroyContext(aclrtContext context) override {
    auto ret = AclDeviceContextStream::aclrtDestroyContext(context);
    if (ret != ACL_ERROR_NONE) {
      return ret;
    }
    if (destroy_context_fail_list_.empty()) {
      return ret;
    }
    auto val = destroy_context_fail_list_.front();
    destroy_context_fail_list_.erase(destroy_context_fail_list_.begin());
    return val ? ACL_ERROR_NONE : 1;
  }

  aclError aclrtCreateStream(aclrtStream *stream) override {
    if (create_stream_fail_list_.empty()) {
      return AclDeviceContextStream::aclrtCreateStream(stream);
    }
    auto val = create_stream_fail_list_.front();
    create_stream_fail_list_.erase(create_stream_fail_list_.begin());
    if (val) {
      return AclDeviceContextStream::aclrtCreateStream(stream);
    }
    return 1;
  }

  aclError aclrtDestroyStream(aclrtStream stream) override {
    auto ret = AclDeviceContextStream::aclrtDestroyStream(stream);
    if (ret != ACL_ERROR_NONE) {
      return ret;
    }
    if (destroy_stream_fail_list_.empty()) {
      return ret;
    }
    auto val = destroy_stream_fail_list_.front();
    destroy_stream_fail_list_.erase(destroy_stream_fail_list_.begin());
    return val ? ACL_ERROR_NONE : 1;
  }
  std::vector<bool> set_device_fail_list_;
  std::vector<bool> reset_device_fail_list_;
  std::vector<bool> create_context_fail_list_;
  std::vector<bool> destroy_context_fail_list_;
  std::vector<bool> create_stream_fail_list_;
  std::vector<bool> destroy_stream_fail_list_;
};

class MockFailAclMemory : public AclMemory {
 public:
  aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) override {
    if (device_mem_fail_list_.empty()) {
      return AclMemory::aclrtMalloc(devPtr, size, policy);
    }
    auto val = device_mem_fail_list_.front();
    device_mem_fail_list_.erase(device_mem_fail_list_.begin());
    if (val) {
      return AclMemory::aclrtMalloc(devPtr, size, policy);
    }
    return 1;
  }
  aclError aclrtMallocHost(void **hostPtr, size_t size) override {
    if (host_mem_fail_list_.empty()) {
      return AclMemory::aclrtMallocHost(hostPtr, size);
    }
    auto val = host_mem_fail_list_.front();
    host_mem_fail_list_.erase(host_mem_fail_list_.begin());
    if (val) {
      return AclMemory::aclrtMallocHost(hostPtr, size);
    }
    return 1;
  }
  aclError acldvppMalloc(void **devPtr, size_t size) override {
    if (dvpp_mem_fail_list_.empty()) {
      return AclMemory::acldvppMalloc(devPtr, size);
    }
    auto val = dvpp_mem_fail_list_.front();
    dvpp_mem_fail_list_.erase(dvpp_mem_fail_list_.begin());
    if (val) {
      return AclMemory::acldvppMalloc(devPtr, size);
    }
    return 1;
  }

  std::vector<bool> device_mem_fail_list_;
  std::vector<bool> host_mem_fail_list_;
  std::vector<bool> dvpp_mem_fail_list_;
};

class AclSessionModelLoadTest : public AclSessionTest {
 public:
  AclSessionModelLoadTest() = default;
  void SetUp() override {
    AclSessionTest::SetUp();
    aclmdlDesc model_desc;
    model_desc.inputs.push_back(
      AclTensorDesc{.dims = {2, 24, 24, 3}, .data_type = ACL_FLOAT, .size = 2 * 24 * 24 * 3 * sizeof(float)});

    model_desc.inputs.push_back(
      AclTensorDesc{.dims = {2, 24, 24, 3}, .data_type = ACL_FLOAT, .size = 2 * 24 * 24 * 3 * sizeof(float)});

    model_desc.outputs.push_back(
      AclTensorDesc{.dims = {2, 24, 24, 3}, .data_type = ACL_FLOAT, .size = 2 * 24 * 24 * 3 * sizeof(float)});

    model_desc.outputs.push_back(
      AclTensorDesc{.dims = {2, 24, 24, 3}, .data_type = ACL_FLOAT, .size = 2 * 24 * 24 * 3 * sizeof(float)});

    mock_model_desc_ = MockModelDesc(model_desc);
    g_acl_model_desc = &mock_model_desc_;
    g_acl_device_context_stream = &fail_acl_device_context_stream_;
    g_acl_memory = &fail_acl_memory_;
  }
  void CreateDefaultRequest(PredictRequest &request) {
    auto input0 = request.add_data();
    CreateTensor(*input0, {2, 24, 24, 3}, ::ms_serving::DataType::MS_FLOAT32);
    auto input1 = request.add_data();
    CreateTensor(*input1, {2, 24, 24, 3}, ::ms_serving::DataType::MS_FLOAT32);
  }

  void CheckDefaultReply(const PredictReply &reply) {
    EXPECT_TRUE(reply.result().size() == 2);
    if (reply.result().size() == 2) {
      CheckTensorItem(reply.result(0), {2, 24, 24, 3}, ::ms_serving::DataType::MS_FLOAT32);
      CheckTensorItem(reply.result(1), {2, 24, 24, 3}, ::ms_serving::DataType::MS_FLOAT32);
    }
  }
  MockModelDesc mock_model_desc_;
  /* Test Resource will be release on something wrong happens*/
  MockFailAclDeviceContextStream fail_acl_device_context_stream_;
  MockFailAclMemory fail_acl_memory_;
};

TEST_F(AclSessionModelLoadTest, TestAclSession_OneTime_Success) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile("fake_model_path", model_id) == SUCCESS);
  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingReply serving_reply(reply);
  EXPECT_TRUE(acl_session.ExecuteModel(model_id, serving_request, serving_reply) == SUCCESS);
  CheckDefaultReply(reply);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionModelLoadTest, TestAclSession_SetDeviceFail) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  fail_acl_device_context_stream_.set_device_fail_list_.push_back(false);
  EXPECT_FALSE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionModelLoadTest, TestAclSession_CreateContextFail) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  fail_acl_device_context_stream_.create_context_fail_list_.push_back(false);
  EXPECT_FALSE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionModelLoadTest, TestAclSession_CreateStreamFail) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  fail_acl_device_context_stream_.create_stream_fail_list_.push_back(false);
  EXPECT_FALSE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionModelLoadTest, TestAclSession_ResetDeviceFail) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  fail_acl_device_context_stream_.reset_device_fail_list_.push_back(false);
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  acl_session.FinalizeEnv();
};

TEST_F(AclSessionModelLoadTest, TestAclSession_DestroyContextFail) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  fail_acl_device_context_stream_.destroy_context_fail_list_.push_back(false);
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  acl_session.FinalizeEnv();
};

TEST_F(AclSessionModelLoadTest, TestAclSession_DestroyStreamFail) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  fail_acl_device_context_stream_.destroy_stream_fail_list_.push_back(false);
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  acl_session.FinalizeEnv();
};

TEST_F(AclSessionModelLoadTest, TestAclSession_MallocFail0_Success) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  fail_acl_memory_.device_mem_fail_list_.push_back(false);  // input0 buffer
  EXPECT_FALSE(acl_session.LoadModelFromFile("fake_model_path", model_id) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionModelLoadTest, TestAclSession_MallocFail1_Success) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  fail_acl_memory_.device_mem_fail_list_.push_back(true);   // input0 buffer
  fail_acl_memory_.device_mem_fail_list_.push_back(false);  // input1 buffer
  EXPECT_FALSE(acl_session.LoadModelFromFile("fake_model_path", model_id) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionModelLoadTest, TestAclSession_MallocFail2_Success) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  fail_acl_memory_.device_mem_fail_list_.push_back(true);   // input0 buffer
  fail_acl_memory_.device_mem_fail_list_.push_back(true);   // input1 buffer
  fail_acl_memory_.device_mem_fail_list_.push_back(false);  // output0 buffer
  EXPECT_FALSE(acl_session.LoadModelFromFile("fake_model_path", model_id) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionModelLoadTest, TestAclSession_MallocFail3_Success) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  fail_acl_memory_.device_mem_fail_list_.push_back(true);   // input0 buffer
  fail_acl_memory_.device_mem_fail_list_.push_back(true);   // input1 buffer
  fail_acl_memory_.device_mem_fail_list_.push_back(true);   // output0 buffer
  fail_acl_memory_.device_mem_fail_list_.push_back(false);  // output1 buffer
  EXPECT_FALSE(acl_session.LoadModelFromFile("fake_model_path", model_id) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionModelLoadTest, TestAclSession_RunOnDevice_MallocFail0_Success) {
  SetDeviceRunMode();
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  fail_acl_memory_.host_mem_fail_list_.push_back(false);  // output0 buffer
  EXPECT_FALSE(acl_session.LoadModelFromFile("fake_model_path", model_id) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionModelLoadTest, TestAclSession_RunOnDevice_MallocFail1_Success) {
  SetDeviceRunMode();
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  fail_acl_memory_.host_mem_fail_list_.push_back(true);   // output0 buffer
  fail_acl_memory_.host_mem_fail_list_.push_back(false);  // output1 buffer
  EXPECT_FALSE(acl_session.LoadModelFromFile("fake_model_path", model_id) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

}  // namespace serving
}  // namespace mindspore