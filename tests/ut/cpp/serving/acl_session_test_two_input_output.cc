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

class AclSessionTwoInputTwoOutputTest : public AclSessionTest {
 public:
  AclSessionTwoInputTwoOutputTest() = default;
  void SetUp() override {
    AclSessionTest::SetUp();
    aclmdlDesc model_desc;
    model_desc.inputs.push_back(
      AclTensorDesc{.dims = {2, 24, 24, 3}, .data_type = ACL_FLOAT, .size = 2 * 24 * 24 * 3 * sizeof(float)});

    model_desc.inputs.push_back(
      AclTensorDesc{.dims = {2, 32}, .data_type = ACL_INT32, .size = 2 * 32 * sizeof(int32_t)});

    model_desc.outputs.push_back(
      AclTensorDesc{.dims = {2, 8, 8, 3}, .data_type = ACL_FLOAT, .size = 2 * 8 * 8 * 3 * sizeof(float)});

    model_desc.outputs.push_back(
      AclTensorDesc{.dims = {2, 1024}, .data_type = ACL_BOOL, .size = 2 * 1024 * sizeof(bool)});

    mock_model_desc_ = MockModelDesc(model_desc);
    g_acl_model_desc = &mock_model_desc_;
  }
  void CreateDefaultRequest(PredictRequest &request) {
    auto input0 = request.add_data();
    CreateTensor(*input0, {2, 24, 24, 3}, ::ms_serving::DataType::MS_FLOAT32);
    auto input1 = request.add_data();
    CreateTensor(*input1, {2, 32}, ::ms_serving::DataType::MS_INT32);
  }

  void CreateInvalidDataSizeRequest0(PredictRequest &request) {
    auto input0 = request.add_data();
    // data size invalid, not match model input required
    CreateTensor(*input0, {2, 24, 24, 2}, ::ms_serving::DataType::MS_FLOAT32);

    auto input1 = request.add_data();
    CreateTensor(*input1, {2, 32}, ::ms_serving::DataType::MS_INT32);
  }

  void CreateInvalidDataSizeRequest1(PredictRequest &request) {
    auto input0 = request.add_data();
    CreateTensor(*input0, {2, 24, 24, 3}, ::ms_serving::DataType::MS_FLOAT32);
    auto input1 = request.add_data();
    // data size invalid, not match model input required
    CreateTensor(*input1, {2, 16}, ::ms_serving::DataType::MS_INT32);
  }

  void CreateInvalidDataSizeRequestOneInput0(PredictRequest &request) {
    // only has one input for input0
    auto input0 = request.add_data();
    CreateTensor(*input0, {2, 24, 24, 3}, ::ms_serving::DataType::MS_FLOAT32);
  }

  void CreateInvalidDataSizeRequestOneInput1(PredictRequest &request) {
    // only has one input for input1
    auto input0 = request.add_data();
    CreateTensor(*input0, {2, 32}, ::ms_serving::DataType::MS_INT32);
  }

  void CheckDefaultReply(const PredictReply &reply) {
    EXPECT_TRUE(reply.result().size() == 2);
    if (reply.result().size() == 2) {
      CheckTensorItem(reply.result(0), {2, 8, 8, 3}, ::ms_serving::DataType::MS_FLOAT32);
      CheckTensorItem(reply.result(1), {2, 1024}, ::ms_serving::DataType::MS_BOOL);
    }
  }

  MockModelDesc mock_model_desc_;
};

TEST_F(AclSessionTwoInputTwoOutputTest, TestAclSession_OneTime_Success) {
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

TEST_F(AclSessionTwoInputTwoOutputTest, TestAclSession_MutilTimes_Success) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile("fake_model_path", model_id) == SUCCESS);
  for (int i = 0; i < 10; i++) {
    // create inputs
    PredictRequest request;
    CreateDefaultRequest(request);

    PredictReply reply;
    ServingRequest serving_request(request);
    ServingReply serving_reply(reply);
    EXPECT_TRUE(acl_session.ExecuteModel(model_id, serving_request, serving_reply) == SUCCESS);
    CheckDefaultReply(reply);
  }
  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionTwoInputTwoOutputTest, TestAclSession_Input0_InvalidDataSize_Fail) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile("fake_model_path", model_id) == SUCCESS);
  // create inputs
  PredictRequest request;
  CreateInvalidDataSizeRequest0(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingReply serving_reply(reply);
  EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_request, serving_reply) == SUCCESS);
  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionTwoInputTwoOutputTest, TestAclSession_Input1_InvalidDataSize_Fail) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile("fake_model_path", model_id) == SUCCESS);
  // create inputs
  PredictRequest request;
  CreateInvalidDataSizeRequest1(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingReply serving_reply(reply);
  EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_request, serving_reply) == SUCCESS);
  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionTwoInputTwoOutputTest, TestAclSession_OnlyInput0_Fail) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile("fake_model_path", model_id) == SUCCESS);
  // create inputs
  PredictRequest request;
  CreateInvalidDataSizeRequestOneInput0(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingReply serving_reply(reply);
  EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_request, serving_reply) == SUCCESS);
  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionTwoInputTwoOutputTest, TestAclSession_OnlyInput1_Fail) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile("fake_model_path", model_id) == SUCCESS);
  // create inputs
  PredictRequest request;
  CreateInvalidDataSizeRequestOneInput1(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingReply serving_reply(reply);
  EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_request, serving_reply) == SUCCESS);
  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionTwoInputTwoOutputTest, TestAclSession_InvalidDataSize_MultiTimes_Fail) {
  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile("fake_model_path", model_id) == SUCCESS);
  for (int i = 0; i < 10; i++) {
    // create inputs
    PredictRequest request;
    CreateInvalidDataSizeRequest0(request);

    PredictReply reply;
    ServingRequest serving_request(request);
    ServingReply serving_reply(reply);
    EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_request, serving_reply) == SUCCESS);
  }
  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

}  // namespace serving
}  // namespace mindspore