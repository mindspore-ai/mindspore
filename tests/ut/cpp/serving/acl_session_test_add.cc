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

class AclSessionAddTest : public AclSessionTest {
 public:
  AclSessionAddTest() = default;
  void SetUp() override {
    AclSessionTest::SetUp();
    aclmdlDesc model_desc;
    model_desc.inputs.push_back(
      AclTensorDesc{.dims = {2, 24, 24, 3}, .data_type = ACL_FLOAT, .size = 2 * 24 * 24 * 3 * sizeof(float)});

    model_desc.inputs.push_back(
      AclTensorDesc{.dims = {2, 24, 24, 3}, .data_type = ACL_FLOAT, .size = 2 * 24 * 24 * 3 * sizeof(float)});

    model_desc.outputs.push_back(
      AclTensorDesc{.dims = {2, 24, 24, 3}, .data_type = ACL_FLOAT, .size = 2 * 24 * 24 * 3 * sizeof(float)});

    mock_model_desc_ = MockModelDesc(model_desc);
    g_acl_model_desc = &mock_model_desc_;
    g_acl_model = &add_mock_model_;
  }
  void CreateDefaultRequest(PredictRequest &request) {
    auto input0 = request.add_data();
    CreateTensor(*input0, {2, 24, 24, 3}, ::ms_serving::DataType::MS_FLOAT32);
    auto input1 = request.add_data();
    CreateTensor(*input1, {2, 24, 24, 3}, ::ms_serving::DataType::MS_FLOAT32);

    auto input0_data = reinterpret_cast<float *>(input0->mutable_data()->data());
    auto input1_data = reinterpret_cast<float *>(input1->mutable_data()->data());
    for (int i = 0; i < 2 * 24 * 24 * 3; i++) {
      input0_data[i] = i % 1024;
      input1_data[i] = i % 1024 + 1;
    }
  }

  void CheckDefaultReply(const PredictReply &reply) {
    EXPECT_TRUE(reply.result().size() == 1);
    if (reply.result().size() == 1) {
      CheckTensorItem(reply.result(0), {2, 24, 24, 3}, ::ms_serving::DataType::MS_FLOAT32);
      auto &output = reply.result(0).data();
      EXPECT_EQ(output.size(), 2 * 24 * 24 * 3 * sizeof(float));
      if (output.size() == 2 * 24 * 24 * 3 * sizeof(float)) {
        auto output_data = reinterpret_cast<const float *>(output.data());
        for (int i = 0; i < 2 * 24 * 24 * 3; i++) {
          EXPECT_EQ(output_data[i], (i % 1024) + (i % 1024 + 1));
          if (output_data[i] != (i % 1024) + (i % 1024 + 1)) {
            break;
          }
        }
      }
    }
  }
  MockModelDesc mock_model_desc_;
  AddMockAclModel add_mock_model_;
};

TEST_F(AclSessionAddTest, TestAclSession_OneTime_Success) {
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

TEST_F(AclSessionAddTest, TestAclSession_MutilTimes_Success) {
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

TEST_F(AclSessionAddTest, TestAclSession_DeviceRunMode_OneTime_Success) {
  SetDeviceRunMode();
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

TEST_F(AclSessionAddTest, TestAclSession_DeviceRunMode_MutilTimes_Success) {
  SetDeviceRunMode();
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

}  // namespace serving
}  // namespace mindspore