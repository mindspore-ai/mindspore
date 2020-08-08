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

#ifndef MINDSPORE_ACL_SESSION_TEST_COMMON_H
#define MINDSPORE_ACL_SESSION_TEST_COMMON_H

#include "common/common_test.h"
#include "serving/core/server.h"
#include "include/inference.h"
#include "include/infer_tensor.h"
#include "serving/core/serving_tensor.h"
#include "serving/acl/acl_session.h"
#include "serving/acl/model_process.h"
#include "serving/acl/dvpp_process.h"
#include "acl_stub.h"

class MockDeviceRunMode : public AclRunMode {
 public:
  aclError aclrtGetRunMode(aclrtRunMode *runMode) override {
    *runMode = aclrtRunMode::ACL_DEVICE;
    return ACL_ERROR_NONE;
  }
};

class AclSessionTest : public testing::Test {
 public:
  AclSessionTest() = default;
  void SetUp() override {
    g_acl_data_buffer = &g_acl_data_buffer_default;
    g_acl_env = &g_acl_env_default;
    g_acl_dataset = &g_acl_dataset_default;
    g_acl_model = &g_acl_model_default;
    g_acl_model_desc = &g_acl_model_desc_default;
    g_acl_device_context_stream = &g_acl_device_context_stream_default;
    g_acl_memory = &g_acl_memory_default;
    g_acl_dvpp_pic_desc = &g_acl_dvpp_pic_desc_default;
    g_acl_dvpp_roi_config = &g_acl_dvpp_roi_config_default;
    g_acl_dvpp_resize_config = &g_acl_dvpp_resize_config_default;
    g_acl_dvpp_channel_desc = &g_acl_dvpp_channel_desc_default;
    g_acl_dvpp_process = &g_acl_dvpp_process_default;
    g_acl_run_mode = &acl_run_mode_default;
    g_acl_jpeg_lib = &acl_jpeg_lib_default;
  }
  void TearDown() override {
    EXPECT_TRUE(g_acl_data_buffer->Check());
    EXPECT_TRUE(g_acl_env->Check());
    EXPECT_TRUE(g_acl_dataset->Check());
    EXPECT_TRUE(g_acl_model->Check());
    EXPECT_TRUE(g_acl_model_desc->Check());
    EXPECT_TRUE(g_acl_device_context_stream->Check());
    EXPECT_TRUE(g_acl_memory->Check());
    EXPECT_TRUE(g_acl_dvpp_pic_desc->Check());
    EXPECT_TRUE(g_acl_dvpp_roi_config->Check());
    EXPECT_TRUE(g_acl_dvpp_resize_config->Check());
    EXPECT_TRUE(g_acl_dvpp_channel_desc->Check());
    EXPECT_TRUE(g_acl_dvpp_process->Check());
    EXPECT_TRUE(g_acl_jpeg_lib->Check());
  }

  AclDataBuffer g_acl_data_buffer_default;
  AclEnv g_acl_env_default;
  AclDataSet g_acl_dataset_default;
  AclModel g_acl_model_default;
  AclModelDesc g_acl_model_desc_default;
  AclDeviceContextStream g_acl_device_context_stream_default;
  AclMemory g_acl_memory_default;
  AclDvppPicDesc g_acl_dvpp_pic_desc_default;
  AclDvppRoiConfig g_acl_dvpp_roi_config_default;
  AclDvppResizeConfig g_acl_dvpp_resize_config_default;
  AclDvppChannelDesc g_acl_dvpp_channel_desc_default;
  AclDvppProcess g_acl_dvpp_process_default;
  AclRunMode acl_run_mode_default;
  MockDeviceRunMode acl_device_run_mode;
  AclJpegLib acl_jpeg_lib_default = AclJpegLib(0, 0);

  void SetDeviceRunMode() { g_acl_run_mode = &acl_device_run_mode; }
  void CreateTensor(ms_serving::Tensor &tensor, const std::vector<int64_t> &shape, ms_serving::DataType data_type,
                    std::size_t data_size = INT64_MAX) {
    if (data_size == INT64_MAX) {
      data_size = GetDataTypeSize(data_type);
      for (auto item : shape) {
        data_size *= item;
      }
    }
    tensor.set_data(std::string(data_size, 0));
    tensor.set_tensor_type(data_type);
    auto tensor_shape = tensor.mutable_tensor_shape();
    for (auto item : shape) {
      tensor_shape->add_dims(item);
    }
  }

  size_t GetDataTypeSize(ms_serving::DataType data_type) {
    const std::map<ms_serving::DataType, size_t> type_size_map{
      {ms_serving::DataType::MS_BOOL, sizeof(bool)},       {ms_serving::DataType::MS_INT8, sizeof(int8_t)},
      {ms_serving::DataType::MS_UINT8, sizeof(uint8_t)},   {ms_serving::DataType::MS_INT16, sizeof(int16_t)},
      {ms_serving::DataType::MS_UINT16, sizeof(uint16_t)}, {ms_serving::DataType::MS_INT32, sizeof(int32_t)},
      {ms_serving::DataType::MS_UINT32, sizeof(uint32_t)}, {ms_serving::DataType::MS_INT64, sizeof(int64_t)},
      {ms_serving::DataType::MS_UINT64, sizeof(uint64_t)}, {ms_serving::DataType::MS_FLOAT16, 2},
      {ms_serving::DataType::MS_FLOAT32, sizeof(float)},   {ms_serving::DataType::MS_FLOAT64, sizeof(double)},
    };
    auto it = type_size_map.find(data_type);
    if (it == type_size_map.end()) {
      EXPECT_TRUE(false);
      return 0;
    }
    return it->second;
  }

  void CheckTensorItem(const ms_serving::Tensor &tensor, const std::vector<int64_t> &expect_shape,
                       ms_serving::DataType expect_data_type) {
    std::vector<int64_t> tensor_shape;
    for (auto item : tensor.tensor_shape().dims()) {
      tensor_shape.push_back(item);
    }
    EXPECT_EQ(expect_shape, tensor_shape);
    EXPECT_EQ(expect_data_type, tensor.tensor_type());
    int64_t elem_cnt = 1;
    for (auto item : expect_shape) {
      elem_cnt *= item;
    }
    auto data_size = GetDataTypeSize(expect_data_type);
    EXPECT_EQ(data_size * elem_cnt, tensor.data().size());
  }
};

class MockModelDesc : public AclModelDesc {
 public:
  MockModelDesc() {}
  MockModelDesc(const aclmdlDesc &mock_model_desc) : mock_model_desc_(mock_model_desc) {}
  aclmdlDesc *aclmdlCreateDesc() override {
    aclmdlDesc *model_desc = AclModelDesc::aclmdlCreateDesc();
    *model_desc = mock_model_desc_;
    return model_desc;
  }
  aclmdlDesc mock_model_desc_;
};

class AddMockAclModel : public AclModel {
 public:
  aclError aclmdlExecute(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output) override {
    if (AclModel::aclmdlExecute(modelId, input, output) != ACL_ERROR_NONE) {
      return 1;
    }
    if (input->data_buffers.size() != 2) {
      return 1;
    }
    auto &input0 = input->data_buffers[0];
    auto &input1 = input->data_buffers[1];
    std::size_t expect_count = input0->size / sizeof(float);
    if (input0->size != expect_count * sizeof(float) || input1->size != expect_count * sizeof(float)) {
      return 1;
    }

    if (output->data_buffers.size() != 1) {
      return 1;
    }
    auto &output0 = output->data_buffers[0];
    if (output0->size != expect_count * sizeof(float)) {
      return 1;
    }

    auto input0_data = reinterpret_cast<const float *>(input0->data);
    auto input1_data = reinterpret_cast<const float *>(input1->data);
    auto output0_data = reinterpret_cast<float *>(output0->data);
    for (size_t i = 0; i < expect_count; i++) {
      output0_data[i] = input0_data[i] + input1_data[i];
    }
    return ACL_ERROR_NONE;
  }

  aclError aclmdlExecuteAsync(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output,
                              aclrtStream stream) override {
    return aclmdlExecute(modelId, input, output);
  }
};

#endif  // MINDSPORE_ACL_SESSION_TEST_COMMON_H
