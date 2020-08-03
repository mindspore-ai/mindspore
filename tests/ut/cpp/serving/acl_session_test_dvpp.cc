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
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

using namespace std;
using namespace mindspore::inference;

namespace mindspore {
namespace serving {

class MockDvppProces : public AclDvppProcess {
 public:
  aclError acldvppVpcResizeAsync(acldvppChannelDesc *channelDesc, acldvppPicDesc *inputDesc, acldvppPicDesc *outputDesc,
                                 acldvppResizeConfig *resizeConfig, aclrtStream stream) override {
    if (resize_fail_list_.empty()) {
      return AclDvppProcess::acldvppVpcResizeAsync(channelDesc, inputDesc, outputDesc, resizeConfig, stream);
    }
    bool val = resize_fail_list_.front();
    resize_fail_list_.erase(resize_fail_list_.begin());
    if (!val) {
      return 1;
    }
    return AclDvppProcess::acldvppVpcResizeAsync(channelDesc, inputDesc, outputDesc, resizeConfig, stream);
  }
  aclError acldvppVpcCropAsync(acldvppChannelDesc *channelDesc, acldvppPicDesc *inputDesc, acldvppPicDesc *outputDesc,
                               acldvppRoiConfig *cropArea, aclrtStream stream) override {
    if (crop_fail_list_.empty()) {
      return AclDvppProcess::acldvppVpcCropAsync(channelDesc, inputDesc, outputDesc, cropArea, stream);
    }
    bool val = crop_fail_list_.front();
    crop_fail_list_.erase(crop_fail_list_.begin());
    if (!val) {
      return 1;
    }
    return AclDvppProcess::acldvppVpcCropAsync(channelDesc, inputDesc, outputDesc, cropArea, stream);
  }
  aclError acldvppVpcCropAndPasteAsync(acldvppChannelDesc *channelDesc, acldvppPicDesc *inputDesc,
                                       acldvppPicDesc *outputDesc, acldvppRoiConfig *cropArea,
                                       acldvppRoiConfig *pasteArea, aclrtStream stream) override {
    if (crop_and_paste_fail_list_.empty()) {
      return AclDvppProcess::acldvppVpcCropAndPasteAsync(channelDesc, inputDesc, outputDesc, cropArea, pasteArea,
                                                         stream);
    }
    bool val = crop_and_paste_fail_list_.front();
    crop_and_paste_fail_list_.erase(crop_and_paste_fail_list_.begin());
    if (!val) {
      return 1;
    }
    return AclDvppProcess::acldvppVpcCropAndPasteAsync(channelDesc, inputDesc, outputDesc, cropArea, pasteArea, stream);
  }
  aclError acldvppJpegDecodeAsync(acldvppChannelDesc *channelDesc, const void *data, uint32_t size,
                                  acldvppPicDesc *outputDesc, aclrtStream stream) override {
    if (decode_fail_list_.empty()) {
      return AclDvppProcess::acldvppJpegDecodeAsync(channelDesc, data, size, outputDesc, stream);
    }
    bool val = decode_fail_list_.front();
    decode_fail_list_.erase(decode_fail_list_.begin());
    if (!val) {
      return 1;
    }
    return AclDvppProcess::acldvppJpegDecodeAsync(channelDesc, data, size, outputDesc, stream);
  }
  vector<bool> decode_fail_list_;
  vector<bool> resize_fail_list_;
  vector<bool> crop_fail_list_;
  vector<bool> crop_and_paste_fail_list_;
};

class AclSessionDvppTest : public AclSessionTest {
 public:
  AclSessionDvppTest() = default;
  void SetUp() override { AclSessionTest::SetUp(); }
  void InitModelDesc(uint32_t batch_size) {
    batch_size_ = batch_size;
    aclmdlDesc model_desc;
    model_desc.inputs.push_back(  // 32-> 16 align, 24->2 align
      AclTensorDesc{
        .dims = {batch_size_, 32, 24, 3}, .data_type = ACL_FLOAT, .size = batch_size_ * 32 * 24 * 3 / 2});  // YUV420SP

    model_desc.outputs.push_back(AclTensorDesc{
      .dims = {batch_size_, 24, 24, 3}, .data_type = ACL_FLOAT, .size = batch_size_ * 24 * 24 * 3 * sizeof(float)});

    model_desc.outputs.push_back(AclTensorDesc{
      .dims = {batch_size_, 24, 24, 3}, .data_type = ACL_FLOAT, .size = batch_size_ * 24 * 24 * 3 * sizeof(float)});

    mock_model_desc_ = MockModelDesc(model_desc);
    g_acl_model_desc = &mock_model_desc_;
    g_acl_dvpp_process = &mock_dvpp_process_;
  }
  void TearDown() override {
    AclSessionTest::TearDown();
    remove(dvpp_config_file_path_.c_str());
  }
  void CreateDefaultRequest(PredictRequest &request, uint32_t image_size = 1) {
    auto input0 = request.add_images();
    for (uint32_t i = 0; i < batch_size_; i++) {
      input0->add_images(std::string(image_size, '\0'));  // any length data
    }
  }

  void CheckDefaultReply(const PredictReply &reply) {
    EXPECT_TRUE(reply.result().size() == 2);
    if (reply.result().size() == 2) {
      CheckTensorItem(reply.result(0), {batch_size_, 24, 24, 3}, ::ms_serving::DataType::MS_FLOAT32);
      CheckTensorItem(reply.result(1), {batch_size_, 24, 24, 3}, ::ms_serving::DataType::MS_FLOAT32);
    }
  }

  void WriteDvppConfig(const std::string &dvpp_config_context) {
    std::ofstream fp(dvpp_config_file_path_);
    ASSERT_TRUE(fp.is_open());
    if (fp.is_open()) {
      fp << dvpp_config_context;
    }
  }

  void SetJpegLib(uint32_t image_width, uint32_t image_height, J_COLOR_SPACE color_space = JCS_YCbCr) {
    acl_jpeg_lib_default.image_width_ = image_width;
    acl_jpeg_lib_default.image_height_ = image_height;
    acl_jpeg_lib_default.color_space_ = color_space;
  }

  void CreateDvppConfig() {
    nlohmann::json dvpp_config;
    auto &preprocess_list = dvpp_config["preprocess"];
    auto &preprocess = preprocess_list[0];
    preprocess["input"]["index"] = 0;
    preprocess["decode_para"]["out_pixel_format"] = pixel_format_;
    auto &dvpp_process = preprocess["dvpp_process"];

    if (to_resize_flag_) {
      dvpp_process["op_name"] = "resize";
      dvpp_process["out_width"] = resize_para_.output_width;
      dvpp_process["out_height"] = resize_para_.output_height;
    } else if (to_crop_flag_) {
      auto &crop_info = crop_para_.crop_info;
      auto &crop_area = crop_info.crop_area;
      dvpp_process["op_name"] = "crop";
      dvpp_process["out_width"] = crop_para_.output_width;
      dvpp_process["out_height"] = crop_para_.output_height;
      if (crop_info.crop_type == kDvppCropTypeOffset) {
        dvpp_process["crop_type"] = "offset";
        dvpp_process["crop_left"] = crop_area.left;
        dvpp_process["crop_top"] = crop_area.top;
        dvpp_process["crop_right"] = crop_area.right;
        dvpp_process["crop_bottom"] = crop_area.bottom;
      } else {
        dvpp_process["crop_type"] = "centre";
        dvpp_process["crop_width"] = crop_info.crop_width;
        dvpp_process["crop_height"] = crop_info.crop_height;
      }
    } else if (to_crop_and_paste_flag_) {
      auto &crop_info = crop_paste_para_.crop_info;
      auto &crop_area = crop_info.crop_area;
      dvpp_process["op_name"] = "crop_and_paste";
      dvpp_process["out_width"] = crop_paste_para_.output_width;
      dvpp_process["out_height"] = crop_paste_para_.output_height;

      dvpp_process["paste_left"] = crop_paste_para_.paste_area.left;
      dvpp_process["paste_right"] = crop_paste_para_.paste_area.right;
      dvpp_process["paste_top"] = crop_paste_para_.paste_area.top;
      dvpp_process["paste_bottom"] = crop_paste_para_.paste_area.bottom;

      if (crop_info.crop_type == kDvppCropTypeOffset) {
        dvpp_process["crop_type"] = "offset";
        dvpp_process["crop_left"] = crop_area.left;
        dvpp_process["crop_top"] = crop_area.top;
        dvpp_process["crop_right"] = crop_area.right;
        dvpp_process["crop_bottom"] = crop_area.bottom;
      } else {
        dvpp_process["crop_type"] = "centre";
        dvpp_process["crop_width"] = crop_info.crop_width;
        dvpp_process["crop_height"] = crop_info.crop_height;
      }
    }
    stringstream output;
    output << dvpp_config;
    WriteDvppConfig(output.str());
  }
  uint32_t batch_size_ = 1;
  MockModelDesc mock_model_desc_;
  MockDvppProces mock_dvpp_process_;

  const std::string model_file_path_ = "/tmp/acl_model_fake_path.om";
  const std::string dvpp_config_file_path_ = "/tmp/acl_model_fake_path_dvpp_config.json";
  inference::DvppResizePara resize_para_;
  inference::DvppCropPara crop_para_;
  inference::DvppCropAndPastePara crop_paste_para_;
  bool to_resize_flag_ = false;
  bool to_crop_flag_ = false;
  bool to_crop_and_paste_flag_ = false;
  std::string pixel_format_;
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppResize_BatchSize1_Success) {
  pixel_format_ = "YUV420SP";
  to_resize_flag_ = true;
  SetJpegLib(128, 128);            // 32*32 ~ 4096*4096
  resize_para_.output_width = 24;  // align to 32
  resize_para_.output_height = 24;
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);
  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingImagesRequest serving_images(request);
  ServingReply serving_reply(reply);
  EXPECT_TRUE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);
  CheckDefaultReply(reply);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->resize_call_times_, 1);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 1);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppResize_BatchSize3_Success) {
  pixel_format_ = "YUV420SP";
  to_resize_flag_ = true;
  SetJpegLib(128, 128);            // 32*32 ~ 4096*4096
  resize_para_.output_width = 24;  // align to 32
  resize_para_.output_height = 24;
  CreateDvppConfig();
  InitModelDesc(3);  // batch_size=3

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);
  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingImagesRequest serving_images(request);
  ServingReply serving_reply(reply);
  EXPECT_TRUE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);
  CheckDefaultReply(reply);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->resize_call_times_, 3);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 3);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppCrop_BatchSize1_Success) {
  pixel_format_ = "YUV420SP";
  to_crop_flag_ = true;
  SetJpegLib(128, 128);          // 32*32 ~ 4096*4096
  crop_para_.output_width = 24;  // align to 32
  crop_para_.output_height = 24;
  crop_para_.crop_info.crop_type = kDvppCropTypeOffset;
  crop_para_.crop_info.crop_area.left = 0;
  crop_para_.crop_info.crop_area.right = 64;
  crop_para_.crop_info.crop_area.top = 0;
  crop_para_.crop_info.crop_area.bottom = 64;
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);
  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingImagesRequest serving_images(request);
  ServingReply serving_reply(reply);
  EXPECT_TRUE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);
  CheckDefaultReply(reply);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 1);
  EXPECT_EQ(g_acl_dvpp_process->crop_call_times_, 1);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppCropPaste_BatchSize1_Success) {
  pixel_format_ = "YUV420SP";
  to_crop_and_paste_flag_ = true;
  SetJpegLib(128, 128);                // 32*32 ~ 4096*4096
  crop_paste_para_.output_width = 24;  // align to 32
  crop_paste_para_.output_height = 24;
  crop_paste_para_.crop_info.crop_type = kDvppCropTypeOffset;
  crop_paste_para_.crop_info.crop_area.left = 0;
  crop_paste_para_.crop_info.crop_area.right = 64;
  crop_paste_para_.crop_info.crop_area.top = 0;
  crop_paste_para_.crop_info.crop_area.bottom = 64;
  crop_paste_para_.paste_area.left = 0;
  crop_paste_para_.paste_area.right = 64;
  crop_paste_para_.paste_area.top = 0;
  crop_paste_para_.paste_area.bottom = 64;
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);
  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingImagesRequest serving_images(request);
  ServingReply serving_reply(reply);
  EXPECT_TRUE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);
  CheckDefaultReply(reply);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 1);
  EXPECT_EQ(g_acl_dvpp_process->crop_paste_call_times_, 1);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppResize_BatchSize3_MultiTime_Success) {
  pixel_format_ = "YUV420SP";
  to_resize_flag_ = true;
  SetJpegLib(128, 128);            // 32*32 ~ 4096*4096
  resize_para_.output_width = 24;  // align to 32
  resize_para_.output_height = 24;
  CreateDvppConfig();
  InitModelDesc(3);  // batch_size=3

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  for (int i = 0; i < 3; i++) {
    // create inputs
    PredictRequest request;
    CreateDefaultRequest(request, i + 1);  // image size

    PredictReply reply;
    ServingRequest serving_request(request);
    ServingImagesRequest serving_images(request);
    ServingReply serving_reply(reply);
    EXPECT_TRUE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);
    CheckDefaultReply(reply);
  }

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->resize_call_times_, 3 * 3);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 3 * 3);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppResize_BatchSize3_MultiTime_SameImageSize_Success) {
  pixel_format_ = "YUV420SP";
  to_resize_flag_ = true;
  SetJpegLib(128, 128);            // 32*32 ~ 4096*4096
  resize_para_.output_width = 24;  // align to 32
  resize_para_.output_height = 24;
  CreateDvppConfig();
  InitModelDesc(3);  // batch_size=3

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  for (int i = 0; i < 3; i++) {
    // create inputs
    PredictRequest request;
    CreateDefaultRequest(request, 1);  // image size

    PredictReply reply;
    ServingRequest serving_request(request);
    ServingImagesRequest serving_images(request);
    ServingReply serving_reply(reply);
    EXPECT_TRUE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);
    CheckDefaultReply(reply);
  }

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->resize_call_times_, 3 * 3);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 3 * 3);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppResize_InvalidImageDim_Fail) {
  pixel_format_ = "YUV420SP";
  to_resize_flag_ = true;
  resize_para_.output_width = 24;  // align to 32
  resize_para_.output_height = 24;
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  SetJpegLib(31, 31);  // 32*32 ~ 4096*4096
  {
    // create inputs
    PredictRequest request;
    CreateDefaultRequest(request);

    PredictReply reply;
    ServingRequest serving_request(request);
    ServingImagesRequest serving_images(request);
    ServingReply serving_reply(reply);
    EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);
  }
  SetJpegLib(4097, 4097);  // 32*32 ~ 4096*4096
  {
    // create inputs
    PredictRequest request;
    CreateDefaultRequest(request);

    PredictReply reply;
    ServingRequest serving_request(request);
    ServingImagesRequest serving_images(request);
    ServingReply serving_reply(reply);
    EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);
  }
  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->resize_call_times_, 0);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 0);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppResize_InvalidResizeWidth_Fail) {
  pixel_format_ = "YUV420SP";
  to_resize_flag_ = true;
  resize_para_.output_width = 15;  // align to 16 16n minimum 32
  resize_para_.output_height = 24;
  SetJpegLib(128, 128);  // 32*32 ~ 4096*4096
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  // load config, check output_width failed
  EXPECT_FALSE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingImagesRequest serving_images(request);
  ServingReply serving_reply(reply);
  EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->resize_call_times_, 0);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 0);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppResize_InvalidResizeHeight_Fail) {
  pixel_format_ = "YUV420SP";
  to_resize_flag_ = true;
  resize_para_.output_width = 32;  // align to 32 16n, minimum 32
  resize_para_.output_height = 3;  // align to 4 2n, minimum 6
  SetJpegLib(128, 128);            // 32*32 ~ 4096*4096
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  // load config, check output_height failed
  EXPECT_FALSE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppCropOffset_CropMini_Success) {
  pixel_format_ = "YUV420SP";
  to_crop_flag_ = true;
  crop_para_.output_width = 24;  // align to 32 16n, minimum 32
  crop_para_.output_height = 6;  // align to 6 2n, minimum 6
  crop_para_.crop_info.crop_type = kDvppCropTypeOffset;
  crop_para_.crop_info.crop_area.left = 4;
  crop_para_.crop_info.crop_area.right = 13;
  crop_para_.crop_info.crop_area.top = 4;
  crop_para_.crop_info.crop_area.bottom = 9;

  SetJpegLib(128, 128);  // 32*32 ~ 4096*4096
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  // load config, check output_height failed
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppCropCentre_CropMini_Success) {
  pixel_format_ = "YUV420SP";
  to_crop_flag_ = true;
  crop_para_.output_width = 24;   // align to 32 16n, minimum 32
  crop_para_.output_height = 24;  // align to 24 2n, minimum 6
  crop_para_.crop_info.crop_type = kDvppCropTypeCentre;
  crop_para_.crop_info.crop_width = 10;
  crop_para_.crop_info.crop_height = 6;

  SetJpegLib(127, 127);  // 32*32 ~ 4096*4096
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  // load config, check output_height failed
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingImagesRequest serving_images(request);
  ServingReply serving_reply(reply);
  EXPECT_TRUE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->crop_call_times_, 1);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 1);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppCropOffset_InvalidCropWidth_Fail) {
  pixel_format_ = "YUV420SP";
  to_crop_flag_ = true;
  crop_para_.output_width = 24;  // align to 32 16n, minimum 32
  crop_para_.output_height = 6;  // align to 6 2n, minimum 6
  crop_para_.crop_info.crop_type = kDvppCropTypeOffset;
  crop_para_.crop_info.crop_area.left = 4;
  crop_para_.crop_info.crop_area.right = 11;  // minimum 10*6
  crop_para_.crop_info.crop_area.top = 4;
  crop_para_.crop_info.crop_area.bottom = 9;
  SetJpegLib(128, 128);  // 32*32 ~ 4096*4096
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  // load config, check crop width failed
  EXPECT_FALSE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppCropOffset_InvalidCropHeight_Fail) {
  pixel_format_ = "YUV420SP";
  to_crop_flag_ = true;
  crop_para_.output_width = 24;  // align to 32 16n, minimum 32
  crop_para_.output_height = 6;  // align to 6 2n, minimum 6
  crop_para_.crop_info.crop_type = kDvppCropTypeOffset;
  crop_para_.crop_info.crop_area.left = 4;
  crop_para_.crop_info.crop_area.right = 13;
  crop_para_.crop_info.crop_area.top = 4;
  crop_para_.crop_info.crop_area.bottom = 7;  // minimum 10*6
  SetJpegLib(128, 128);                       // 32*32 ~ 4096*4096
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  // load config, check crop height failed
  EXPECT_FALSE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppCropCentre_InvalidCropHeight_Fail) {
  pixel_format_ = "YUV420SP";
  to_crop_flag_ = true;
  crop_para_.output_width = 24;   // align to 32 16n, minimum 32
  crop_para_.output_height = 24;  // align to 24 2n, minimum 6
  crop_para_.crop_info.crop_type = kDvppCropTypeCentre;
  crop_para_.crop_info.crop_width = 10;  // minimum 10*6
  crop_para_.crop_info.crop_height = 4;
  SetJpegLib(128, 128);  // 32*32 ~ 4096*4096
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  // load config, check crop_height failed
  EXPECT_FALSE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppCropPasteOffset_CropMini_Success) {
  pixel_format_ = "YUV420SP";
  to_crop_and_paste_flag_ = true;
  crop_paste_para_.output_width = 24;   // align to 32 16n, minimum 32
  crop_paste_para_.output_height = 24;  // align to 24 2n, minimum 6
  crop_paste_para_.crop_info.crop_type = kDvppCropTypeOffset;
  crop_paste_para_.crop_info.crop_area.left = 4;
  crop_paste_para_.crop_info.crop_area.right = 13;
  crop_paste_para_.crop_info.crop_area.top = 4;
  crop_paste_para_.crop_info.crop_area.bottom = 9;
  crop_paste_para_.paste_area.left = 4;
  crop_paste_para_.paste_area.right = 13;
  crop_paste_para_.paste_area.top = 4;
  crop_paste_para_.paste_area.bottom = 9;

  SetJpegLib(128, 128);  // 32*32 ~ 4096*4096
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  // load config, check output_height failed
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingImagesRequest serving_images(request);
  ServingReply serving_reply(reply);
  EXPECT_TRUE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->crop_paste_call_times_, 1);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 1);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppCropPasteCentre_CropMini_Success) {
  pixel_format_ = "YUV420SP";
  to_crop_and_paste_flag_ = true;
  crop_paste_para_.output_width = 24;   // align to 32 16n, minimum 32
  crop_paste_para_.output_height = 24;  // align to 24 2n, minimum 6
  crop_paste_para_.crop_info.crop_type = kDvppCropTypeCentre;
  crop_paste_para_.crop_info.crop_width = 10;
  crop_paste_para_.crop_info.crop_height = 6;
  crop_paste_para_.paste_area.left = 4;
  crop_paste_para_.paste_area.right = 13;
  crop_paste_para_.paste_area.top = 4;
  crop_paste_para_.paste_area.bottom = 9;

  SetJpegLib(128, 128);  // 32*32 ~ 4096*4096
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  // load config, check output_height failed
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingImagesRequest serving_images(request);
  ServingReply serving_reply(reply);
  EXPECT_TRUE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->crop_paste_call_times_, 1);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 1);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppCropPasteCentre_InvalidPasteWidth_Fail) {
  pixel_format_ = "YUV420SP";
  to_crop_and_paste_flag_ = true;
  crop_paste_para_.output_width = 24;   // align to 32 16n, minimum 32
  crop_paste_para_.output_height = 24;  // align to 24 2n, minimum 6
  crop_paste_para_.crop_info.crop_type = kDvppCropTypeCentre;
  crop_paste_para_.crop_info.crop_width = 10;
  crop_paste_para_.crop_info.crop_height = 6;
  crop_paste_para_.paste_area.left = 4;
  crop_paste_para_.paste_area.right = 11;
  crop_paste_para_.paste_area.top = 4;
  crop_paste_para_.paste_area.bottom = 9;

  SetJpegLib(128, 128);  // 32*32 ~ 4096*4096
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  // load config, check output_height failed
  EXPECT_FALSE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppCropPasteCentre_InvalidPasteHeight_Fail) {
  pixel_format_ = "YUV420SP";
  to_crop_and_paste_flag_ = true;
  crop_paste_para_.output_width = 24;   // align to 32 16n, minimum 32
  crop_paste_para_.output_height = 24;  // align to 24 2n, minimum 6
  crop_paste_para_.crop_info.crop_type = kDvppCropTypeCentre;
  crop_paste_para_.crop_info.crop_width = 10;
  crop_paste_para_.crop_info.crop_height = 6;
  crop_paste_para_.paste_area.left = 4;
  crop_paste_para_.paste_area.right = 13;
  crop_paste_para_.paste_area.top = 4;
  crop_paste_para_.paste_area.bottom = 7;

  SetJpegLib(128, 128);  // 32*32 ~ 4096*4096
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  // load config, check output_height failed
  EXPECT_FALSE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

// dvpp proces fail, test resource release ok
TEST_F(AclSessionDvppTest, TestAclSession_DvppDecode_BatchSize1_DvppFail) {
  pixel_format_ = "YUV420SP";
  to_resize_flag_ = true;
  SetJpegLib(128, 128);            // 32*32 ~ 4096*4096
  resize_para_.output_width = 24;  // align to 32
  resize_para_.output_height = 24;
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  mock_dvpp_process_.decode_fail_list_.push_back(false);
  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingImagesRequest serving_images(request);
  ServingReply serving_reply(reply);
  EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppResize_BatchSize1_DvppFail) {
  pixel_format_ = "YUV420SP";
  to_resize_flag_ = true;
  SetJpegLib(128, 128);            // 32*32 ~ 4096*4096
  resize_para_.output_width = 24;  // align to 32
  resize_para_.output_height = 24;
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  mock_dvpp_process_.resize_fail_list_.push_back(false);
  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingImagesRequest serving_images(request);
  ServingReply serving_reply(reply);
  EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 1);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppResize_BatchSize3_DvppFail0) {
  pixel_format_ = "YUV420SP";
  to_resize_flag_ = true;
  SetJpegLib(128, 128);            // 32*32 ~ 4096*4096
  resize_para_.output_width = 24;  // align to 32
  resize_para_.output_height = 24;
  CreateDvppConfig();
  InitModelDesc(3);  // batch_size=3

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);
  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  mock_dvpp_process_.resize_fail_list_.push_back(false);  // image 0 fail
  PredictReply reply;
  ServingRequest serving_request(request);
  ServingImagesRequest serving_images(request);
  ServingReply serving_reply(reply);
  EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 1);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppResize_BatchSize3_DvppFail1) {
  pixel_format_ = "YUV420SP";
  to_resize_flag_ = true;
  SetJpegLib(128, 128);            // 32*32 ~ 4096*4096
  resize_para_.output_width = 24;  // align to 32
  resize_para_.output_height = 24;
  CreateDvppConfig();
  InitModelDesc(3);  // batch_size=3

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);
  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  mock_dvpp_process_.resize_fail_list_.push_back(true);   // image 0 success
  mock_dvpp_process_.resize_fail_list_.push_back(false);  // image 1 fail
  PredictReply reply;
  ServingRequest serving_request(request);
  ServingImagesRequest serving_images(request);
  ServingReply serving_reply(reply);
  EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 2);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppResize_BatchSize3_DvppFail2) {
  pixel_format_ = "YUV420SP";
  to_resize_flag_ = true;
  SetJpegLib(128, 128);            // 32*32 ~ 4096*4096
  resize_para_.output_width = 24;  // align to 32
  resize_para_.output_height = 24;
  CreateDvppConfig();
  InitModelDesc(3);  // batch_size=3

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);
  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  mock_dvpp_process_.resize_fail_list_.push_back(true);   // image 0 success
  mock_dvpp_process_.resize_fail_list_.push_back(true);   // image 1 success
  mock_dvpp_process_.resize_fail_list_.push_back(false);  // image 2 fail
  PredictReply reply;
  ServingRequest serving_request(request);
  ServingImagesRequest serving_images(request);
  ServingReply serving_reply(reply);
  EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 3);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppCrop_BatchSize1_DvppFail) {
  pixel_format_ = "YUV420SP";
  to_crop_flag_ = true;
  SetJpegLib(128, 128);          // 32*32 ~ 4096*4096
  crop_para_.output_width = 24;  // align to 32
  crop_para_.output_height = 24;
  crop_para_.crop_info.crop_type = kDvppCropTypeOffset;
  crop_para_.crop_info.crop_area.left = 0;
  crop_para_.crop_info.crop_area.right = 64;
  crop_para_.crop_info.crop_area.top = 0;
  crop_para_.crop_info.crop_area.bottom = 64;
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);
  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  mock_dvpp_process_.crop_fail_list_.push_back(false);
  PredictReply reply;
  ServingRequest serving_request(request);
  ServingImagesRequest serving_images(request);
  ServingReply serving_reply(reply);
  EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 1);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppCropPaste_BatchSize1_DvppFail) {
  pixel_format_ = "YUV420SP";
  to_crop_and_paste_flag_ = true;
  SetJpegLib(128, 128);                // 32*32 ~ 4096*4096
  crop_paste_para_.output_width = 24;  // align to 32
  crop_paste_para_.output_height = 24;
  crop_paste_para_.crop_info.crop_type = kDvppCropTypeOffset;
  crop_paste_para_.crop_info.crop_area.left = 0;
  crop_paste_para_.crop_info.crop_area.right = 64;
  crop_paste_para_.crop_info.crop_area.top = 0;
  crop_paste_para_.crop_info.crop_area.bottom = 64;
  crop_paste_para_.paste_area.left = 0;
  crop_paste_para_.paste_area.right = 64;
  crop_paste_para_.paste_area.top = 0;
  crop_paste_para_.paste_area.bottom = 64;
  CreateDvppConfig();
  InitModelDesc(1);  // batch_size=1

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  mock_dvpp_process_.crop_and_paste_fail_list_.push_back(false);
  // create inputs
  PredictRequest request;
  CreateDefaultRequest(request);

  PredictReply reply;
  ServingRequest serving_request(request);
  ServingImagesRequest serving_images(request);
  ServingReply serving_reply(reply);
  EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 1);
};

TEST_F(AclSessionDvppTest, TestAclSession_DvppCrop_BatchSize3_MultiTime_DvppFail) {
  pixel_format_ = "YUV420SP";
  to_crop_flag_ = true;
  SetJpegLib(128, 128);          // 32*32 ~ 4096*4096
  crop_para_.output_width = 24;  // align to 32
  crop_para_.output_height = 24;
  crop_para_.crop_info.crop_type = kDvppCropTypeCentre;
  crop_para_.crop_info.crop_width = 10;
  crop_para_.crop_info.crop_height = 6;

  CreateDvppConfig();
  InitModelDesc(3);  // batch_size=3

  inference::AclSession acl_session;
  uint32_t device_id = 1;
  EXPECT_TRUE(acl_session.InitEnv("Ascend", device_id) == SUCCESS);
  uint32_t model_id = 0;
  EXPECT_TRUE(acl_session.LoadModelFromFile(model_file_path_, model_id) == SUCCESS);

  for (int i = 0; i < 3; i++) {
    mock_dvpp_process_.crop_fail_list_.push_back(false);
    // create inputs
    PredictRequest request;
    CreateDefaultRequest(request, i + 1);  // image size

    PredictReply reply;
    ServingRequest serving_request(request);
    ServingImagesRequest serving_images(request);
    ServingReply serving_reply(reply);
    EXPECT_FALSE(acl_session.ExecuteModel(model_id, serving_images, serving_request, serving_reply) == SUCCESS);
  }

  EXPECT_TRUE(acl_session.UnloadModel(model_id) == SUCCESS);
  EXPECT_TRUE(acl_session.FinalizeEnv() == SUCCESS);
  EXPECT_EQ(g_acl_dvpp_process->decode_call_times_, 3);
};

}  // namespace serving
}  // namespace mindspore