/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include <iostream>
#include <string>
#include <vector>
#include "common/common_test.h"
#include "include/api/types.h"
#include "minddata/dataset/include/dataset/execute.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/include/dataset/vision.h"
#ifdef ENABLE_ACL
#include "minddata/dataset/include/dataset/vision_ascend.h"
#endif
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/path.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/context.h"

using namespace mindspore;
using namespace mindspore::dataset;
using namespace mindspore::dataset::vision;

class TestDE : public ST::Common {
 public:
  TestDE() {}
};

mindspore::MSTensor ReadFileToTensor(const std::string &file) {
  if (file.empty()) {
    std::cout << "[ERROR]Pointer file is nullptr, return an empty Tensor." << std::endl;
    return mindspore::MSTensor();
  }
  std::ifstream ifs(file);
  if (!ifs.good()) {
    std::cout << "[ERROR]File: " << file << " does not exist, return an empty Tensor." << std::endl;
    return mindspore::MSTensor();
  }
  if (!ifs.is_open()) {
    std::cout << "[ERROR]File: " << file << "open failed, return an empty Tensor." << std::endl;
    return mindspore::MSTensor();
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  mindspore::MSTensor buf("file", mindspore::DataType::kNumberTypeUInt8, {static_cast<int64_t>(size)}, nullptr, size);

  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char *>(buf.MutableData()), size);
  ifs.close();

  return buf;
}

TEST_F(TestDE, TestResNetPreprocess) {
  // Read images
  auto image = ReadFileToTensor("./data/dataset/apple.jpg");

  // Define transform operations
  std::shared_ptr<TensorTransform> decode(new vision::Decode());
  std::shared_ptr<TensorTransform> resize(new vision::Resize({224, 224}));
  std::shared_ptr<TensorTransform> normalize(
    new vision::Normalize({0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}));
  std::shared_ptr<TensorTransform> hwc2chw(new vision::HWC2CHW());

  mindspore::dataset::Execute Transform({decode, resize, normalize, hwc2chw});

  // Apply transform on images
  Status rc = Transform(image, &image);

  // Check image info
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(image.Shape().size(), 3);
  ASSERT_EQ(image.Shape()[0], 3);
  ASSERT_EQ(image.Shape()[1], 224);
  ASSERT_EQ(image.Shape()[2], 224);
}

TEST_F(TestDE, TestDvpp) {
#ifdef ENABLE_ACL
  // Read images from target directory

  /* Old internal method, we deprecate it
  std::shared_ptr<mindspore::dataset::Tensor> de_tensor;
  Status rc = mindspore::dataset::Tensor::CreateFromFile("./data/dataset/apple.jpg", &de_tensor);
  ASSERT_TRUE(rc.IsOk());
  auto image = MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));
  */
  auto context = ContextAutoSet();
  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->MutableDeviceInfo().size() == 1);
  auto ascend310_info = context->MutableDeviceInfo()[0]->Cast<AscendDeviceInfo>();
  ASSERT_TRUE(ascend310_info != nullptr);
  auto device_id = ascend310_info->GetDeviceID();

  auto image = ReadFileToTensor("./data/dataset/apple.jpg");

  // Define dvpp transform
  std::vector<uint32_t> crop_paras = {224, 224};
  std::vector<uint32_t> resize_paras = {256, 256};
  std::shared_ptr<TensorTransform> decode_resize_crop(new vision::DvppDecodeResizeCropJpeg(crop_paras, resize_paras));
  mindspore::dataset::Execute Transform(decode_resize_crop, MapTargetDevice::kAscend310, device_id);

  // Apply transform on images
  Status rc = Transform(image, &image);
  std::string aipp_cfg = Transform.AippCfgGenerator();
  ASSERT_EQ(aipp_cfg, "./aipp.cfg");

  // Check image info
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(image.Shape().size(), 2);
  int32_t real_h = 0;
  int32_t real_w = 0;
  int32_t remainder = crop_paras[crop_paras.size() - 1] % 16;
  if (crop_paras.size() == 1) {
    real_h = (crop_paras[0] % 2 == 0) ? crop_paras[0] : crop_paras[0] + 1;
    real_w = (remainder == 0) ? crop_paras[0] : crop_paras[0] + 16 - remainder;
  } else {
    real_h = (crop_paras[0] % 2 == 0) ? crop_paras[0] : crop_paras[0] + 1;
    real_w = (remainder == 0) ? crop_paras[1] : crop_paras[1] + 16 - remainder;
  }

  ASSERT_EQ(image.Shape()[0], real_h);  // For image in YUV format, each pixel takes 1.5 byte
  ASSERT_EQ(image.Shape()[1], real_w);
  ASSERT_EQ(image.DataSize(), real_h * real_w * 1.5);

  ASSERT_TRUE(image.Data().get() != nullptr);
  ASSERT_EQ(image.DataType(), mindspore::DataType::kNumberTypeUInt8);
  ASSERT_EQ(image.IsDevice(), true);

  /* This is the criterion for previous method(Without pop)
  ASSERT_EQ(image.Shape()[0], 1.5 * real_h * real_w);  // For image in YUV format, each pixel takes 1.5 byte
  ASSERT_EQ(image.Shape()[1], 1);
  ASSERT_EQ(image.Shape()[2], 1);
  ASSERT_EQ(image.DataSize(), real_h * real_w * 1.5);
  */
#endif
}

TEST_F(TestDE, TestDvppSinkMode) {
#ifdef ENABLE_ACL
  auto context = ContextAutoSet();
  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->MutableDeviceInfo().size() == 1);
  auto ascend310_info = context->MutableDeviceInfo()[0]->Cast<AscendDeviceInfo>();
  ASSERT_TRUE(ascend310_info != nullptr);
  auto device_id = ascend310_info->GetDeviceID();

  // Read images from target directory
  auto image = ReadFileToTensor("./data/dataset/apple.jpg");

  // Define dvpp transform
  std::vector<int32_t> crop_paras = {224, 224};
  std::vector<int32_t> resize_paras = {256};
  std::shared_ptr<TensorTransform> decode(new vision::Decode());
  std::shared_ptr<TensorTransform> resize(new vision::Resize(resize_paras));
  std::shared_ptr<TensorTransform> centercrop(new vision::CenterCrop(crop_paras));
  std::vector<std::shared_ptr<TensorTransform>> trans_list = {decode, resize, centercrop};
  mindspore::dataset::Execute Transform(trans_list, MapTargetDevice::kAscend310, device_id);

  // Apply transform on images
  Status rc = Transform(image, &image);

  // Check image info
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(image.Shape().size(), 2);
  int32_t real_h = 0;
  int32_t real_w = 0;
  int32_t remainder = crop_paras[crop_paras.size() - 1] % 16;
  if (crop_paras.size() == 1) {
    real_h = (crop_paras[0] % 2 == 0) ? crop_paras[0] : crop_paras[0] + 1;
    real_w = (remainder == 0) ? crop_paras[0] : crop_paras[0] + 16 - remainder;
  } else {
    real_h = (crop_paras[0] % 2 == 0) ? crop_paras[0] : crop_paras[0] + 1;
    real_w = (remainder == 0) ? crop_paras[1] : crop_paras[1] + 16 - remainder;
  }
  ASSERT_EQ(image.Shape()[0], real_h);  // For image in YUV format, each pixel takes 1.5 byte
  ASSERT_EQ(image.Shape()[1], real_w);
  ASSERT_EQ(image.DataSize(), real_h * real_w * 1.5);

  ASSERT_TRUE(image.Data().get() != nullptr);
  ASSERT_EQ(image.DataType(), mindspore::DataType::kNumberTypeUInt8);
  ASSERT_EQ(image.IsDevice(), true);
  Transform.DeviceMemoryRelease();
#endif
}

TEST_F(TestDE, TestDvppDecodeResizeCropNormalize) {
#ifdef ENABLE_ACL
  auto context = ContextAutoSet();
  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->MutableDeviceInfo().size() == 1);
  auto ascend310_info = context->MutableDeviceInfo()[0]->Cast<AscendDeviceInfo>();
  ASSERT_TRUE(ascend310_info != nullptr);
  auto device_id = ascend310_info->GetDeviceID();

  auto image = ReadFileToTensor("./data/dataset/apple.jpg");

  // Define dvpp transform
  std::vector<int32_t> crop_paras = {416};
  std::vector<int32_t> resize_paras = {512};
  std::vector<float> mean = {0.485 * 255, 0.456 * 255, 0.406 * 255};
  std::vector<float> std = {0.229 * 255, 0.224 * 255, 0.225 * 255};

  std::shared_ptr<TensorTransform> decode(new vision::Decode());
  std::shared_ptr<TensorTransform> resize(new vision::Resize(resize_paras));
  std::shared_ptr<TensorTransform> centercrop(new vision::CenterCrop(crop_paras));
  std::shared_ptr<TensorTransform> normalize(new vision::Normalize(mean, std));

  std::vector<std::shared_ptr<TensorTransform>> trans_list = {decode, resize, centercrop, normalize};
  mindspore::dataset::Execute Transform(trans_list, MapTargetDevice::kAscend310, device_id);

  std::string aipp_cfg = Transform.AippCfgGenerator();
  ASSERT_EQ(aipp_cfg, "./aipp.cfg");

  // Apply transform on images
  Status rc = Transform(image, &image);

  // Check image info
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(image.Shape().size(), 2);
  int32_t real_h = 0;
  int32_t real_w = 0;
  int32_t remainder = crop_paras[crop_paras.size() - 1] % 16;
  if (crop_paras.size() == 1) {
    real_h = (crop_paras[0] % 2 == 0) ? crop_paras[0] : crop_paras[0] + 1;
    real_w = (remainder == 0) ? crop_paras[0] : crop_paras[0] + 16 - remainder;
  } else {
    real_h = (crop_paras[0] % 2 == 0) ? crop_paras[0] : crop_paras[0] + 1;
    real_w = (remainder == 0) ? crop_paras[1] : crop_paras[1] + 16 - remainder;
  }

  ASSERT_EQ(image.Shape()[0], real_h);  // For image in YUV format, each pixel takes 1.5 byte
  ASSERT_EQ(image.Shape()[1], real_w);
  ASSERT_EQ(image.DataSize(), real_h * real_w * 1.5);

  ASSERT_TRUE(image.Data().get() != nullptr);
  ASSERT_EQ(image.DataType(), mindspore::DataType::kNumberTypeUInt8);
  ASSERT_EQ(image.IsDevice(), true);
  Transform.DeviceMemoryRelease();
#endif
}

/// Feature: Dvpp decode video
/// Description: 8 frames H265 video
/// Expectation: success
TEST_F(TestDE, TestDvppDecodeVideoH265) {
#ifdef ENABLE_ACL
  auto context = ContextAutoSet();
  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->MutableDeviceInfo().size() == 1);
  auto ascend310_info = context->MutableDeviceInfo()[0]->Cast<AscendDeviceInfo>();
  ASSERT_TRUE(ascend310_info != nullptr);
  auto device_id = ascend310_info->GetDeviceID();

  auto video = ReadFileToTensor("./data/dataset/FHD_8frame_1920x1080.h265");

  // Define dvpp transform
  std::vector<uint32_t> size = {1080, 1920};
  std::string path = "./output/frames";
  Path output(path);
  auto remove_all = [](Path &path) {
      std::vector<Path> all_files;
      auto dir_it = Path::DirIterator::OpenDirectory(&path);
      while(dir_it !=nullptr && dir_it->HasNext()) {
          auto file = dir_it->Next();
          all_files.emplace_back(file);
      }
      for(auto f: all_files) {
          f.Remove();
      }
  };

  remove_all(output);
  std::shared_ptr<TensorTransform> decode(new vision::DvppDecodeVideo(size, VdecStreamFormat::kH265MainLevel, path));
  mindspore::dataset::Execute Transform({decode}, MapTargetDevice::kCpu, device_id);

  // Apply transform on video
  Status rc = Transform(video, &video);
  ASSERT_TRUE(rc.IsOk());

  int count = 0;
  auto dir_it = Path::DirIterator::OpenDirectory(&output);
  while(dir_it !=nullptr && dir_it->HasNext()) {
      auto file = dir_it->Next();
      count ++;
  }
  ASSERT_EQ(count, 8);
  remove_all(output);
#endif
}
