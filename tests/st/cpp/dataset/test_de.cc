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
#include <string>
#include <vector>
#include "common/common_test.h"
#include "include/api/types.h"
#include "minddata/dataset/include/execute.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision.h"
#ifdef ENABLE_ACL
#include "minddata/dataset/include/vision_ascend.h"
#endif
#include "minddata/dataset/kernels/tensor_op.h"
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

TEST_F(TestDE, TestResNetPreprocess) {
  // Read images
  std::shared_ptr<mindspore::dataset::Tensor> de_tensor;
  mindspore::dataset::Tensor::CreateFromFile("./data/dataset/apple.jpg", &de_tensor);
  auto image = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));

  // Define transform operations
  auto decode(new vision::Decode());
  auto resize(new vision::Resize({224, 224}));
  auto normalize(
    new vision::Normalize({0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}));
  auto hwc2chw(new vision::HWC2CHW());

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
  std::shared_ptr<mindspore::dataset::Tensor> de_tensor;
  mindspore::dataset::Tensor::CreateFromFile("./data/dataset/apple.jpg", &de_tensor);
  auto image = MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));

  // Define dvpp transform
  std::vector<uint32_t> crop_paras = {224, 224};
  std::vector<uint32_t> resize_paras = {256, 256};
  mindspore::dataset::Execute Transform(DvppDecodeResizeCropJpeg(crop_paras, resize_paras));

  // Apply transform on images
  Status rc = Transform(image, &image);

  // Check image info
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(image.Shape().size(), 3);
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
  ASSERT_EQ(image.Shape()[0], real_h * real_w * 1.5);  // For image in YUV format, each pixel takes 1.5 byte
  ASSERT_EQ(image.Shape()[1], 1);
  ASSERT_EQ(image.Shape()[2], 1);
#endif
}

TEST_F(TestDE, TestDvppSinkMode) {
#ifdef ENABLE_ACL
  // Read images from target directory
  std::shared_ptr<mindspore::dataset::Tensor> de_tensor;
  mindspore::dataset::Tensor::CreateFromFile("./data/dataset/apple.jpg", &de_tensor);
  auto image = MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));

  // Define dvpp transform
  std::vector<uint32_t> crop_paras = {224, 224};
  std::vector<uint32_t> resize_paras = {256};
  mindspore::dataset::Execute Transform({DvppDecodeJpeg(), DvppResizeJpeg(resize_paras), DvppCropJpeg(crop_paras)},
                                        "Ascend310");

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
  ASSERT_EQ(image.DataSize(), 1.5 * real_w * real_h);
  Transform.DeviceMemoryRelease();
#endif
}

TEST_F(TestDE, TestDvppDecodeResizeCrop) {
#ifdef ENABLE_ACL
  std::shared_ptr<mindspore::dataset::Tensor> de_tensor;
  mindspore::dataset::Tensor::CreateFromFile("./data/dataset/apple.jpg", &de_tensor);
  auto image = MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));

  // Define dvpp transform
  std::vector<uint32_t> crop_paras = {416};
  std::vector<uint32_t> resize_paras = {512};
  mindspore::dataset::Execute Transform(DvppDecodeResizeCropJpeg(crop_paras, resize_paras), "Ascend310");

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
  ASSERT_EQ(image.DataSize(), 1.5 * real_w * real_h);
  Transform.DeviceMemoryRelease();
#endif
}
