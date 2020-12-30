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
#include "minddata/dataset/include/minddata_eager.h"
#include "minddata/dataset/include/vision.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/context.h"

using namespace mindspore::api;
using namespace mindspore::dataset::vision;

class TestDE : public ST::Common {
 public:
  TestDE() {}
};

TEST_F(TestDE, TestResNetPreprocess) {
  // Read images from target directory
  std::vector<std::shared_ptr<Tensor>> images;
  MindDataEager::LoadImageFromDir("/home/workspace/mindspore_dataset/imagenet/imagenet_original/val/n01440764",
                                  &images);

  // Define transform operations
  MindDataEager Transform({Decode(), Resize({224, 224}),
                           Normalize({0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}),
                           HWC2CHW()});

  // Apply transform on images
  for (auto &img : images) {
    img = Transform(img);
  }

  // Check shape of result
  ASSERT_NE(images.size(), 0);
  ASSERT_EQ(images[0]->Shape().size(), 3);
  ASSERT_EQ(images[0]->Shape()[0], 3);
  ASSERT_EQ(images[0]->Shape()[1], 224);
  ASSERT_EQ(images[0]->Shape()[2], 224);
}

TEST_F(TestDE, TestDvpp) {
  ContextAutoSet();

  // Read images from target directory
  std::vector<std::shared_ptr<Tensor>> images;
  MindDataEager::LoadImageFromDir("/home/workspace/mindspore_dataset/imagenet/imagenet_original/val/n01440764",
                                  &images);

  // Define dvpp transform
  std::vector<uint32_t> crop_size = {224, 224};
  std::vector<uint32_t> resize_size = {256, 256};
  MindDataEager Transform({DvppDecodeResizeCropJpeg(crop_size, resize_size)});

  // Apply transform on images
  for (auto &img : images) {
    img = Transform(img);
    ASSERT_NE(img, nullptr);
    ASSERT_EQ(img->Shape().size(), 3);
    int32_t real_h = 0;
    int32_t real_w = 0;
    int32_t remainder = crop_size[crop_size.size() - 1] % 16;
    if (crop_size.size() == 1) {
      real_h = (crop_size[0] % 2 == 0) ? crop_size[0] : crop_size[0] + 1;
      real_w = (remainder == 0) ? crop_size[0] : crop_size[0] + 16 - remainder;
    } else {
      real_h = (crop_size[0] % 2 == 0) ? crop_size[0] : crop_size[0] + 1;
      real_w = (remainder == 0) ? crop_size[1] : crop_size[1] + 16 - remainder;
    }
    ASSERT_EQ(img->Shape()[0], real_h * real_w * 1.5);  // For image in YUV format, each pixel takes 1.5 byte
    ASSERT_EQ(img->Shape()[1], 1);
    ASSERT_EQ(img->Shape()[2], 1);
  }
}
