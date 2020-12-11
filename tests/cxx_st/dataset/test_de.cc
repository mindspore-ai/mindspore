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

using namespace mindspore::api;
using namespace mindspore::dataset::vision;

class TestDE : public ST::Common {
 public:
  TestDE() {}
};

TEST_F(TestDE, ResNetPreprocess) {
  std::vector<std::shared_ptr<Tensor>> images;
  MindDataEager::LoadImageFromDir("/home/workspace/mindspore_dataset/imagenet/imagenet_original/val/n01440764",
                                  &images);

  MindDataEager Compose({Decode(), Resize({224, 224}),
                         Normalize({0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}),
                         HWC2CHW()});

  for (auto &img : images) {
    img = Compose(img);
  }

  ASSERT_EQ(images[0]->Shape().size(), 3);
  ASSERT_EQ(images[0]->Shape()[0], 3);
  ASSERT_EQ(images[0]->Shape()[1], 224);
  ASSERT_EQ(images[0]->Shape()[2], 224);
}

TEST_F(TestDE, TestDvpp) {
  std::vector<std::shared_ptr<Tensor>> images;
  MindDataEager::LoadImageFromDir("/root/Dvpp_Unit_Dev/val2014_test/", &images);

  MindDataEager Solo({DvppDecodeResizeCropJpeg({224, 224}, {256, 256})});

  for (auto &img : images) {
    img = Solo(img);
  }

  ASSERT_EQ(images[0]->Shape().size(), 3);
  ASSERT_EQ(images[0]->Shape()[0], 224 * 224 * 1.5);
  ASSERT_EQ(images[0]->Shape()[1], 1);
  ASSERT_EQ(images[0]->Shape()[2], 1);
}
