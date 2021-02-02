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
#include <chrono>
#include "common/common_test.h"
#include "gtest/gtest.h"
#include "./securec.h"
#include "minddata/dataset/include/tensor.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision.h"
#include "minddata/dataset/include/execute.h"
#include "minddata/dataset/util/path.h"
#include "mindspore/lite/src/common/log_adapter.h"
#include "include/api/types.h"

using MSTensor = mindspore::tensor::MSTensor;
using DETensor = mindspore::tensor::DETensor;
using mindspore::dataset::vision::Decode;
using mindspore::dataset::vision::Normalize;
using mindspore::dataset::vision::Resize;
using Execute = mindspore::dataset::Execute;
using Path = mindspore::dataset::Path;

class MindDataTestEager : public mindspore::CommonTest {
 public:
  MindDataTestEager() {}
};

TEST_F(MindDataTestEager, Test1) {
#if defined(ENABLE_ARM64) || defined(ENABLE_ARM32)
  std::string in_dir = "/sdcard/data/testPK/data/class1";
#else
  std::string in_dir = "data/testPK/data/class1";
#endif
  Path base_dir = Path(in_dir);
  MS_LOG(WARNING) << base_dir.toString() << ".";
  if (!base_dir.IsDirectory() || !base_dir.Exists()) {
    MS_LOG(INFO) << "Input dir is not a directory or doesn't exist"
                 << ".";
  }
  auto t_start = std::chrono::high_resolution_clock::now();
  // check if output_dir exists and create it if it does not exist

  // iterate over in dir and create json for all images
  auto dir_it = Path::DirIterator::OpenDirectory(&base_dir);
  while (dir_it->hasNext()) {
    Path v = dir_it->next();
    // MS_LOG(WARNING) << v.toString() << ".";
    std::shared_ptr<mindspore::dataset::Tensor> de_tensor;
    mindspore::dataset::Tensor::CreateFromFile(v.toString(), &de_tensor);
    auto image = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));

    (void)Execute(Decode())(image, &image);
    EXPECT_TRUE(image != nullptr);
    (void)Execute(Normalize({121.0, 115.0, 100.0}, {70.0, 68.0, 71.0}))(image, &image);
    EXPECT_TRUE(image != nullptr);
    (void)Execute(Resize({224, 224}))(image, &image);
    EXPECT_TRUE(image != nullptr);
    EXPECT_EQ(image.Shape()[0], 224);
    EXPECT_EQ(image.Shape()[1], 224);
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  MS_LOG(INFO) << "duration: " << elapsed_time_ms << " ms\n";
}
