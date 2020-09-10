/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <gtest/gtest.h>
#include <string>
#include "tools/converter/converter.h"
#include "common/common_test.h"

namespace mindspore {
namespace lite {
class ConverterTest : public mindspore::CommonTest {
 public:
  ConverterTest() {}
};

TEST_F(ConverterTest, TestLenet) {
  const char *argv[] = {"./converter", "--fmk=MS", "--modelFile=./common/lenet_bin.pb",
                        "--outputFile=./models/lenet_bin"};
  auto status = RunConverter(4, argv);
  ASSERT_EQ(status, RET_OK);
}

TEST_F(ConverterTest, TestVideo) {
  const char *argv[] = {"./converter", "--fmk=TFLITE", "--modelFile=./hiai/hiai_label_and_video.tflite",
                        "--outputFile=./models/hiai_label_and_video"};
  auto status = RunConverter(4, argv);
  ASSERT_EQ(status, RET_OK);
}

TEST_F(ConverterTest, TestOCR_02) {
  const char *argv[] = {"./converter", "--fmk=TFLITE", "--modelFile=./hiai/hiai_cv_focusShootOCRMOdel_02.tflite",
                        "--outputFile=./models/hiai_cv_focusShootOCRMOdel_02"};
  auto status = RunConverter(4, argv);
  ASSERT_EQ(status, RET_OK);
}

TEST_F(ConverterTest, TestHebing) {
  const char *argv[] = {"./converter", "--fmk=CAFFE", "--modelFile=./hiai/model_hebing_3branch.prototxt",
                        "--weightFile=./models/model_hebing_3branch.caffemodel",
                        "--outputFile=./models/model_hebing_3branch"};
  auto status = RunConverter(5, argv);
  ASSERT_EQ(status, RET_OK);
}
}  // namespace lite
}  // namespace mindspore
