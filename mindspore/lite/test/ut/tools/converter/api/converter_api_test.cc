/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "gtest/gtest.h"
#include "include/converter.h"

TEST(TestConverterAPI, ConvertCaffe) {
  std::string caffe_model = "./detect_mbv1_640_480_nopostprocess_simplified.prototxt";
  std::string caffe_weight = "./detect_mbv1_640_480_nopostprocess_simplified.caffemodel";
  std::string output_model = "./detect_mbv1_640_480_nopostprocess_simplified.ms";

  mindspore::Converter converter(mindspore::converter::FmkType::kFmkTypeCaffe, caffe_model, output_model, caffe_weight);
  ASSERT_TRUE(converter.Convert().IsOk());
}

TEST(TestConverterAPI, ConvertCaffeWithNotExistWeight) {
  std::string caffe_model = "./detect_mbv1_640_480_nopostprocess_simplified.prototxt";
  std::string caffe_weight = "./not-exist.caffemodel";
  std::string output_model = "./detect_mbv1_640_480_nopostprocess_simplified.ms";

  mindspore::Converter converter(mindspore::converter::FmkType::kFmkTypeCaffe, caffe_model, output_model, caffe_weight);
  ASSERT_FALSE(converter.Convert().IsOk());
}
