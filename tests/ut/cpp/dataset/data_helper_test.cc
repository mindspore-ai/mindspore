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
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include "common/common.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/pk_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"
#include "minddata/dataset/include/data_helper.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::ERROR;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestDataHelper : public UT::DatasetOpTesting  {
 protected: 
};

TEST_F(MindDataTestDataHelper, MindDataTestHelper) {
  std::string file_path = datasets_root_path_ + "/testAlbum/images/1.json";
  DataHelper dh; 
  std::vector<std::string> new_label = {"3", "4"};
  Status rc = dh.UpdateArray(file_path, "label", new_label); 
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during label update: "  << ".";
    EXPECT_TRUE(false);
  }
}

TEST_F(MindDataTestDataHelper, MindDataTestAlbumGen) {
  std::string file_path = datasets_root_path_ + "/testAlbum/original";
  std::string out_path = datasets_root_path_ + "/testAlbum/testout";
  DataHelper dh;
  Status rc = dh.CreateAlbum(file_path, out_path);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during album generation: "  << ".";
    EXPECT_TRUE(false);
  }
}

TEST_F(MindDataTestDataHelper, MindDataTestTemplateUpdateArrayInt) {
  std::string file_path = datasets_root_path_ + "/testAlbum/testout/2.json";
  DataHelper dh;
  std::vector<int> new_label = {3, 4};
  Status rc = dh.UpdateArray(file_path, "label", new_label);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during json int array update: "  << ".";
    EXPECT_TRUE(false);
  }
}

TEST_F(MindDataTestDataHelper, MindDataTestTemplateUpdateArrayString) {
  std::string file_path = datasets_root_path_ + "/testAlbum/testout/3.json";
  DataHelper dh;
  std::vector<std::string> new_label = {"3", "4"};
  Status rc = dh.UpdateArray(file_path, "label", new_label);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during json string array update: "  << ".";
    EXPECT_TRUE(false);
  }
}

TEST_F(MindDataTestDataHelper, MindDataTestTemplateUpdateValueInt) {
  std::string file_path = datasets_root_path_ + "/testAlbum/testout/4.json";
  DataHelper dh;
  int new_label = 3;
  Status rc = dh.UpdateValue(file_path, "label", new_label);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during json int update: "  << ".";
    EXPECT_TRUE(false);
  }
}

TEST_F(MindDataTestDataHelper, MindDataTestTemplateUpdateString) {
  std::string file_path = datasets_root_path_ + "/testAlbum/testout/5.json";
  DataHelper dh;
  std::string new_label = "new label";
  Status rc = dh.UpdateValue(file_path, "label", new_label);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during json string update: "  << ".";
    EXPECT_TRUE(false);
  }
}

TEST_F(MindDataTestDataHelper, MindDataTestDeleteKey) {
  std::string file_path = datasets_root_path_ + "/testAlbum/testout/5.json";
  DataHelper dh;
  Status rc = dh.RemoveKey(file_path, "label");
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during json key remove: "  << ".";
    EXPECT_TRUE(false);
  }
}

TEST_F(MindDataTestDataHelper, MindDataTestBinWrite) {
  std::string file_path = datasets_root_path_ + "/testAlbum/1.bin";
  DataHelper dh;
  std::vector<float> bin_content = {3, 4};
  Status rc = dh.WriteBinFile(file_path, bin_content);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during bin file write: "  << ".";
    EXPECT_TRUE(false);
  }
}

TEST_F(MindDataTestDataHelper, MindDataTestBinWritePointer) {
  std::string file_path = datasets_root_path_ + "/testAlbum/2.bin";
  DataHelper dh;
  std::vector<float> bin_content = {3, 4};

  Status rc = dh.WriteBinFile(file_path, &bin_content[0], bin_content.size());
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during binfile write: "  << ".";
    EXPECT_TRUE(false);
  }
}

TEST_F(MindDataTestDataHelper, MindDataTestTensorWriteFloat) {
  // create tensor
  std::vector<float> y = {2.5, 3.0, 3.5, 4.0};
  std::shared_ptr<Tensor> t;
  Tensor::CreateFromVector(y, &t);
  // create buffer using system mempool
  DataHelper dh;
  void *data = malloc(t->SizeInBytes());
  if (data == nullptr) {
    MS_LOG(ERROR) << "malloc failed";
    ASSERT_TRUE(false);
  }
  auto bytes_copied = dh.DumpData(t->GetBuffer(), t->SizeInBytes(), data, t->SizeInBytes());
  if (bytes_copied != t->SizeInBytes()) {
    EXPECT_TRUE(false);
  }
  float *array = static_cast<float *>(data);
  if (array[0] != 2.5) { EXPECT_TRUE(false); }
  if (array[1] != 3.0) { EXPECT_TRUE(false); }
  if (array[2] != 3.5) { EXPECT_TRUE(false); }
  if (array[3] != 4.0) { EXPECT_TRUE(false); }
  std::free(data); 
}

TEST_F(MindDataTestDataHelper, MindDataTestTensorWriteUInt) {
  // create tensor
  std::vector<uint8_t> y = {1, 2, 3, 4};
  std::shared_ptr<Tensor> t;
  Tensor::CreateFromVector(y, &t);
  uint8_t o;
  t->GetItemAt<uint8_t>(&o, {0, 0});
  MS_LOG(INFO) << "before op :" << std::to_string(o) << ".";

  // create buffer using system mempool
  DataHelper dh;
  void *data = malloc(t->SizeInBytes());
  if (data == nullptr) {
    MS_LOG(ERROR) << "malloc failed";
    ASSERT_TRUE(false);
  }
  auto bytes_copied = dh.DumpData(t->GetBuffer(), t->SizeInBytes(), data, t->SizeInBytes());
  if (bytes_copied != t->SizeInBytes()) {
    EXPECT_TRUE(false);
  }
  t->GetItemAt<uint8_t>(&o, {});
  MS_LOG(INFO) << "after op :" << std::to_string(o) << ".";

  uint8_t *array = static_cast<uint8_t *>(data);
  if (array[0] != 1) { EXPECT_TRUE(false); }
  if (array[1] != 2) { EXPECT_TRUE(false); }
  if (array[2] != 3) { EXPECT_TRUE(false); }
  if (array[3] != 4) { EXPECT_TRUE(false); }
  std::free(data);
}


