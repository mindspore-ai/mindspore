/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <memory>
#include <string>

#include "common/common.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/datasetops/source/voc_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);

class MindDataTestVOCOp : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestVOCOp, TestVOCDetection) {
  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/testVOC2012";
  std::shared_ptr<Dataset> ds =
    VOC(dataset_path, "Detection", "train", {}, false, std::make_shared<SequentialSampler>(0, 0));
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  int row_count = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
    row_count++;
  }
  ASSERT_EQ(row_count, 9);
  iter->Stop();
}

TEST_F(MindDataTestVOCOp, TestVOCSegmentation) {
  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/testVOC2012";
  std::shared_ptr<Dataset> ds =
    VOC(dataset_path, "Segmentation", "train", {}, false, std::make_shared<SequentialSampler>(0, 0));
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  int row_count = 0;
  while (!row.empty()) {
    auto image = row["image"];
    auto target = row["target"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor target shape: " << target.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
    row_count++;
  }
  ASSERT_EQ(row_count, 10);
  iter->Stop();
}

TEST_F(MindDataTestVOCOp, TestVOCClassIndex) {
  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/testVOC2012";
  std::map<std::string, int32_t> class_index;
  class_index["car"] = 0;
  class_index["cat"] = 1;
  class_index["train"] = 5;
  std::shared_ptr<Dataset> ds =
    VOC(dataset_path, "Detection", "train", class_index, false, std::make_shared<SequentialSampler>(0, 0));
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  int row_count = 0;
  while (!row.empty()) {
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
    row_count++;
  }
  ASSERT_EQ(row_count, 6);
  iter->Stop();
}
