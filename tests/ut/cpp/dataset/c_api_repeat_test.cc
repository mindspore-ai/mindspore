/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "common/common.h"
#include "minddata/dataset/include/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestRepeatSetNumWorkers) {
  MS_LOG(INFO) << "Doing MindDataTestRepeat-TestRepeatSetNumWorkers.";

  std::string file_path = datasets_root_path_ + "/testTFTestAllTypes/test.data";
  std::shared_ptr<Dataset> ds = TFRecord({file_path});
  ds = ds->SetNumWorkers(8);
  ds = ds->Repeat(32);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect a valid iterator
  ASSERT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    iter->GetNextRow(&row);
  }

  // Verify correct number of rows fetched
  EXPECT_EQ(i, 12 * 32);

  // Manually terminate the pipeline
  iter->Stop();
}
