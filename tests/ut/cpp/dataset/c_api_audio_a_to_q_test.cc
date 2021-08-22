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
#include "include/api/types.h"
#include "utils/log_adapter.h"

#include "minddata/dataset/include/dataset/audio.h"
#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;
using namespace std;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, Level0_TestBandBiquad001) {
  MS_LOG(INFO) << "Basic Function Test";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto BandBiquadOp = audio::BandBiquad(44100, 200.0);

  ds = ds->Map({BandBiquadOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by bandbiquad
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {2, 200};

  int i = 0;
  while (row.size() != 0) {
    auto col = row["inputData"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);

  iter->Stop();
}

TEST_F(MindDataTestPipeline, Level0_TestBandBiquad002) {
  MS_LOG(INFO) << "Wrong Arg.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  auto band_biquad_op_01 = audio::BandBiquad(0, 200);
  ds01 = ds->Map({band_biquad_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  // Check Q_
  MS_LOG(INFO) << "Q_ is zero.";
  auto band_biquad_op_02 = audio::BandBiquad(44100, 200, 0);
  ds02 = ds->Map({band_biquad_op_02});
  EXPECT_NE(ds02, nullptr);

  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);
}