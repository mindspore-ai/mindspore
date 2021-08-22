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
#include <fstream>
#include <iostream>
#include <memory>
#include <string>


#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

#include "utils/ms_utils.h"
#include "common/common.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/datasetops/source/libri_speech_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/pk_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"

namespace common = mindspore::common;
using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

std::shared_ptr<RepeatOp> Repeat(int repeat_cnt);

std::shared_ptr<ExecutionTree> Build(std::vector<std::shared_ptr<DatasetOp>> ops);

class MindDataTestLibriSpeechSampler : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestLibriSpeechSampler, TestSequentialLibriSpeechWithRepeat) {
  std::string folder_path = "/home/user06/zjm/data/libri_speech/LibriSpeech/";
  int64_t num_samples = 10;
  int64_t start_index = 0;
  std::shared_ptr<Dataset> ds =
    LibriSpeech(folder_path, "dev-clean", std::make_shared<SequentialSampler>(start_index, num_samples));
  EXPECT_NE(ds, nullptr);
  ds = ds->Repeat(2);
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::string_view utterance;
  uint32_t rate;
  uint32_t utterance_id;
  uint32_t speaker_id;
  uint32_t chapter_id;

  
  uint64_t i = 0;
  while (row.size() != 0) {

    auto waveform = row["waveform"];
    auto sample_rate = row["sample_rate"];
    auto utterance_ = row["utterance"];
    auto utterance_id_ = row["utterance_id"];
    auto speaker_id_ = row["speaker_id"];
    auto chapter_id_ = row["chapter_id"];

    MS_LOG(ERROR) << "Tensor image shape: " << waveform.Shape();

    std::shared_ptr<Tensor> t_rate;
    ASSERT_OK(Tensor::CreateFromMSTensor(sample_rate, &t_rate));
    ASSERT_OK(t_rate->GetItemAt<uint32_t>(&rate, {}));
    MS_LOG(ERROR) << "Tensor rate: " << rate;

    std::shared_ptr<Tensor> t_utterance;
    ASSERT_OK(Tensor::CreateFromMSTensor(utterance_, &t_utterance));
    ASSERT_OK(t_utterance->GetItemAt(&utterance, {}));
    MS_LOG(ERROR) << "Tensor utterance value: " << utterance;

    std::shared_ptr<Tensor> t_speaker_id;
    ASSERT_OK(Tensor::CreateFromMSTensor(speaker_id_, &t_speaker_id));
    ASSERT_OK(t_speaker_id->GetItemAt<uint32_t>(&speaker_id, {}));
    MS_LOG(ERROR) << "Tensor speaker_id value: " << speaker_id;

    std::shared_ptr<Tensor> t_chapter_id;
    ASSERT_OK(Tensor::CreateFromMSTensor(chapter_id_, &t_chapter_id));
    ASSERT_OK(t_chapter_id->GetItemAt<uint32_t>(&chapter_id, {}));
    MS_LOG(ERROR) << "Tensor chapter_id value: " << chapter_id;


    std::shared_ptr<Tensor> t_utterance_id;
    ASSERT_OK(Tensor::CreateFromMSTensor(utterance_id_, &t_utterance_id));
    ASSERT_OK(t_utterance_id->GetItemAt<uint32_t>(&utterance_id, {}));
    MS_LOG(ERROR) << "Tensor utterance_id value: " << utterance_id;



    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 20);
  iter->Stop();
}

// TEST_F(MindDataTestMnistSampler, TestSequentialImageFolderWithRepeatBatch) {
//   std::string folder_path = datasets_root_path_ + "/testMnistData/";
//   int64_t num_samples = 10;
//   int64_t start_index = 0;
//   std::shared_ptr<Dataset> ds =
//     Mnist(folder_path, "all", std::make_shared<SequentialSampler>(start_index, num_samples));
//   EXPECT_NE(ds, nullptr);
//   ds = ds->Repeat(2);
//   EXPECT_NE(ds, nullptr);
//   ds = ds->Batch(5);
//   EXPECT_NE(ds, nullptr);
//   std::shared_ptr<Iterator> iter = ds->CreateIterator();
//   EXPECT_NE(iter, nullptr);
//   std::vector<std::vector<uint32_t>> expected = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
//   std::unordered_map<std::string, mindspore::MSTensor> row;
//   ASSERT_OK(iter->GetNextRow(&row));
//   uint64_t i = 0;
//   while (row.size() != 0) {
//     auto image = row["image"];
//     auto label = row["label"];
//     MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
//     TEST_MS_LOG_MSTENSOR(INFO, "Tensor label: ", label);
//     std::shared_ptr<Tensor> de_expected_label;
//     ASSERT_OK(Tensor::CreateFromVector(expected[i % 4], &de_expected_label));
//     mindspore::MSTensor expected_label =
//       mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_label));
//     EXPECT_MSTENSOR_EQ(label, expected_label);
//     ASSERT_OK(iter->GetNextRow(&row));
//     i++;
//   }
//   EXPECT_EQ(i, 4);
//   iter->Stop();
// }


