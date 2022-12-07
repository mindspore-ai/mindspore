/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/vision.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on Album
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableAlbum) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableAlbum.";

  std::string folder_path = datasets_root_path_ + "/testAlbum/images";
  std::string schema_file = datasets_root_path_ + "/testAlbum/datasetSchema.json";
  std::vector<std::string> column_names = {"label", "image"};
  std::shared_ptr<Dataset> ds = Album(folder_path, schema_file, column_names);
  EXPECT_NE(ds, nullptr);

  std::vector<mindspore::MSTensor> row;
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  ASSERT_OK(iter->GetNextRow(&row));
  auto count = 0;
  while (row.size() > 0) {
    ++count;
    auto label = row[0];
    auto image = row[1];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor level shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  EXPECT_EQ(count, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on CelebA
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableCelebA) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableCelebA.";

  // Create a CelebA Dataset
  std::string folder_path = datasets_root_path_ + "/testCelebAData/";
  std::shared_ptr<Dataset> ds = CelebA(folder_path);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  // order: image, level
  // Check if CelebA() read correct images/attr
  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row[0];
    auto level = row[1];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor level shape: " << level.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on Cifar10
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableCifar10) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableCifar10.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 2);

  // order: image, label
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    auto label = row[1];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on Cifar100
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableCifar100) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableCifar100.";

  // Create a Cifar100 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar100Data/";
  std::shared_ptr<Dataset> ds = Cifar100(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 3);

  // order: image, coarse_label, fine_label
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    auto coarse_label = row[1];
    MS_LOG(INFO) << "Tensor coarse_label shape: " << coarse_label.Shape();
    auto fine_label = row[2];
    MS_LOG(INFO) << "Tensor fine_label shape: " << fine_label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on Cityscapes
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableCityscapes) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableCityscapes.";

  std::string dataset_path = datasets_root_path_ + "/testCityscapesData/cityscapes";
  std::string usage = "train";        // quality_mode=fine 'train', 'test', 'val'  else 'train', 'train_extra', 'val'
  std::string quality_mode = "fine";  // fine coarse
  std::string task = "color";         // instance semantic polygon color

  // Create a Cityscapes Dataset
  std::shared_ptr<Dataset> ds = Cityscapes(dataset_path, usage, quality_mode, task);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[1];
    auto task = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on CMUArctic
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableCMUArctic) {
  MS_LOG(INFO) << "Doing CMUArcticDataTestPipeline-TestGetNextPullBasedMappableCMUArctic.";

  std::string folder_path = datasets_root_path_ + "/testCMUArcticData";
  // Create a CMUArctic Dataset.
  std::shared_ptr<Dataset> ds = CMUArctic(folder_path);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::string_view transcript_idx, utterance_id_idx;
  uint32_t rate = 0;
  uint64_t i = 0;

  while (row.size() != 0) {
    // order: waveform, sample_rate, transcript, utterance_id
    auto waveform = row[0];
    auto sample_rate = row[1];
    auto transcript = row[2];
    auto utterance_id = row[3];

    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();

    std::shared_ptr<Tensor> trate;
    ASSERT_OK(Tensor::CreateFromMSTensor(sample_rate, &trate));
    ASSERT_OK(trate->GetItemAt<uint32_t>(&rate, {}));
    MS_LOG(INFO) << "Audio sample rate: " << rate;

    std::shared_ptr<Tensor> de_transcript;
    ASSERT_OK(Tensor::CreateFromMSTensor(transcript, &de_transcript));
    ASSERT_OK(de_transcript->GetItemAt(&transcript_idx, {}));
    std::string s_transcript(transcript_idx);
    MS_LOG(INFO) << "Tensor transcript value: " << transcript_idx;

    std::shared_ptr<Tensor> de_utterance_id;
    ASSERT_OK(Tensor::CreateFromMSTensor(utterance_id, &de_utterance_id));
    ASSERT_OK(de_utterance_id->GetItemAt(&utterance_id_idx, {}));
    std::string s_utterance_id(utterance_id_idx);
    MS_LOG(INFO) << "Tensor utterance_id value: " << utterance_id_idx;
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on Coco
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableCoco) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableCoco.";
  // Create a Coco Dataset.
  std::string folder_path = datasets_root_path_ + "/testCOCO/train";
  std::string annotation_file = datasets_root_path_ + "/testCOCO/annotations/train.json";

  std::shared_ptr<Dataset> ds = Coco(folder_path, annotation_file);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: image, bbox, category_id, iscrowd
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 4);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    auto bbox = row[1];
    auto category_id = row[2];
    auto iscrowd = row[3];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor bbox shape: " << bbox.Shape();
    MS_LOG(INFO) << "Tensor category_id shape: " << category_id.Shape();
    MS_LOG(INFO) << "Tensor iscrowd shape: " << iscrowd.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on DIV2K
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableDIV2K) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableDIV2K.";

  std::string dataset_path = datasets_root_path_ + "/testDIV2KData/div2k";
  std::string usage = "train";        // train valid, all
  std::string downgrade = "bicubic";  // bicubic, unknown, mild, difficult, wild
  int32_t scale = 2;                  // 2, 3, 4, 8

  // Create a DIV2K Dataset
  std::shared_ptr<Dataset> ds = DIV2K(dataset_path, usage, downgrade, scale);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: hr_image, lr_image
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 2);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto hr_image = row[0];
    auto lr_image = row[1];
    MS_LOG(INFO) << "Tensor hr_image shape: " << hr_image.Shape();
    MS_LOG(INFO) << "Tensor lr_image shape: " << lr_image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on FakeImage
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableFakeImage) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableFakeImage.";

  // Create a FakeImage Dataset
  std::shared_ptr<Dataset> ds = FakeImage(50, {28, 28, 3}, 3, 0, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: image, label
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 2);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    auto label = row[1];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on Flickr
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableFlickr) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableFlickr.";

  std::string dataset_path = datasets_root_path_ + "/testFlickrData/flickr30k/flickr30k-images";
  std::string file_path = datasets_root_path_ + "/testFlickrData/flickr30k/test1.token";

  // Create a Flickr30k Dataset
  std::shared_ptr<Dataset> ds = Flickr(dataset_path, file_path);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: image, annotation
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    auto annotation = row[1];
    MS_LOG(INFO) << "Tensor annotation shape: " << annotation.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on GTZAN
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableGTZAN) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableGTZAN.";

  std::string file_path = datasets_root_path_ + "/testGTZANData";
  // Create a GTZAN Dataset
  std::shared_ptr<Dataset> ds = GTZAN(file_path);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  // order: waveform, sample_rate, label
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::string_view label_idx;
  uint32_t rate = 0;
  uint64_t i = 0;

  while (row.size() != 0) {
    i++;
    auto waveform = row[0];
    auto sample_rate = row[1];
    auto label = row[2];
    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();

    std::shared_ptr<Tensor> trate;
    ASSERT_OK(Tensor::CreateFromMSTensor(sample_rate, &trate));
    ASSERT_OK(trate->GetItemAt<uint32_t>(&rate, {}));
    EXPECT_EQ(rate, 22050);
    MS_LOG(INFO) << "Tensor label rate: " << rate;

    std::shared_ptr<Tensor> de_label;
    ASSERT_OK(Tensor::CreateFromMSTensor(label, &de_label));
    ASSERT_OK(de_label->GetItemAt(&label_idx, {}));
    std::string s_label(label_idx);
    std::string expected("blues");
    EXPECT_STREQ(s_label.c_str(), expected.c_str());
    MS_LOG(INFO) << "Tensor label value: " << label_idx;
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on ImageFolder
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableImageFolder) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableImageFolder.";
  // Test get dataset size in distributed scenario when num_per_shard is more than num_samples

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<DistributedSampler>(4, 0, false, 10));
  EXPECT_NE(ds, nullptr);

  // num_per_shard is equal to 44/4 = 11 which is more than num_samples = 10, so the output is 10
  EXPECT_EQ(ds->GetDatasetSize(), 10);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 2);
  MS_LOG(INFO) << "row.size() " << row.size();

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  // The value of i should be equal to the result of get dataset size
  EXPECT_EQ(i, 10);
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on IMDB
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableIMDB) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableIMDB.";

  std::string dataset_path = datasets_root_path_ + "/testIMDBDataset";
  std::string usage = "all";  // 'train', 'test', 'all'

  // Create a IMDB Dataset
  std::shared_ptr<Dataset> ds = IMDB(dataset_path, usage);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: text, label
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto text = row[0];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape();
    auto label = row[1];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on KITTI
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableKITTI) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableKITTI.";

  // Create a KITTI Dataset.
  std::string folder_path = datasets_root_path_ + "/testKITTI";
  std::shared_ptr<Dataset> ds = KITTI(folder_path, "train", false, std::make_shared<SequentialSampler>(0, 2));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  // order: image
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  // Check if KITTI() read correct images.
  std::string expect_file[] = {"000000", "000001", "000002"};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row[0];
    auto label = row[1];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();

    mindspore::MSTensor expect_image =
      ReadFileToTensor(folder_path + "/data_object_image_2/training/image_2/" + expect_file[i] + ".png");
    EXPECT_MSTENSOR_EQ(image, expect_image);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on LFW
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableLFW) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableLFW.";

  // Create a LFW Dataset.
  std::string folder_path = datasets_root_path_ + "/testLFW";
  std::shared_ptr<Dataset> ds = LFW(folder_path, "people", "all", "original", false);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  // order: image, label
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_EQ(row.size(), 2);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    auto label = row[1];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on LibriTTS
/// Expectation: Output is the same as the normal iterator
// Note: test files in /testLibriTTSData incompleted
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableLibriTTS) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableLibriTTS.";

  std::string folder_path = datasets_root_path_ + "/testLibriTTSData";
  std::shared_ptr<Dataset> ds = LibriTTS(folder_path);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  // order: waveform, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 7);
  uint64_t i = 0;

  while (row.size() != 0) {
    auto waveform = row[0];
    auto sample_rate = row[1];
    auto original_text = row[2];
    auto normalized_text = row[3];
    auto speaker_id = row[4];
    auto chapter_id = row[5];
    auto utterance_id = row[6];
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }
  EXPECT_EQ(i, 3);
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on LJSpeech
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableLJSpeech) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableLJSpeech.";
  std::string folder_path = datasets_root_path_ + "/testLJSpeechData/";
  std::shared_ptr<Dataset> ds = LJSpeech(folder_path, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  // order: waveform, sample_rate, transcription, normalized_transcript
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 4);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row[0];
    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();
    auto sample_rate = row[1];
    MS_LOG(INFO) << "Tensor sample_rate shape: " << sample_rate.Shape();
    auto transcription = row[2];
    MS_LOG(INFO) << "Tensor transcription shape: " << transcription.Shape();
    auto normalized_transcript = row[3];
    MS_LOG(INFO) << "Tensor normalized_transcript shape: " << normalized_transcript.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on Manifest
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableManifest) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableManifest.";

  std::string file_path = datasets_root_path_ + "/testManifestData/cpp.json";
  // Create a Manifest Dataset
  std::shared_ptr<Dataset> ds = Manifest(file_path);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: image, label
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 2);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    auto label = row[1];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on MindRecord
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableMinddata) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableMinddata with string file pattern.";
  // Create a MindData Dataset
  // Pass one mindrecord shard file to parse dataset info, and search for other mindrecord files with same dataset info,
  // thus all records in imagenet.mindrecord0 ~ imagenet.mindrecord3 will be read
  std::string file_path = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::vector<std::string> column_names = {"file_name", "label", "data"};
  std::shared_ptr<Dataset> ds = MindData(file_path, column_names);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 3);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    TEST_MS_LOG_MSTENSOR(INFO, "Tensor image file name: ", image);
    auto label = row[1];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    auto data = row[2];
    MS_LOG(INFO) << "Tensor data shape: " << data.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  // Each *.mindrecord file has 5 rows, so there are 20 rows in total(imagenet.mindrecord0 ~ imagenet.mindrecord3)
  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on Mnist
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableMnist) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableMnist.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // iterate over the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    auto label = row[1];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on PhotoTour
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappablePhotoTour) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappablePhotoTour.";

  // Create a PhotoTour Test Dataset
  std::string folder_path = datasets_root_path_ + "/testPhotoTourData";
  std::shared_ptr<Dataset> ds = PhotoTour(folder_path, "liberty", "test", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row

  std::vector<mindspore::MSTensor> row;
  // order: image1, image2, label
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_EQ(row.size(), 3);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image1 = row[0];
    MS_LOG(INFO) << "Tensor image1 shape: " << image1.Shape();
    auto image2 = row[1];
    MS_LOG(INFO) << "Tensor image2 shape: " << image2.Shape();
    auto label = row[2];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on Places365 with train standard
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappablePlaces365TrainStandard) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappablePlaces365TrainStandard.";

  // Create a Places365 Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testPlaces365Data";
  std::shared_ptr<Dataset> ds =
    Places365(folder_path, "train-standard", true, true, std::make_shared<RandomSampler>(false, 4));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: image, label
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 2);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    auto label = row[1];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on RandomData
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableRandomData) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableRandomData.";

  // Create a RandomDataset
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("image", mindspore::DataType::kNumberTypeUInt8, {2}));
  ASSERT_OK(schema->add_column("label", mindspore::DataType::kNumberTypeUInt8, {1}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: image, label
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 2);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    auto label = row[1];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  EXPECT_EQ(i, 50);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on SBU
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableSBU) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableSBU.";

  // Create a SBU Dataset
  std::string folder_path = datasets_root_path_ + "/testSBUDataset/";
  std::shared_ptr<Dataset> ds = SBU(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: image, caption
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 2);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    auto caption = row[1];
    MS_LOG(INFO) << "Tensor caption shape: " << caption.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on Semeion
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableSemeion) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableSemeion.";

  // Create a Semeion Dataset.
  std::string folder_path = datasets_root_path_ + "/testSemeionData";
  std::shared_ptr<Dataset> ds = Semeion(folder_path, std::make_shared<RandomSampler>(false, 5), nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: image, label
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 2);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    auto label = row[1];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on SpeechCommands
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableSpeechCommands) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableSpeechCommands.";
  std::string folder_path = datasets_root_path_ + "/testSpeechCommandsData/";
  std::shared_ptr<Dataset> ds = SpeechCommands(folder_path, "all", std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: waveform, sample_rate, label, speaker_id, utterance_number
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 5);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row[0];
    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();
    auto sample_rate = row[1];
    MS_LOG(INFO) << "Tensor sample_rate shape: " << sample_rate.Shape();
    auto label = row[2];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    auto speaker_id = row[3];
    MS_LOG(INFO) << "Tensor speaker_id shape: " << speaker_id.Shape();
    auto utterance_number = row[4];
    MS_LOG(INFO) << "Tensor utterance_number shape: " << utterance_number.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on STL10Dataset with train+unlabeled
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableSTL10) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableSTL10.";

  // Create a STL10 Dataset
  std::string folder_path = datasets_root_path_ + "/testSTL10Data/";
  std::shared_ptr<Dataset> ds = STL10(folder_path, "train+unlabeled", std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: image, label
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 2);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row[0];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    auto label = row[1];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on Tedlium
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableTedlium) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableTedlium.";
  // Create a Tedlium Dataset.
  std::string folder_path12 = datasets_root_path_ + "/testTedliumData/TEDLIUM_release1";
  std::shared_ptr<Dataset> ds =
    Tedlium(folder_path12, "release1", "all", ".sph", std::make_shared<RandomSampler>(false, 4), nullptr);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::vector<std::string> columns = {"waveform", "sample_rate", "transcript", "talk_id", "speaker_id", "identifier"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: waveform, sample_rate, transcript, talk_id, speaker_id, identifier
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 6);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row[0];
    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();
    auto sample_rate = row[1];
    MS_LOG(INFO) << "Tensor sample_rate shape: " << sample_rate.Shape();
    auto transcript = row[2];
    MS_LOG(INFO) << "Tensor transcript shape: " << transcript.Shape();
    auto talk_id = row[3];
    MS_LOG(INFO) << "Tensor talk_id shape: " << talk_id.Shape();
    auto speaker_id = row[4];
    MS_LOG(INFO) << "Tensor speaker_id shape: " << speaker_id.Shape();
    auto identifier = row[5];
    MS_LOG(INFO) << "Tensor identifier shape: " << identifier.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on VOC
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableVOC) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableVOC.";

  // Create a VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::map<std::string, int32_t> class_index;
  class_index["car"] = 0;
  class_index["cat"] = 1;
  class_index["train"] = 9;

  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", class_index, false, std::make_shared<SequentialSampler>(0, 6));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: image, bbox, label, difficult, truncate
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 5);

  // Check if VOC() read correct labels
  // When we provide class_index, label of ["car","cat","train"] become [0,1,9]
  std::shared_ptr<Tensor> de_expect_label;
  ASSERT_OK(Tensor::CreateFromMemory(TensorShape({1, 1}), DataType(DataType::DE_UINT32), nullptr, &de_expect_label));
  uint32_t expect[] = {9, 9, 9, 1, 1, 0};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row[0];
    auto bbox = row[1];
    auto label = row[2];
    auto difficult = row[3];
    auto truncate = row[4];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor bbox shape: " << bbox.Shape();
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    MS_LOG(INFO) << "Tensor difficult shape: " << difficult.Shape();
    MS_LOG(INFO) << "Tensor truncate shape: " << truncate.Shape();

    ASSERT_OK(de_expect_label->SetItemAt({0, 0}, expect[i]));
    mindspore::MSTensor expect_label =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expect_label));
    EXPECT_MSTENSOR_EQ(label, expect_label);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on WIDERFace
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableWIDERFace) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableWIDERFace.";
  // Create a WIDERFace Dataset.
  std::string folder_path = datasets_root_path_ + "/testWIDERFace/";
  std::shared_ptr<Dataset> ds = WIDERFace(folder_path);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: image, bbox, blur, expression, illumination, occlusion, pose, invalid
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 8);

  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row[0];
    auto bbox = row[1];
    auto blur = row[2];
    auto expression = row[3];
    auto illumination = row[4];
    auto occlusion = row[5];
    auto pose = row[6];
    auto invalid = row[7];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor bbox shape: " << bbox.Shape();
    MS_LOG(INFO) << "Tensor blur shape: " << blur.Shape();
    MS_LOG(INFO) << "Tensor expression shape: " << expression.Shape();
    MS_LOG(INFO) << "Tensor illumination shape: " << illumination.Shape();
    MS_LOG(INFO) << "Tensor occlusion shape: " << occlusion.Shape();
    MS_LOG(INFO) << "Tensor pose shape: " << pose.Shape();
    MS_LOG(INFO) << "Tensor invalid shape: " << invalid.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);
  iter->Stop();
}

/// Feature: PullBasedIterator GetNextRowPullMode
/// Description: Test PullBasedIterator on YesNo
/// Expectation: Output is the same as the normal iterator
TEST_F(MindDataTestPipeline, TestGetNextPullBasedMappableYesNo) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetNextPullBasedMappableYesNo.";
  // Create a YesNoDataset
  std::string folder_path = datasets_root_path_ + "/testYesNoData/";
  std::shared_ptr<Dataset> ds = YesNo(folder_path, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<PullIterator> iter = ds->CreatePullBasedIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  // order: waveform, sample_rate, label
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 3);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row[0];
    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();
    auto sample_rate = row[1];
    MS_LOG(INFO) << "Tensor sample_rate shape: " << sample_rate.Shape();
    auto label = row[2];
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}
