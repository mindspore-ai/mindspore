/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/core/tensor.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestMindDataSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMindDataSuccess1 with string file pattern.";

  // Create a MindData Dataset
  // Pass one mindrecord shard file to parse dataset info, and search for other mindrecord files with same dataset info,
  // thus all records in imagenet.mindrecord0 ~ imagenet.mindrecord3 will be read
  std::string file_path = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::shared_ptr<Dataset> ds = MindData(file_path);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["file_name"];
    TEST_MS_LOG_MSTENSOR(INFO, "Tensor image file name: ", image);

    iter->GetNextRow(&row);
  }

  // Each *.mindrecord file has 5 rows, so there are 20 rows in total(imagenet.mindrecord0 ~ imagenet.mindrecord3)
  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestMindDataGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMindDataGetters with string file pattern.";

  // Create a MindData Dataset
  // Pass one mindrecord shard file to parse dataset info, and search for other mindrecord files with same dataset info,
  // thus all records in imagenet.mindrecord0 ~ imagenet.mindrecord3 will be read
  std::string file_path = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::shared_ptr<Dataset> ds = MindData(file_path);
  EXPECT_NE(ds, nullptr);

  std::vector<std::string> column_names = {"data", "file_name", "label"};

  EXPECT_EQ(ds->GetDatasetSize(), 20);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

TEST_F(MindDataTestPipeline, TestMindDataSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMindDataSuccess2 with a vector of single mindrecord file.";

  // Create a MindData Dataset
  // Pass a list of mindrecord file name, files in list will be read directly but not search for related files
  std::string file_path1 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::shared_ptr<Dataset> ds = MindData(std::vector<std::string>{file_path1});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["file_name"];
    TEST_MS_LOG_MSTENSOR(INFO, "Tensor image file name: ", image);

    iter->GetNextRow(&row);
  }

  // Only records in imagenet.mindrecord0 are read
  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestMindDataSuccess3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMindDataSuccess3 with a vector of multiple mindrecord files.";

  // Create a MindData Dataset
  // Pass a list of mindrecord file name, files in list will be read directly but not search for related files
  std::string file_path1 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::string file_path2 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord1";
  std::vector<std::string> file_list = {file_path1, file_path2};
  std::shared_ptr<Dataset> ds = MindData(file_list);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["file_name"];
    TEST_MS_LOG_MSTENSOR(INFO, "Tensor image file name: ", image);

    iter->GetNextRow(&row);
  }

  // Only records in imagenet.mindrecord0 and imagenet.mindrecord1 are read
  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestMindDataSuccess4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMindDataSuccess4 with specified column.";

  // Create a MindData Dataset
  // Pass one mindrecord shard file to parse dataset info, and search for other mindrecord files with same dataset info,
  // thus all records in imagenet.mindrecord0 ~ imagenet.mindrecord3 will be read
  std::string file_path1 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord1";
  std::shared_ptr<Dataset> ds = MindData(file_path1, {"label"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];
    TEST_MS_LOG_MSTENSOR(INFO, "Tensor label: ", label);
    iter->GetNextRow(&row);
  }

  // Shard file "mindrecord0/mindrecord1/mindrecord2/mindrecord3" have same dataset info,
  // thus if input file is any of them, all records in imagenet.mindrecord* will be read
  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestMindDataSuccess5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMindDataSuccess5 with specified sampler.";

  // Create a MindData Dataset
  // Pass one mindrecord shard file to parse dataset info, and search for other mindrecord files with same dataset info,
  // thus all records in imagenet.mindrecord0 ~ imagenet.mindrecord3 will be read
  std::string file_path1 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::shared_ptr<Dataset> ds = MindData(file_path1, {}, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::shared_ptr<Tensor> de_expect_item;
  ASSERT_OK(Tensor::CreateScalar((int64_t)0, &de_expect_item));
  mindspore::MSTensor expect_item = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expect_item));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];

    EXPECT_MSTENSOR_EQ(label, expect_item);

    iter->GetNextRow(&row);
  }

  // SequentialSampler will return 3 samples
  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestMindDataSuccess6) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMindDataSuccess6 with num_samples out of range.";

  // Create a MindData Dataset
  // Pass a list of mindrecord file name, files in list will be read directly but not search for related files
  // imagenet.mindrecord0 file has 5 rows, but num_samples is larger than 5
  std::string file_path1 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::vector<std::string> file_list = {file_path1};

  // Check sequential sampler, output number is 5
  std::shared_ptr<Dataset> ds1 = MindData(file_list, {}, std::make_shared<SequentialSampler>(0, 10));
  EXPECT_NE(ds1, nullptr);

  // Check random sampler, output number is 5, same rows with file
  std::shared_ptr<Dataset> ds2 = MindData(file_list, {}, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds2, nullptr);

  // Check pk sampler, output number is 2, get 2 samples with label 0
  std::shared_ptr<Dataset> ds3 = MindData(file_list, {}, std::make_shared<PKSampler>(2, false, 10));
  EXPECT_NE(ds3, nullptr);

  // Check distributed sampler, output number is 3, get 3 samples in shard 0
  std::shared_ptr<Dataset> ds4 = MindData(file_list, {}, std::make_shared<DistributedSampler>(2, 0, false, 10));
  EXPECT_NE(ds4, nullptr);

  // Check distributed sampler get 3 samples with indice 0, 1 ,2
  std::shared_ptr<Dataset> ds5 = MindData(file_list, {}, new SubsetRandomSampler({0, 1, 2}, 10));
  EXPECT_NE(ds5, nullptr);

  std::shared_ptr<Dataset> ds6 = MindData(file_list, {}, new SubsetSampler({1, 2}, 10));
  EXPECT_NE(ds5, nullptr);

  std::vector<std::shared_ptr<Dataset>> ds = {ds1, ds2, ds3, ds4, ds5, ds6};
  std::vector<int32_t> expected_samples = {5, 5, 2, 3, 3, 2};

  for (int32_t i = 0; i < ds.size(); i++) {
    // Create an iterator over the result of the above dataset
    // This will trigger the creation of the Execution Tree and launch it.
    std::shared_ptr<Iterator> iter = ds[i]->CreateIterator();
    EXPECT_NE(iter, nullptr);

    // Iterate the dataset and get each row
    std::unordered_map<std::string, mindspore::MSTensor> row;
    iter->GetNextRow(&row);

    uint64_t j = 0;
    while (row.size() != 0) {
      j++;
      auto label = row["label"];
      TEST_MS_LOG_MSTENSOR(INFO, "Tensor label: ", label);
      iter->GetNextRow(&row);
    }
    EXPECT_EQ(j, expected_samples[i]);

    // Manually terminate the pipeline
    iter->Stop();
  }
}

TEST_F(MindDataTestPipeline, TestMindDataSuccess7) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMindDataSuccess7 with padded sample.";

  // Create pad sample for MindDataset
  auto pad = nlohmann::json::object();
  pad["file_name"] = "does_not_exist.jpg";
  pad["label"] = 999;

  // Create a MindData Dataset
  // Pass a list of mindrecord file name, files in list will be read directly but not search for related files
  std::string file_path1 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::vector<std::string> file_list = {file_path1};
  std::shared_ptr<Dataset> ds =
    MindData(file_list, {"file_name", "label"}, std::make_shared<SequentialSampler>(), &pad, 4);
  EXPECT_NE(ds, nullptr);

  // Create a Skip operation on ds, skip original data in mindrecord and get padded samples
  ds = ds->Skip(5);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::shared_ptr<Tensor> de_expect_item;
  ASSERT_OK(Tensor::CreateScalar((int64_t)999, &de_expect_item));
  mindspore::MSTensor expect_item = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expect_item));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["file_name"];
    auto label = row["label"];
    TEST_MS_LOG_MSTENSOR(INFO, "Tensor image file name: ", image);
    TEST_MS_LOG_MSTENSOR(INFO, "Tensor label: ", label);

    EXPECT_MSTENSOR_EQ(label, expect_item);

    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestMindDataSuccess8) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMindDataSuccess8 with padded sample.";

  // Create pad sample for MindDataset
  auto pad = nlohmann::json::object();
  pad["file_name"] = "does_not_exist.jpg";
  pad["label"] = 999;

  // Create a MindData Dataset
  // Pass a list of mindrecord file name, files in list will be read directly but not search for related files
  std::string file_path1 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::vector<std::string> file_list = {file_path1};
  std::shared_ptr<Dataset> ds =
    MindData(file_list, {"file_name", "label"}, std::make_shared<SequentialSampler>(), &pad, 4);
  EXPECT_NE(ds, nullptr);

  std::vector<mindspore::dataset::DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<mindspore::dataset::TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"file_name", "label"};
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0].ToString(), "string");
  EXPECT_EQ(types[1].ToString(), "int64");
  EXPECT_EQ(shapes.size(), 2);
  EXPECT_EQ(shapes[0].ToString(), "<>");
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(ds->GetDatasetSize(), 5);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetColumnNames(), column_names);

  // Create a Skip operation on ds, skip original data in mindrecord and get padded samples
  ds = ds->Skip(5);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);
  EXPECT_EQ(ds->GetRepeatCount(), 2);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::shared_ptr<Tensor> de_expect_item;
  ASSERT_OK(Tensor::CreateScalar((int64_t)999, &de_expect_item));
  mindspore::MSTensor expect_item = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expect_item));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["file_name"];
    auto label = row["label"];
    TEST_MS_LOG_MSTENSOR(INFO, "Tensor image file name: ", image);
    TEST_MS_LOG_MSTENSOR(INFO, "Tensor label: ", label);

    EXPECT_MSTENSOR_EQ(label, expect_item);

    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestMindDataSuccess9) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMindDataSuccess9 with padded sample.";

  // Create pad sample for MindDataset
  auto pad = nlohmann::json::object();
  pad["file_name"] = "does_not_exist.jpg";
  pad["label"] = 999;

  // Create a MindData Dataset
  // Pass a list of mindrecord file name, files in list will be read directly but not search for related files
  std::string file_path1 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::vector<std::string> file_list = {file_path1};
  std::shared_ptr<Dataset> ds1 =
    MindData(file_list, {"file_name", "label"}, std::make_shared<SequentialSampler>(), &pad, 4);
  EXPECT_NE(ds1, nullptr);
  ds1 = ds1->Skip(5);
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Dataset> ds2 =
    MindData(file_list, {"file_name", "label"}, std::make_shared<SequentialSampler>(), &pad, 4);
  EXPECT_NE(ds2, nullptr);
  ds2 = ds2->Skip(5);
  EXPECT_NE(ds2, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 3;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create a Project operation on ds
  std::vector<std::string> column_project = {"label"};
  ds1 = ds1->Project(column_project);
  EXPECT_NE(ds1, nullptr);
  ds2 = ds2->Project(column_project);
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds
  ds1 = ds1->Concat({ds2});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::shared_ptr<Tensor> de_expect_item;
  ASSERT_OK(Tensor::CreateScalar((int64_t)999, &de_expect_item));
  mindspore::MSTensor expect_item = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expect_item));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto label = row["label"];
    TEST_MS_LOG_MSTENSOR(INFO, "Tensor label: ", label);

    EXPECT_MSTENSOR_EQ(label, expect_item);

    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestMindDataFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMindDataFail1 with incorrect file path.";

  // Create a MindData Dataset with incorrect pattern
  std::string file_path1 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/apple.mindrecord0";
  std::shared_ptr<Dataset> ds1 = MindData(file_path1);
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid MindData input with incorrect pattern
  EXPECT_EQ(iter1, nullptr);

  // Create a MindData Dataset with incorrect file path
  std::string file_path2 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/apple.mindrecord0";
  std::vector<std::string> file_list = {file_path2};
  std::shared_ptr<Dataset> ds2 = MindData(file_list);
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid MindData input with incorrect file path
  EXPECT_EQ(iter2, nullptr);

  // Create a MindData Dataset with incorrect file path
  // ATTENTION: file_path3 is not a pattern to search for ".mindrecord*"
  std::string file_path3 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord";
  std::shared_ptr<Dataset> ds3 = MindData(file_path3);
  EXPECT_NE(ds3, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid MindData input with incorrect file path
  EXPECT_EQ(iter3, nullptr);
}

TEST_F(MindDataTestPipeline, TestMindDataFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMindDataFail2 with incorrect column name.";

  // Create a MindData Dataset with incorrect column name
  std::string file_path1 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::shared_ptr<Dataset> ds1 = MindData(file_path1, {""});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid MindData input with incorrect column name
  EXPECT_EQ(iter1, nullptr);

  // Create a MindData Dataset with duplicate column name
  std::string file_path2 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::shared_ptr<Dataset> ds2 = MindData(file_path2, {"label", "label"});
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid MindData input with duplicate column name
  EXPECT_EQ(iter2, nullptr);

  // Create a MindData Dataset with unexpected column name
  std::string file_path3 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::vector<std::string> file_list = {file_path3};
  std::shared_ptr<Dataset> ds3 = MindData(file_list, {"label", "not_exist"});
  EXPECT_NE(ds3, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid MindData input with unexpected column name
  EXPECT_EQ(iter3, nullptr);
}

TEST_F(MindDataTestPipeline, TestMindDataFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMindDataFail3 with unsupported sampler.";

  // Create a MindData Dataset with unsupported sampler
  std::string file_path1 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::shared_ptr<Dataset> ds1 = MindData(file_path1, {}, new WeightedRandomSampler({1, 1, 1, 1}));
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid MindData input with unsupported sampler
  EXPECT_EQ(iter1, nullptr);

  // Create a MindData Dataset with incorrect sampler
  std::string file_path2 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::shared_ptr<Dataset> ds2 = MindData(file_path2, {}, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid MindData input with incorrect sampler
  EXPECT_EQ(iter2, nullptr);
}

TEST_F(MindDataTestPipeline, TestMindDataFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMindData with padded sample.";

  // Create a MindData Dataset
  std::string file_path1 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::shared_ptr<Dataset> ds1 = MindData(file_path1, {}, std::make_shared<RandomSampler>(), nullptr, 2);
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid MindData input, num_padded is specified but padded_sample is not
  EXPECT_EQ(iter1, nullptr);

  // Create padded sample for MindDataset
  auto pad = nlohmann::json::object();
  pad["file_name"] = "1.jpg";
  pad["label"] = 123456;

  // Create a MindData Dataset
  std::string file_path2 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::shared_ptr<Dataset> ds2 = MindData(file_path2, {"label"}, std::make_shared<RandomSampler>(), &pad, -2);
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid MindData input, num_padded is not greater than or equal to zero
  EXPECT_EQ(iter2, nullptr);

  // Create a MindData Dataset
  std::string file_path3 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::shared_ptr<Dataset> ds3 = MindData(file_path3, {}, std::make_shared<RandomSampler>(), &pad, 1);
  EXPECT_NE(ds3, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid MindData input, padded_sample is specified but requires columns_list as well
  EXPECT_EQ(iter3, nullptr);

  // Create padded sample with unmatched column name
  auto pad2 = nlohmann::json::object();
  pad2["a"] = "1.jpg";
  pad2["b"] = 123456;

  // Create a MindData Dataset
  std::string file_path4 = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::shared_ptr<Dataset> ds4 =
    MindData(file_path4, {"file_name", "label"}, std::make_shared<RandomSampler>(), &pad2, 1);
  EXPECT_NE(ds4, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid MindData input, columns_list does not match any column in padded_sample
  EXPECT_EQ(iter4, nullptr);
}
