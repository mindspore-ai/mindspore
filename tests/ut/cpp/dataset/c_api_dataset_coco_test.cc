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

using namespace mindspore::dataset;
using mindspore::dataset::dsize_t;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestCocoDefault) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCocoDefault.";
  // Create a Coco Dataset
  std::string folder_path = datasets_root_path_ + "/testCOCO/train";
  std::string annotation_file = datasets_root_path_ + "/testCOCO/annotations/train.json";

  std::shared_ptr<Dataset> ds = Coco(folder_path, annotation_file);
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
    auto image = row["image"];
    auto bbox = row["bbox"];
    auto category_id = row["category_id"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor bbox shape: " << bbox.Shape();
    MS_LOG(INFO) << "Tensor category_id shape: " << category_id.Shape();
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCocoDefaultWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCocoDefaultWithPipeline.";
  // Create two Coco Dataset
  std::string folder_path = datasets_root_path_ + "/testCOCO/train";
  std::string annotation_file = datasets_root_path_ + "/testCOCO/annotations/train.json";

  std::shared_ptr<Dataset> ds1 = Coco(folder_path, annotation_file);
  std::shared_ptr<Dataset> ds2 = Coco(folder_path, annotation_file);
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds
  int32_t repeat_num = 2;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 3;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds
  std::vector<std::string> column_project = {"image", "bbox", "category_id"};
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

  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto bbox = row["bbox"];
    auto category_id = row["category_id"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor bbox shape: " << bbox.Shape();
    MS_LOG(INFO) << "Tensor category_id shape: " << category_id.Shape();
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 30);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCocoGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCocoGetters.";
  // Create a Coco Dataset
  std::string folder_path = datasets_root_path_ + "/testCOCO/train";
  std::string annotation_file = datasets_root_path_ + "/testCOCO/annotations/train.json";

  std::shared_ptr<Dataset> ds = Coco(folder_path, annotation_file);
  EXPECT_NE(ds, nullptr);

  std::vector<std::string> column_names = {"image", "bbox", "category_id", "iscrowd"};
  EXPECT_EQ(ds->GetDatasetSize(), 6);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

TEST_F(MindDataTestPipeline, TestCocoDetection) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCocoDetection.";
  // Create a Coco Dataset
  std::string folder_path = datasets_root_path_ + "/testCOCO/train";
  std::string annotation_file = datasets_root_path_ + "/testCOCO/annotations/train.json";

  std::shared_ptr<Dataset> ds =
    Coco(folder_path, annotation_file, "Detection", false, std::make_shared<SequentialSampler>(0, 6));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::string expect_file[] = {"000000391895", "000000318219", "000000554625",
                               "000000574769", "000000060623", "000000309022"};
  std::vector<std::vector<float>> expect_bbox_vector = {{10.0, 10.0, 10.0, 10.0, 70.0, 70.0, 70.0, 70.0},
                                                        {20.0, 20.0, 20.0, 20.0, 80.0, 80.0, 80.0, 80.0},
                                                        {30.0, 30.0, 30.0, 30.0},
                                                        {40.0, 40.0, 40.0, 40.0},
                                                        {50.0, 50.0, 50.0, 50.0},
                                                        {60.0, 60.0, 60.0, 60.0}};
  std::vector<std::vector<uint32_t>> expect_catagoryid_list = {{1, 7}, {2, 8}, {3}, {4}, {5}, {6}};
  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto bbox = row["bbox"];
    auto category_id = row["category_id"];

    mindspore::MSTensor expect_image = ReadFileToTensor(folder_path + "/" + expect_file[i] + ".jpg");
    EXPECT_MSTENSOR_EQ(image, expect_image);

    std::shared_ptr<Tensor> de_expect_bbox;
    dsize_t bbox_num = static_cast<dsize_t>(expect_bbox_vector[i].size() / 4);
    ASSERT_OK(Tensor::CreateFromVector(expect_bbox_vector[i], TensorShape({bbox_num, 4}), &de_expect_bbox));
    mindspore::MSTensor expect_bbox =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expect_bbox));
    EXPECT_MSTENSOR_EQ(bbox, expect_bbox);

    std::shared_ptr<Tensor> de_expect_categoryid;
    ASSERT_OK(Tensor::CreateFromVector(expect_catagoryid_list[i], TensorShape({bbox_num, 1}), &de_expect_categoryid));
    mindspore::MSTensor expect_categoryid =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expect_categoryid));
    EXPECT_MSTENSOR_EQ(category_id, expect_categoryid);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCocoFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCocoFail.";
  // Create a Coco Dataset
  std::string folder_path = datasets_root_path_ + "/testCOCO/train";
  std::string annotation_file = datasets_root_path_ + "/testCOCO/annotations/train.json";
  std::string invalid_folder_path = "./NotExist";
  std::string invalid_annotation_file = "./NotExistFile";

  std::shared_ptr<Dataset> ds0 = Coco(invalid_folder_path, annotation_file);
  EXPECT_NE(ds0, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid COCO input
  EXPECT_EQ(iter0, nullptr);

  std::shared_ptr<Dataset> ds1 = Coco(folder_path, invalid_annotation_file);
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid COCO input
  EXPECT_EQ(iter1, nullptr);

  std::shared_ptr<Dataset> ds2 = Coco(folder_path, annotation_file, "valid_mode");
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid COCO input
  EXPECT_EQ(iter2, nullptr);
}

TEST_F(MindDataTestPipeline, TestCocoKeypoint) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCocoKeypoint.";
  // Create a Coco Dataset
  std::string folder_path = datasets_root_path_ + "/testCOCO/train";
  std::string annotation_file = datasets_root_path_ + "/testCOCO/annotations/key_point.json";

  std::shared_ptr<Dataset> ds =
    Coco(folder_path, annotation_file, "Keypoint", false, std::make_shared<SequentialSampler>(0, 2));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::string expect_file[] = {"000000391895", "000000318219"};
  std::vector<std::vector<float>> expect_keypoint_vector = {
    {368.0, 61.0,  1.0, 369.0, 52.0,  2.0, 0.0,   0.0,   0.0, 382.0, 48.0,  2.0, 0.0,   0.0,   0.0, 368.0, 84.0,  2.0,
     435.0, 81.0,  2.0, 362.0, 125.0, 2.0, 446.0, 125.0, 2.0, 360.0, 153.0, 2.0, 0.0,   0.0,   0.0, 397.0, 167.0, 1.0,
     439.0, 166.0, 1.0, 369.0, 193.0, 2.0, 461.0, 234.0, 2.0, 361.0, 246.0, 2.0, 474.0, 287.0, 2.0},
    {244.0, 139.0, 2.0, 0.0,   0.0,   0.0, 226.0, 118.0, 2.0, 0.0,   0.0,   0.0, 154.0, 159.0, 2.0, 143.0, 261.0, 2.0,
     135.0, 312.0, 2.0, 271.0, 423.0, 2.0, 184.0, 530.0, 2.0, 261.0, 280.0, 2.0, 347.0, 592.0, 2.0, 0.0,   0.0,   0.0,
     123.0, 596.0, 2.0, 0.0,   0.0,   0.0, 0.0,   0.0,   0.0, 0.0,   0.0,   0.0, 0.0,   0.0,   0.0}};
  std::vector<std::vector<dsize_t>> expect_size = {{1, 51}, {1, 51}};
  std::vector<std::vector<uint32_t>> expect_num_keypoints_list = {{14}, {10}};
  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto keypoints = row["keypoints"];
    auto num_keypoints = row["num_keypoints"];

    mindspore::MSTensor expect_image = ReadFileToTensor(folder_path + "/" + expect_file[i] + ".jpg");
    EXPECT_MSTENSOR_EQ(image, expect_image);

    std::shared_ptr<Tensor> de_expect_keypoints;
    dsize_t keypoints_size = expect_size[i][0];
    ASSERT_OK(Tensor::CreateFromVector(expect_keypoint_vector[i], TensorShape(expect_size[i]), &de_expect_keypoints));
    mindspore::MSTensor expect_keypoints =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expect_keypoints));
    EXPECT_MSTENSOR_EQ(keypoints, expect_keypoints);

    std::shared_ptr<Tensor> de_expect_num_keypoints;
    ASSERT_OK(Tensor::CreateFromVector(expect_num_keypoints_list[i], TensorShape({keypoints_size, 1}),
                                       &de_expect_num_keypoints));
    mindspore::MSTensor expect_num_keypoints =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expect_num_keypoints));
    EXPECT_MSTENSOR_EQ(num_keypoints, expect_num_keypoints);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCocoPanoptic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCocoPanoptic.";
  // Create a Coco Dataset
  std::string folder_path = datasets_root_path_ + "/testCOCO/train";
  std::string annotation_file = datasets_root_path_ + "/testCOCO/annotations/panoptic.json";

  std::shared_ptr<Dataset> ds =
    Coco(folder_path, annotation_file, "Panoptic", false, std::make_shared<SequentialSampler>(0, 2));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::string expect_file[] = {"000000391895", "000000574769"};
  std::vector<std::vector<float>> expect_bbox_vector = {{472, 173, 36, 48, 340, 22, 154, 301, 486, 183, 30, 35},
                                                        {103, 133, 229, 422, 243, 175, 93, 164}};
  std::vector<std::vector<uint32_t>> expect_categoryid_vector = {{1, 1, 2}, {1, 3}};
  std::vector<std::vector<uint32_t>> expect_iscrowd_vector = {{0, 0, 0}, {0, 0}};
  std::vector<std::vector<uint32_t>> expect_area_vector = {{705, 14062, 626}, {43102, 6079}};
  std::vector<std::vector<dsize_t>> expect_size = {{3, 4}, {2, 4}};
  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto bbox = row["bbox"];
    auto category_id = row["category_id"];
    auto iscrowd = row["iscrowd"];
    auto area = row["area"];

    mindspore::MSTensor expect_image = ReadFileToTensor(folder_path + "/" + expect_file[i] + ".jpg");
    EXPECT_MSTENSOR_EQ(image, expect_image);

    std::shared_ptr<Tensor> de_expect_bbox;
    dsize_t bbox_size = expect_size[i][0];
    ASSERT_OK(Tensor::CreateFromVector(expect_bbox_vector[i], TensorShape(expect_size[i]), &de_expect_bbox));
    mindspore::MSTensor expect_bbox =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expect_bbox));
    EXPECT_MSTENSOR_EQ(bbox, expect_bbox);

    std::shared_ptr<Tensor> de_expect_categoryid;
    ASSERT_OK(Tensor::CreateFromVector(expect_categoryid_vector[i], TensorShape({bbox_size, 1}), &de_expect_categoryid));
    mindspore::MSTensor expect_categoryid =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expect_categoryid));
    EXPECT_MSTENSOR_EQ(category_id, expect_categoryid);

    std::shared_ptr<Tensor> de_expect_iscrowd;
    ASSERT_OK(Tensor::CreateFromVector(expect_iscrowd_vector[i], TensorShape({bbox_size, 1}), &de_expect_iscrowd));
    mindspore::MSTensor expect_iscrowd =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expect_iscrowd));
    EXPECT_MSTENSOR_EQ(iscrowd, expect_iscrowd);

    std::shared_ptr<Tensor> de_expect_area;
    ASSERT_OK(Tensor::CreateFromVector(expect_area_vector[i], TensorShape({bbox_size, 1}), &de_expect_area));
    mindspore::MSTensor expect_area =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expect_area));
    EXPECT_MSTENSOR_EQ(area, expect_area);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCocoPanopticGetClassIndex) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCocoPanopticGetClassIndex.";
  // Create a Coco Dataset
  std::string folder_path = datasets_root_path_ + "/testCOCO/train";
  std::string annotation_file = datasets_root_path_ + "/testCOCO/annotations/panoptic.json";

  std::shared_ptr<Dataset> ds =
    Coco(folder_path, annotation_file, "Panoptic", false, std::make_shared<SequentialSampler>(0, 2));
  EXPECT_NE(ds, nullptr);

  std::vector<std::pair<std::string, std::vector<int32_t>>> class_index1 = ds->GetClassIndexing();
  EXPECT_EQ(class_index1.size(), 3);
  EXPECT_EQ(class_index1[0].first, "person");
  EXPECT_EQ(class_index1[0].second[0], 1);
  EXPECT_EQ(class_index1[0].second[1], 1);
  EXPECT_EQ(class_index1[1].first, "bicycle");
  EXPECT_EQ(class_index1[1].second[0], 2);
  EXPECT_EQ(class_index1[1].second[1], 1);
  EXPECT_EQ(class_index1[2].first, "car");
  EXPECT_EQ(class_index1[2].second[0], 3);
  EXPECT_EQ(class_index1[2].second[1], 1);
}

TEST_F(MindDataTestPipeline, TestCocoStuff) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCocoStuff.";
  // Create a Coco Dataset
  std::string folder_path = datasets_root_path_ + "/testCOCO/train";
  std::string annotation_file = datasets_root_path_ + "/testCOCO/annotations/train.json";

  std::shared_ptr<Dataset> ds =
    Coco(folder_path, annotation_file, "Stuff", false, std::make_shared<SequentialSampler>(0, 6));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::string expect_file[] = {"000000391895", "000000318219", "000000554625",
                               "000000574769", "000000060623", "000000309022"};
  std::vector<std::vector<float>> expect_segmentation_vector = {
    {10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
     70.0, 72.0, 73.0, 74.0, 75.0, -1.0, -1.0, -1.0, -1.0, -1.0},
    {20.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
     10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, -1.0},
    {40.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 40.0, 41.0, 42.0},
    {50.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0},
    {60.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0},
    {60.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0}};
  std::vector<std::vector<dsize_t>> expect_size = {{2, 10}, {2, 11}, {1, 12}, {1, 13}, {1, 14}, {2, 7}};
  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto segmentation = row["segmentation"];

    mindspore::MSTensor expect_image = ReadFileToTensor(folder_path + "/" + expect_file[i] + ".jpg");
    EXPECT_MSTENSOR_EQ(image, expect_image);

    std::shared_ptr<Tensor> de_expect_segmentation;
    ASSERT_OK(Tensor::CreateFromVector(expect_segmentation_vector[i], TensorShape(expect_size[i]), &de_expect_segmentation));
    mindspore::MSTensor expect_segmentation =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expect_segmentation));
    EXPECT_MSTENSOR_EQ(segmentation, expect_segmentation);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCocoWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCocoWithNullSamplerFail.";
  // Create a Coco Dataset
  std::string folder_path = datasets_root_path_ + "/testCOCO/train";
  std::string annotation_file = datasets_root_path_ + "/testCOCO/annotations/train.json";

  std::shared_ptr<Dataset> ds = Coco(folder_path, annotation_file, "Detection", false, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid COCO input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}
