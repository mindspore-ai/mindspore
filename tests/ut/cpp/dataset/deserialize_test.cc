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
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/serdes.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/vision.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/kernels/ir/data/transforms_ir.h"

using namespace mindspore::dataset;
using mindspore::dataset::DatasetNode;

using mindspore::dataset::ShuffleMode;
using mindspore::dataset::Tensor;

class MindDataTestDeserialize : public UT::DatasetOpTesting {
 protected:
};

void compare_dataset(std::shared_ptr<DatasetNode> ds) {
  nlohmann::json out_json;
  ASSERT_OK(Serdes::SaveToJSON(ds, "dataset_pipeline.json", &out_json));
  // output the deserialized out_json to ds1 and then out_json1
  std::shared_ptr<DatasetNode> ds1;
  ASSERT_OK(Serdes::Deserialize("dataset_pipeline.json", &ds1));
  EXPECT_NE(ds1, nullptr);

  // check original and deserialized dataset are the same
  nlohmann::json out_json1;
  ASSERT_OK(Serdes::SaveToJSON(ds1, "dataset_pipeline_1.json", &out_json1));
  std::stringstream json_ss;
  json_ss << out_json;
  std::stringstream json_ss1;
  json_ss1 << out_json1;
  EXPECT_EQ(json_ss.str(), json_ss1.str());
  return;
}

// test mnist dataset, and special cases of tensor operations (no input or tensor operation input)
TEST_F(MindDataTestDeserialize, TestDeserializeMnist) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-Minist.";
  std::string data_dir = "./data/dataset/testMnistData";
  std::string usage = "all";
  std::shared_ptr<SamplerObj> sampler = std::make_shared<RandomSamplerObj>(true, 100);
  std::shared_ptr<DatasetNode> ds = std::make_shared<MnistNode>(data_dir, usage, sampler, nullptr);
  std::shared_ptr<TensorOperation> operation0 = std::make_shared<vision::EqualizeOperation>();
  std::shared_ptr<TensorOperation> operation1 = std::make_shared<vision::BoundingBoxAugmentOperation>(operation0, 0.5);
  std::shared_ptr<TensorOperation> operation2 = std::make_shared<vision::HorizontalFlipOperation>();
  std::shared_ptr<TensorOperation> operation3 = std::make_shared<vision::HwcToChwOperation>();
  std::shared_ptr<TensorOperation> operation4 = std::make_shared<vision::RgbaToBgrOperation>();
  std::shared_ptr<TensorOperation> operation5 = std::make_shared<vision::RgbaToRgbOperation>();
  std::shared_ptr<TensorOperation> operation6 = std::make_shared<vision::SwapRedBlueOperation>();
  std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> policy;
  std::vector<std::pair<std::shared_ptr<TensorOperation>, double>> sub_policy;
  sub_policy.push_back(std::make_pair(operation1, 0.4));
  policy.push_back(sub_policy);
  std::shared_ptr<TensorOperation> operation7 = std::make_shared<vision::RandomSelectSubpolicyOperation>(policy);
  std::vector<std::shared_ptr<TensorOperation>> transforms;
  transforms.push_back(operation2);
  transforms.push_back(operation3);
  transforms.push_back(operation4);
  std::shared_ptr<TensorOperation> operation8 = std::make_shared<vision::UniformAugOperation>(transforms, 3);
  transforms.push_back(operation5);
  transforms.push_back(operation6);
  transforms.push_back(operation7);
  transforms.push_back(operation8);
  ds = std::make_shared<MapNode>(ds, transforms);
  ds = std::make_shared<BatchNode>(ds, 10, true);
  compare_dataset(ds);
}

// test celeba dataset and part of the tensor operation
TEST_F(MindDataTestDeserialize, TestDeserializeCelebA) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-CelebA.";
  std::string data_dir = "./data/dataset/testCelebAData/";
  std::string usage = "all";
  std::shared_ptr<SamplerObj> sampler = std::make_shared<DistributedSamplerObj>(1, 0, true, 2, 1, 1, true);
  bool decode = true;
  std::set<std::string> extensions = {};
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::shared_ptr<DatasetNode> ds = std::make_shared<CelebANode>(data_dir, usage, sampler, decode, extensions, cache);
  std::vector<int32_t> size = {80, 80};
  std::vector<int32_t> size1 = {80, 80};
  std::vector<int32_t> coordinates = {5, 5};
  std::vector<int32_t> padding = {20, 20, 20, 20};
  std::vector<uint8_t> fill_value = {20, 20, 20};
  std::vector<uint32_t> ignore = {20, 20, 20, 20};
  std::vector<float> mean = {2.0, 2.0, 2.0, 2.0};
  std::vector<float> std = {0.5, 0.5, 0.5, 0.5};
  std::vector<float> translation = {0.5, 0.5};
  std::vector<float> shear = {0.5, 0.5};
  std::vector<float> sigma = {0.5, 0.5};
  InterpolationMode interpolation = InterpolationMode::kLinear;
  std::shared_ptr<TensorOperation> operation0 =
    std::make_shared<vision::AffineOperation>(0.0, translation, 0.5, shear, interpolation, fill_value);
  std::shared_ptr<TensorOperation> operation1 = std::make_shared<vision::AutoContrastOperation>(0.5, ignore);
  std::shared_ptr<TensorOperation> operation2 = std::make_shared<vision::CenterCropOperation>(size);
  std::shared_ptr<TensorOperation> operation3 =
    std::make_shared<vision::CutMixBatchOperation>(ImageBatchFormat::kNHWC, 0.1, 0.1);
  std::shared_ptr<TensorOperation> operation4 = std::make_shared<vision::CutOutOperation>(1, 1);
  std::shared_ptr<TensorOperation> operation5 = std::make_shared<vision::DecodeOperation>(true);
  std::shared_ptr<TensorOperation> operation6 = std::make_shared<vision::GaussianBlurOperation>(coordinates, sigma);
  std::shared_ptr<TensorOperation> operation7 = std::make_shared<vision::MixUpBatchOperation>(1.0);
  std::shared_ptr<TensorOperation> operation8 = std::make_shared<vision::NormalizeOperation>(mean, std);
  std::shared_ptr<TensorOperation> operation9 = std::make_shared<vision::NormalizePadOperation>(mean, std, "float");
  std::shared_ptr<TensorOperation> operation10 =
    std::make_shared<vision::PadOperation>(padding, fill_value, BorderType::kConstant);
  std::shared_ptr<TensorOperation> operation11 = std::make_shared<vision::RescaleOperation>(1.0, 0.5);
  std::shared_ptr<TensorOperation> operation12 = std::make_shared<vision::ResizePreserveAROperation>(10, 10, 0);
  std::shared_ptr<TensorOperation> operation13 = std::make_shared<vision::ResizeWithBBoxOperation>(size, interpolation);
  std::shared_ptr<TensorOperation> operation14 = std::make_shared<vision::ResizeOperation>(size, interpolation);
  std::vector<std::shared_ptr<TensorOperation>> operations;
  operations.push_back(operation0);
  operations.push_back(operation1);
  operations.push_back(operation2);
  operations.push_back(operation3);
  operations.push_back(operation4);
  operations.push_back(operation5);
  operations.push_back(operation6);
  operations.push_back(operation7);
  operations.push_back(operation8);
  operations.push_back(operation9);
  operations.push_back(operation10);
  operations.push_back(operation11);
  operations.push_back(operation12);
  operations.push_back(operation13);
  operations.push_back(operation14);
  ds = std::make_shared<RepeatNode>(ds, 2);
  ds = std::make_shared<MapNode>(ds, operations);
  compare_dataset(ds);
}

// test cifar10 dataset and random tensor operations
TEST_F(MindDataTestDeserialize, TestDeserializeCifar10) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-Cifar10.";
  std::string data_dir = "./data/dataset/testCifar10Data";
  std::string usage = "all";
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::shared_ptr<SamplerObj> sampler = std::make_shared<SequentialSamplerObj>(0, 10);
  std::shared_ptr<DatasetNode> ds = std::make_shared<Cifar10Node>(data_dir, usage, sampler, cache);
  std::vector<float> center = {50.0, 50.0};
  std::vector<uint8_t> threshold = {5, 5};
  std::vector<uint8_t> fill_value = {150, 150, 150};
  std::vector<uint8_t> bit_range = {5, 15};
  std::vector<float> degrees = {0.0, 0.0};
  std::vector<float> scale = {0.5, 0.5};
  std::vector<float> ratio = {0.5, 0.5};
  std::vector<int32_t> size = {224, 224};
  std::vector<int32_t> padding = {20, 20, 20, 20};
  std::vector<float_t> translate_range = {0.0, 0.0, 0.0, 0.0};
  std::vector<float_t> scale_range = {1.0, 1.0};
  std::vector<float_t> shear_ranges = {0.0, 0.0, 0.0, 0.0};
  InterpolationMode interpolation = InterpolationMode::kLinear;
  std::shared_ptr<TensorOperation> operation1 = std::make_shared<vision::RandomRotationOperation>(
    degrees, InterpolationMode::kNearestNeighbour, true, center, fill_value);
  std::shared_ptr<TensorOperation> operation2 = std::make_shared<vision::RandomAffineOperation>(
    degrees, translate_range, scale_range, shear_ranges, interpolation, fill_value);
  std::shared_ptr<TensorOperation> operation3 = std::make_shared<vision::RandomColorOperation>(0.5, 10.5);
  std::shared_ptr<TensorOperation> operation4 =
    std::make_shared<vision::RandomCropDecodeResizeOperation>(size, scale, ratio, interpolation, 2);
  std::shared_ptr<TensorOperation> operation5 =
    std::make_shared<vision::RandomCropWithBBoxOperation>(size, padding, true, fill_value, BorderType::kConstant);
  std::shared_ptr<TensorOperation> operation6 = std::make_shared<vision::RandomHorizontalFlipOperation>(0.1);
  std::shared_ptr<TensorOperation> operation7 = std::make_shared<vision::RandomHorizontalFlipWithBBoxOperation>(0.1);
  std::shared_ptr<TensorOperation> operation8 = std::make_shared<vision::RandomPosterizeOperation>(bit_range);
  std::shared_ptr<TensorOperation> operation9 = std::make_shared<vision::RandomResizeOperation>(size);
  std::shared_ptr<TensorOperation> operation10 = std::make_shared<vision::RandomResizeWithBBoxOperation>(size);
  std::shared_ptr<TensorOperation> operation11 =
    std::make_shared<vision::RandomResizedCropOperation>(size, scale, ratio, interpolation, 2);
  std::shared_ptr<TensorOperation> operation12 =
    std::make_shared<vision::RandomResizedCropWithBBoxOperation>(size, scale, ratio, interpolation, 2);
  std::shared_ptr<TensorOperation> operation13 =
    std::make_shared<vision::RandomRotationOperation>(degrees, interpolation, true, center, fill_value);
  std::shared_ptr<TensorOperation> operation14 = std::make_shared<vision::RandomSharpnessOperation>(degrees);
  std::shared_ptr<TensorOperation> operation15 = std::make_shared<vision::RandomSolarizeOperation>(threshold);
  std::shared_ptr<TensorOperation> operation16 = std::make_shared<vision::RandomVerticalFlipOperation>(0.1);
  std::shared_ptr<TensorOperation> operation17 = std::make_shared<vision::RandomVerticalFlipWithBBoxOperation>(0.1);
  std::vector<std::shared_ptr<TensorOperation>> operations;
  operations.push_back(operation1);
  operations.push_back(operation2);
  operations.push_back(operation3);
  operations.push_back(operation4);
  operations.push_back(operation5);
  operations.push_back(operation6);
  operations.push_back(operation7);
  operations.push_back(operation8);
  operations.push_back(operation9);
  operations.push_back(operation10);
  operations.push_back(operation11);
  operations.push_back(operation12);
  operations.push_back(operation13);
  operations.push_back(operation14);
  operations.push_back(operation15);
  operations.push_back(operation16);
  operations.push_back(operation17);
  ds = std::make_shared<MapNode>(ds, operations);
  ds = std::make_shared<BatchNode>(ds, 1, true);
  ds = std::make_shared<SkipNode>(ds, 1);
  compare_dataset(ds);
}

TEST_F(MindDataTestDeserialize, TestDeserializeCifar100) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-Cifar100.";
  std::string data_dir = "./data/dataset/testCifar100Data";
  std::string usage = "all";
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::shared_ptr<SamplerObj> sampler = std::make_shared<SequentialSamplerObj>(0, 10);
  std::shared_ptr<DatasetNode> ds = std::make_shared<Cifar100Node>(data_dir, usage, sampler, cache);
  ds = std::make_shared<TakeNode>(ds, 6);
  std::shared_ptr<TensorOperation> operation = std::make_shared<vision::HorizontalFlipOperation>();
  std::vector<std::shared_ptr<TensorOperation>> ops = {operation};
  ds = std::make_shared<MapNode>(ds, ops);
  std::vector<std::shared_ptr<TensorOperation>> operations;
  std::vector<int32_t> size = {32, 32};
  std::vector<int32_t> padding = {4, 4, 4, 4};
  bool pad_if_needed = false;
  std::vector<uint8_t> fill_value = {4, 4, 4};
  InterpolationMode interpolation = InterpolationMode::kLinear;
  std::shared_ptr<TensorOperation> operation1 =
    std::make_shared<vision::RandomCropOperation>(size, padding, pad_if_needed, fill_value, BorderType::kConstant);
  size = {224, 224};
  std::shared_ptr<TensorOperation> operation2 = std::make_shared<vision::ResizeOperation>(size, interpolation);
  std::shared_ptr<TensorOperation> operation3 = std::make_shared<vision::RescaleOperation>(0.5, 0.0);
  std::vector<float> mean = {0.49, 0.48, 0.46};
  std::vector<float> std = {0.20, 0.199, 0.201};
  std::shared_ptr<TensorOperation> operation4 = std::make_shared<vision::NormalizeOperation>(mean, std);
  operations.push_back(operation1);
  operations.push_back(operation2);
  operations.push_back(operation3);
  operations.push_back(operation4);
  ds = std::make_shared<MapNode>(ds, operations);
  ds = std::make_shared<BatchNode>(ds, 3, true);
  ds = std::make_shared<RepeatNode>(ds, 1);
  compare_dataset(ds);
}

TEST_F(MindDataTestDeserialize, TestDeserializeCSV) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-CSV.";
  std::string data_file = "./data/dataset/testCSV/1.csv";
  std::vector<std::string> dataset_files = {data_file};
  char field_delim = ',';
  std::vector<std::string> column_names = {"col1", "col2", "col3", "col4"};
  std::vector<std::string> columns = {"col1", "col4", "col2"};
  std::vector<std::shared_ptr<CsvBase>> column_defaults = {};
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::shared_ptr<SamplerObj> sampler = std::make_shared<SequentialSamplerObj>(0, 10);
  std::shared_ptr<DatasetNode> ds = std::make_shared<CSVNode>(dataset_files, field_delim, column_defaults, column_names,
                                                              3, ShuffleMode::kGlobal, 1, 0, cache);
  ds = std::make_shared<ProjectNode>(ds, columns);
  compare_dataset(ds);
}

TEST_F(MindDataTestDeserialize, TestDeserializeImageFolder) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-ImageFolder.";
  std::string dataset_dir = "./data/dataset/testPK/data";
  std::shared_ptr<SamplerObj> child_sampler = std::make_shared<PKSamplerObj>(3, true, 1);
  std::vector<double> weights = {1.0, 0.1, 0.02, 0.3, 0.4, 0.05, 1.2, 0.13, 0.14, 0.015, 0.16, 1.1};
  std::set<std::string> extensions = {};
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::map<std::string, int32_t> class_indexing = {};
  std::shared_ptr<SamplerObj> sampler = std::make_shared<WeightedRandomSamplerObj>(weights, 11);
  sampler->AddChildSampler(child_sampler);
  std::shared_ptr<DatasetNode> ds =
    std::make_shared<ImageFolderNode>(dataset_dir, false, sampler, false, extensions, class_indexing, cache);
  ds = std::make_shared<RepeatNode>(ds, 1);
  std::vector<int32_t> size = {224, 224};
  std::vector<float> scale = {0.5, 0.5};
  std::vector<float> ratio = {0.5, 0.5};
  std::vector<float> center = {50.0, 50.0};
  std::vector<uint8_t> fill_value = {150, 150, 150};
  InterpolationMode interpolation = InterpolationMode::kLinear;
  std::shared_ptr<TensorOperation> operation1 = std::make_shared<vision::SoftDvppDecodeResizeJpegOperation>(size);
  std::vector<std::shared_ptr<TensorOperation>> ops = {operation1};
  ds = std::make_shared<MapNode>(ds, ops);
  std::vector<std::shared_ptr<TensorOperation>> operations;
  std::shared_ptr<TensorOperation> operation2 =
    std::make_shared<vision::SoftDvppDecodeRandomCropResizeJpegOperation>(size, scale, ratio, 2);
  std::shared_ptr<TensorOperation> operation3 =
    std::make_shared<vision::RotateOperation>(0.5, interpolation, true, center, fill_value);
  operations.push_back(operation2);
  operations.push_back(operation3);
  ds = std::make_shared<MapNode>(ds, operations);
  ds = std::make_shared<BatchNode>(ds, 2, true);
  compare_dataset(ds);
}

TEST_F(MindDataTestDeserialize, TestDeserializeManifest) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-Manifest.";
  std::string data_file = "./data/dataset/testManifestData/cpp.json";
  std::shared_ptr<SamplerObj> sampler = std::make_shared<SequentialSamplerObj>(0, 10);
  std::map<std::string, int32_t> class_indexing = {};
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::shared_ptr<DatasetNode> ds =
    std::make_shared<ManifestNode>(data_file, "train", sampler, class_indexing, false, cache);
  std::vector<int32_t> coordinates = {50, 50};
  std::vector<int32_t> size = {224, 224};
  std::shared_ptr<TensorOperation> operation1 = std::make_shared<vision::CropOperation>(coordinates, size);
  std::shared_ptr<TensorOperation> operation2 = std::make_shared<vision::RgbToBgrOperation>();
  std::shared_ptr<TensorOperation> operation3 = std::make_shared<vision::RgbToGrayOperation>();
  std::shared_ptr<TensorOperation> operation4 =
    std::make_shared<vision::SlicePatchesOperation>(5, 5, SliceMode::kDrop, 1);
  std::shared_ptr<TensorOperation> operation5 = std::make_shared<vision::VerticalFlipOperation>();
  std::vector<std::shared_ptr<TensorOperation>> operations;
  operations.push_back(operation1);
  operations.push_back(operation2);
  operations.push_back(operation3);
  operations.push_back(operation4);
  operations.push_back(operation5);
  ds = std::make_shared<MapNode>(ds, operations);
  ds = std::make_shared<BatchNode>(ds, 2, false);
  compare_dataset(ds);
}

TEST_F(MindDataTestDeserialize, TestDeserializeVOC) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-VOC.";
  std::string dataset_dir = "./data/dataset/testVOC2012";
  std::vector<int64_t> indices = {0, 1};
  std::shared_ptr<SamplerObj> sampler = std::make_shared<SubsetRandomSamplerObj>(indices, 3);
  std::string task = "Detection";
  std::string usage = "train";
  std::map<std::string, int32_t> class_indexing = {};
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::shared_ptr<DatasetNode> ds =
    std::make_shared<VOCNode>(dataset_dir, task, usage, class_indexing, true, sampler, cache);
  std::vector<float> brightness = {0.5, 0.5};
  std::vector<float> contrast = {1.0, 1.0};
  std::vector<float> hue = {0.0, 0.0};
  std::vector<float> saturation = {1.0, 1.0};
  std::shared_ptr<TensorOperation> operation =
    std::make_shared<vision::RandomColorAdjustOperation>(brightness, contrast, saturation, hue);
  std::vector<std::shared_ptr<TensorOperation>> ops = {operation};
  ds = std::make_shared<MapNode>(ds, ops);
  ds = std::make_shared<SkipNode>(ds, 2);
  compare_dataset(ds);
}

TEST_F(MindDataTestDeserialize, TestDeserializeCLUE) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-CLUE.";
  std::string train_file = "./data/dataset/testCLUE/afqmc/train.json";
  std::string task = "AFQMC";
  std::string usage = "train";
  std::vector<std::string> files = {train_file};
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::shared_ptr<DatasetNode> ds = std::make_shared<CLUENode>(files, task, usage, 1, ShuffleMode::kFalse, 1, 0, cache);
  ds = std::make_shared<RepeatNode>(ds, 1);
  std::shared_ptr<TensorOperation> operation1 = std::make_shared<vision::DecodeOperation>(true);
  std::vector<std::shared_ptr<TensorOperation>> ops = {operation1};
  ds = std::make_shared<MapNode>(ds, ops);
  compare_dataset(ds);
}

TEST_F(MindDataTestDeserialize, TestDeserializeCoco) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-Coco.";
  std::string folder_path = "./data/dataset/testCOCO/train";
  std::string annotation_file = "./data/dataset/testCOCO/annotations/train.json";
  std::string task = "Detection";
  std::vector<int64_t> indices = {0, 1};
  std::shared_ptr<SamplerObj> sampler = std::make_shared<SubsetRandomSamplerObj>(indices, 3);
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::shared_ptr<DatasetNode> ds =
    std::make_shared<CocoNode>(folder_path, annotation_file, task, true, sampler, cache, false);
  std::vector<uint8_t> fill_value = {150, 150, 150};
  std::vector<float> degrees = {0.0, 0.0};
  std::vector<float> scale = {0.5, 0.5};
  std::vector<float> ratio = {0.5, 0.5};
  std::vector<int32_t> size = {224, 224};
  std::vector<int32_t> padding = {20, 20, 20, 20};
  InterpolationMode interpolation = InterpolationMode::kLinear;
  std::shared_ptr<TensorOperation> operation1 =
    std::make_shared<vision::RandomCropDecodeResizeOperation>(size, scale, ratio, interpolation, 2);
  std::shared_ptr<TensorOperation> operation2 =
    std::make_shared<vision::RandomCropWithBBoxOperation>(size, padding, true, fill_value, BorderType::kConstant);
  std::shared_ptr<TensorOperation> operation3 = std::make_shared<vision::RandomHorizontalFlipOperation>(0.1);
  std::shared_ptr<TensorOperation> operation4 = std::make_shared<vision::RandomHorizontalFlipWithBBoxOperation>(0.1);
  std::vector<std::shared_ptr<TensorOperation>> operations;
  operations.push_back(operation1);
  operations.push_back(operation2);
  operations.push_back(operation3);
  operations.push_back(operation4);
  ds = std::make_shared<MapNode>(ds, operations);
  compare_dataset(ds);
}

TEST_F(MindDataTestDeserialize, TestDeserializeTFRecord) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-TFRecord.";
  int num_samples = 12;
  int32_t num_shards = 1;
  int32_t shard_id = 0;
  bool shard_equal_rows = false;
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::vector<std::string> columns_list = {};
  std::vector<std::string> dataset_files = {"./data/dataset/testTFTestAllTypes/test.data"};

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt32, {4}));
  ASSERT_OK(schema->add_column("col2", mindspore::DataType::kNumberTypeInt64, {4}));

  std::shared_ptr<DatasetNode> ds =
    std::make_shared<TFRecordNode>(dataset_files, schema, columns_list, num_samples, ShuffleMode::kFiles, num_shards,
                                   shard_id, shard_equal_rows, cache);
  ds = std::make_shared<ShuffleNode>(ds, 10000, true);
  std::vector<std::string> input_columns = {"col_sint16", "col_sint32", "col_sint64", "col_float",
                                            "col_1d",     "col_2d",     "col_3d",     "col_binary"};
  std::vector<std::string> output_columns = {"column_sint16", "column_sint32", "column_sint64", "column_float",
                                             "column_1d",     "column_2d",     "column_3d",     "column_binary"};
  std::shared_ptr<TensorOperation> operation = std::make_shared<vision::InvertOperation>();
  std::vector<std::shared_ptr<TensorOperation>> ops = {operation};
  ds = std::make_shared<MapNode>(ds, ops, input_columns, output_columns);
  std::string train_file = "./data/dataset/testCLUE/afqmc/train.json";
  std::string task1 = "AFQMC";
  std::string usage = "train";
  std::vector<std::string> files = {train_file};
  std::shared_ptr<DatasetNode> ds_child1 =
    std::make_shared<CLUENode>(files, task1, usage, 0, ShuffleMode::kFalse, 1, 0, cache);
  std::vector<std::string> dataset_files2 = {"./data/dataset/testTextFileDataset/1.txt"};
  std::shared_ptr<DatasetNode> ds_child2 =
    std::make_shared<TextFileNode>(dataset_files2, 2, ShuffleMode::kFiles, 1, 0, cache);
  std::vector<std::shared_ptr<DatasetNode>> datasets = {ds, ds_child1, ds_child2};
  ds = std::make_shared<ZipNode>(datasets);
  compare_dataset(ds);
}

TEST_F(MindDataTestDeserialize, TestDeserializeTextfile) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-Textfile.";
  std::vector<std::string> dataset_files = {"./data/dataset/testTextFileDataset/1.txt"};
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::shared_ptr<DatasetNode> ds = std::make_shared<TextFileNode>(dataset_files, 2, ShuffleMode::kFiles, 1, 0, cache);
  std::shared_ptr<TensorOperation> operation = std::make_shared<vision::InvertOperation>();
  std::vector<std::shared_ptr<TensorOperation>> ops = {operation};
  ds = std::make_shared<MapNode>(ds, ops);
  ds = std::make_shared<BatchNode>(ds, 10, true);
  compare_dataset(ds);
}

TEST_F(MindDataTestDeserialize, TestDeserializeInvalidJson) {
  std::shared_ptr<DatasetNode> ds;
  // check the invalid json path would return error
  ASSERT_ERROR(Serdes::Deserialize("invalid_dataset.json", &ds));
  // check the invalid json object would return error
  ASSERT_ERROR(Serdes::Deserialize("./data/dataset/testDataset1/datasetTestInvalidJson.json", &ds));
  EXPECT_EQ(ds, nullptr);
}

TEST_F(MindDataTestDeserialize, TestDeserializeFill) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-Fill.";
  std::vector<std::string> dataset_files = {"./data/dataset/testTextFileDataset/1.txt"};
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::shared_ptr<DatasetNode> ds = std::make_shared<TextFileNode>(dataset_files, 2, ShuffleMode::kFiles, 1, 0, cache);
  std::shared_ptr<Tensor> fill_value;
  ASSERT_OK(Tensor::CreateScalar(true, &fill_value));
  std::shared_ptr<TensorOperation> operation1 = std::make_shared<transforms::FillOperation>(fill_value);
  std::shared_ptr<TensorOperation> operation2 = std::make_shared<text::ToNumberOperation>("int32_t");
  std::vector<std::shared_ptr<TensorOperation>> ops = {operation1, operation2};
  ds = std::make_shared<MapNode>(ds, ops);
  ds = std::make_shared<TransferNode>(ds, "queue", "type", 1, true, 10, true);
  compare_dataset(ds);
}

TEST_F(MindDataTestDeserialize, TestDeserializeTensor) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-Tensor.";
  std::shared_ptr<Tensor> test_tensor;
  std::vector<float> input = {1.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.2, 0.7, 0.8, 0.9, 1.0, 2.0, 1.3, 3.0, 4.0};
  ASSERT_OK(Tensor::CreateFromVector(input, TensorShape{3, 5}, &test_tensor));
  nlohmann::json json_obj;
  ASSERT_OK(test_tensor->to_json(&json_obj));
  std::shared_ptr<Tensor> test_tensor1;
  ASSERT_OK(Tensor::from_json(json_obj, &test_tensor1));
  nlohmann::json json_obj1;
  ASSERT_OK(test_tensor1->to_json(&json_obj1));
  std::stringstream json_ss;
  json_ss << json_obj;
  std::stringstream json_ss1;
  json_ss1 << json_obj1;
  EXPECT_EQ(json_ss.str(), json_ss1.str());
}

// Helper function to get the session id from SESSION_ID env variable
Status GetSessionFromEnv(session_id_type *session_id);

TEST_F(MindDataTestDeserialize, DISABLED_TestDeserializeCache) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-Cache.";
  std::string data_dir = "./data/dataset/testCache";
  std::string usage = "all";
  session_id_type env_session;
  ASSERT_TRUE(GetSessionFromEnv(&env_session));
  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, false, "127.0.0.1", 50052, 1, 1);

  std::shared_ptr<SamplerObj> sampler = std::make_shared<SequentialSamplerObj>(0, 10);
  std::shared_ptr<DatasetNode> ds = std::make_shared<Cifar10Node>(data_dir, usage, sampler, some_cache);
  compare_dataset(ds);
}

TEST_F(MindDataTestDeserialize, TestDeserializeConcatAlbumFlickr) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-ConcatAlbumFlickr.";
  std::string dataset_dir = "./data/dataset/testAlbum";
  std::vector<std::string> column_names = {"col1", "col2", "col3"};
  bool decode = false;
  std::shared_ptr<SamplerObj> sampler = std::make_shared<SequentialSamplerObj>(0, 10);
  std::string data_schema = "./data/dataset/testAlbum/datasetSchema.json";
  std::shared_ptr<DatasetNode> ds =
    std::make_shared<AlbumNode>(dataset_dir, data_schema, column_names, decode, sampler, nullptr);
  std::shared_ptr<TensorOperation> operation = std::make_shared<vision::AdjustGammaOperation>(0.5, 0.5);
  std::vector<std::shared_ptr<TensorOperation>> ops = {operation};
  ds = std::make_shared<MapNode>(ds, ops);
  std::string dataset_path = "./data/dataset/testFlickrData/flickr30k/flickr30k-images";
  std::string annotation_file = "./data/dataset/testFlickrData/flickr30k/test1.token";
  std::shared_ptr<DatasetNode> ds_child1 =
    std::make_shared<FlickrNode>(dataset_path, annotation_file, decode, sampler, nullptr);
  std::vector<std::shared_ptr<DatasetNode>> datasets = {ds, ds_child1};
  std::pair<int, int> pair = std::make_pair(1, 1);
  std::vector<std::pair<int, int>> children_flag_and_nums = {pair};
  std::vector<std::pair<int, int>> children_start_end_index = {pair};
  ds = std::make_shared<ConcatNode>(datasets, sampler, children_flag_and_nums, children_start_end_index);
  compare_dataset(ds);
}

TEST_F(MindDataTestDeserialize, TestDeserializePyFunc) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-PyFunc.";
  std::shared_ptr<DatasetNode> ds1;
  ASSERT_OK(Serdes::Deserialize("./data/dataset/tf_file_dataset/pyvision_dataset_pipeline.json", &ds1));
  EXPECT_NE(ds1, nullptr);
}