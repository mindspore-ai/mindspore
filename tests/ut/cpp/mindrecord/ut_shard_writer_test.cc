/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "utils/ms_utils.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "minddata/mindrecord/include/shard_reader.h"
#include "minddata/mindrecord/include/shard_writer.h"
#include "minddata/mindrecord/include/shard_index_generator.h"
#include "securec.h"
#include "ut_common.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;
using mindspore::MsLogLevel::INFO;

namespace mindspore {
namespace mindrecord {
class TestShardWriter : public UT::Common {
 public:
  TestShardWriter() {}
};

TEST_F(TestShardWriter, TestShardWriterBench) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test write imageNet"));

  ShardWriterImageNet();
  for (int i = 1; i <= 4; i++) {
    string filename = std::string("./imagenet.shard0") + std::to_string(i);
    string db_name = std::string("./imagenet.shard0") + std::to_string(i) + ".db";
    remove(common::SafeCStr(filename));
    remove(common::SafeCStr(db_name));
  }
}

TEST_F(TestShardWriter, TestShardWriterOneSample) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test write imageNet int32 of sample less than num of shards"));
  ShardWriterImageNetOneSample();
  std::string filename = "./OneSample.shard01";

  ShardReader dataset;
  MSRStatus ret = dataset.Open({filename}, true, 4);
  ASSERT_EQ(ret, SUCCESS);
  dataset.Launch();

  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto &j : x) {
      MS_LOG(INFO) << "item size: " << std::get<0>(j).size();
      for (auto &item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << common::SafeCStr(item.key()) << ", value: " << common::SafeCStr(item.value().dump());
      }
    }
  }
  dataset.Close();
  for (int i = 1; i <= 4; i++) {
    string filename = std::string("./OneSample.shard0") + std::to_string(i);
    string db_name = std::string("./OneSample.shard0") + std::to_string(i) + ".db";
    remove(common::SafeCStr(filename));
    remove(common::SafeCStr(db_name));
  }
}

TEST_F(TestShardWriter, TestShardWriterShiftRawPage) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test write shift raw page"));
  const int kMaxNum = 10;
  auto column_list = std::vector<std::string>{"file_name_0"};

  string input_path1 = "./data/mindrecord/testCBGData/data/image_raw_meta.data";
  string input_path3 = "./data/mindrecord/testCBGData/statistics/statistics.txt";
  std::string path_dir = "./data/mindrecord/testCBGData/data/pictures";

  std::vector<std::vector<uint8_t>> bin_data;

  // buffer init
  std::vector<json> json_buffer1;            // store the image_raw_meta.data
  std::vector<json> json_buffer3;            // store the pictures
  std::vector<json> json_buffer4;            // store the statistics data
  std::vector<std::string> image_filenames;  // save all files' path within path_dir

  // read image_raw_meta.data
  LoadData(input_path1, json_buffer1, kMaxNum);
  MS_LOG(INFO) << "Load Meta Data Already.";

  // get files' paths stored in vector<string> image_filenames
  mindrecord::GetAbsoluteFiles(path_dir, image_filenames);  // get all files whose path within path_dir
  MS_LOG(INFO) << "Only process 10 file names:";
  image_filenames.resize(kMaxNum);
  MS_LOG(INFO) << "Load Img Filenames Already.";

  // read pictures
  // mindrecord::Img2DataUint8(image_filenames, bin_data);

  // init shardHeader
  mindrecord::ShardHeader header_data;
  MS_LOG(INFO) << "Init ShardHeader Already.";

  // create schema
  json image_schema_json = R"({"name":{"type":"string"}})"_json;
  json anno_schema_json = R"({"name":{"type":"string"},"anno_tool":{"type":"string"},"creation_time":{"type":"string"},
                                  "max_shape_id":{"type":"int32"},"max_entity_id":{"type":"int32"},
                                  "entity_instances":{"type":"bytes"}})"_json;

  std::shared_ptr<mindrecord::Schema> image_schema = mindrecord::Schema::Build("picture", image_schema_json);
  if (image_schema == nullptr) {
    MS_LOG(ERROR) << "Build image schema failed";
    return;
  }

  // add schema to shardHeader
  int image_schema_id = header_data.AddSchema(image_schema);
  MS_LOG(INFO) << "Init Schema Already.";

  // create/init statistics
  LoadData(input_path3, json_buffer4, 2);
  json static1_json = json_buffer4[0];
  json static2_json = json_buffer4[1];
  MS_LOG(INFO) << "Initial statistics 1 is: " << common::SafeCStr(static1_json.dump());
  MS_LOG(INFO) << "Initial statistics 2 is: " << common::SafeCStr(static2_json.dump());
  std::shared_ptr<mindrecord::Statistics> static1 =
    mindrecord::Statistics::Build(static1_json["description"], static1_json["statistics"]);
  std::shared_ptr<mindrecord::Statistics> static2 =
    mindrecord::Statistics::Build(static2_json["description"], static2_json["statistics"]);
  MS_LOG(INFO) << "Init Statistics Already.";

  // add statistics to shardHeader
  if (static1 == nullptr) {
    MS_LOG(ERROR) << "static1 is nullptr";
    return;
  } else {
    header_data.AddStatistic(static1);
  }
  if (static2 == nullptr) {
    MS_LOG(ERROR) << "static2 is nullptr";
    return;
  } else {
    header_data.AddStatistic(static2);
  }

  // create index field by schema
  std::pair<uint64_t, std::string> index_field1(image_schema_id, "name");
  std::vector<std::pair<uint64_t, std::string>> fields;
  fields.push_back(index_field1);

  // add index to shardHeader
  header_data.AddIndexFields(fields);

  std::map<std::uint64_t, std::vector<json>> rawdatas;
  // merge imgBinaryData(json_buffer3) and imgShardHeader(json_buffer1) to imgBinaryData(json_buffer3)
  std::string dummy_str = std::string(3000, 'a');
  json dummyJson = {};
  dummyJson["name"] = dummy_str;
  std::vector<json> json_buffer;
  for (std::size_t i = 0; i < kMaxNum; i++) {
    json_buffer.push_back(dummyJson);
  }
  rawdatas.insert(pair<uint64_t, vector<json>>(0, json_buffer));

  bin_data.clear();
  auto image = std::vector<uint8_t>(10240, 1);
  for (std::size_t i = 0; i < kMaxNum; i++) {
    bin_data.push_back(image);
  }
  // init file_writer
  MS_LOG(INFO) << "Init Writer ...";
  std::vector<std::string> file_names;

  file_names.push_back("./train_base64.mindrecord01");

  {
    mindrecord::ShardWriter fw;
    fw.Open(file_names);
    uint64_t header_size = 1 << 14;
    uint64_t page_size = 1 << 15;
    fw.SetHeaderSize(header_size);
    fw.SetPageSize(page_size);

    // set shardHeader
    fw.SetShardHeader(std::make_shared<mindrecord::ShardHeader>(header_data));
    fw.WriteRawData(rawdatas, bin_data);
    fw.Commit();
  }

  {
    mindrecord::ShardWriter fw;
    fw.OpenForAppend(file_names[0]);
    fw.WriteRawData(rawdatas, bin_data);
    fw.Commit();
  }

  for (const auto &oneFile : file_names) {
    remove(common::SafeCStr(oneFile));
  }
}

TEST_F(TestShardWriter, TestShardWriterTrial) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test write trial data"));
  int kMaxNum = 10;
  auto column_list = std::vector<std::string>{"file_name_0"};

  string input_path1 = "./data/mindrecord/testCBGData/data/image_raw_meta.data";
  string input_path3 = "./data/mindrecord/testCBGData/statistics/statistics.txt";
  std::string path_dir = "./data/mindrecord/testCBGData/data/pictures";

  std::vector<std::vector<uint8_t>> bin_data;

  // buffer init
  std::vector<json> json_buffer1;            // store the image_raw_meta.data
  std::vector<json> json_buffer3;            // store the pictures
  std::vector<json> json_buffer4;            // store the statistics data
  std::vector<std::string> image_filenames;  // save all files' path within path_dir

  // read image_raw_meta.data
  LoadData(input_path1, json_buffer1, kMaxNum);
  MS_LOG(INFO) << "Load Meta Data Already.";

  // get files' paths stored in vector<string> image_filenames
  mindrecord::GetAbsoluteFiles(path_dir, image_filenames);  // get all files whose path within path_dir
  MS_LOG(INFO) << "Only process 10 file names:";
  image_filenames.resize(kMaxNum);
  MS_LOG(INFO) << "Load Img Filenames Already.";

  // read pictures
  mindrecord::Img2DataUint8(image_filenames, bin_data);

  // init shardHeader
  mindrecord::ShardHeader header_data;
  MS_LOG(INFO) << "Init ShardHeader Already.";

  // create schema
  json image_schema_json = R"({"name":{"type":"string"},"size":{"type":"int32"},"type":{"type":"string"},
                                        "dataset_id":{"type":"int32"},"creation_time":{"type":"string"}})"_json;
  json anno_schema_json = R"({"name":{"type":"string"},"anno_tool":{"type":"string"},"creation_time":{"type":"string"},
                                  "max_shape_id":{"type":"int32"},"max_entity_id":{"type":"int32"},
                                  "entity_instances":{"type":"bytes"}})"_json;

  std::shared_ptr<mindrecord::Schema> image_schema = mindrecord::Schema::Build("picture", image_schema_json);
  if (image_schema == nullptr) {
    MS_LOG(ERROR) << "Build image schema failed";
    return;
  }

  // add schema to shardHeader
  int image_schema_id = header_data.AddSchema(image_schema);
  MS_LOG(INFO) << "Init Schema Already.";

  // create/init statistics
  LoadData(input_path3, json_buffer4, 2);
  json static1_json = json_buffer4[0];
  json static2_json = json_buffer4[1];
  MS_LOG(INFO) << "Initial statistics 1 is: " << common::SafeCStr(static1_json.dump());
  MS_LOG(INFO) << "Initial statistics 2 is: " << common::SafeCStr(static2_json.dump());
  std::shared_ptr<mindrecord::Statistics> static1 =
    mindrecord::Statistics::Build(static1_json["description"], static1_json["statistics"]);
  std::shared_ptr<mindrecord::Statistics> static2 =
    mindrecord::Statistics::Build(static2_json["description"], static2_json["statistics"]);
  MS_LOG(INFO) << "Init Statistics Already.";

  // add statistics to shardHeader
  if (static1 == nullptr) {
    MS_LOG(ERROR) << "static1 is nullptr";
    return;
  } else {
    header_data.AddStatistic(static1);
  }
  if (static2 == nullptr) {
    MS_LOG(ERROR) << "static2 is nullptr";
    return;
  } else {
    header_data.AddStatistic(static2);
  }

  // create index field by schema
  std::pair<uint64_t, std::string> index_field1(image_schema_id, "name");
  std::vector<std::pair<uint64_t, std::string>> fields;
  fields.push_back(index_field1);

  // add index to shardHeader
  header_data.AddIndexFields(fields);

  // merge imgBinaryData(json_buffer3) and imgShardHeader(json_buffer1) to imgBinaryData(json_buffer3)
  for (std::size_t i = 0; i < json_buffer1.size(); i++) {
    json_buffer3.push_back(json{});
  }
  for (std::size_t i = 0; i < json_buffer1.size(); i++) {
    json_buffer3[i] = json_buffer1[i];  // add meta_data to json_buffer3's json variable
  }

  // get json2bson size indicate image size
  json j_test = json_buffer3[0];

  // reference json variable
  std::vector<json> &images = json_buffer3;  // imgBinaryData && imgShardHeader

  // add data
  std::map<std::uint64_t, std::vector<json>> rawdatas;
  rawdatas.insert(pair<uint64_t, vector<json>>(image_schema_id, images));

  // init file_writer
  MS_LOG(INFO) << "Init Writer ...";
  std::vector<std::string> file_names;

  // std::vector<std::string> file_names = {"train_base64.mindrecord01", "train_base64.mindrecord02",
  // "train_base64.mindrecord03"};
  file_names.push_back("./train_base64.mindrecord01");
  file_names.push_back("./train_base64.mindrecord02");
  file_names.push_back("./train_base64.mindrecord03");
  mindrecord::ShardWriter fw;
  fw.Open(file_names);
  uint64_t header_size = 1 << 14;
  uint64_t page_size = 1 << 17;
  fw.SetHeaderSize(header_size);
  fw.SetPageSize(page_size);

  // set shardHeader
  fw.SetShardHeader(std::make_shared<mindrecord::ShardHeader>(header_data));

  // write rawdata
  fw.WriteRawData(rawdatas, bin_data);

  // close file_writer
  fw.Commit();
  std::string filename = "./train_base64.mindrecord01";
  mindrecord::ShardIndexGenerator sg{filename};
  sg.Build();
  sg.WriteToDatabase();
  MS_LOG(INFO) << "Done create index";
  for (const auto &filename : file_names) {
    auto filename_db = filename + ".db";
    remove(common::SafeCStr(filename_db));
    remove(common::SafeCStr(filename));
  }
}

TEST_F(TestShardWriter, TestShardWriterTrialNoFields) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test write trial data"));
  int kMaxNum = 10;
  auto column_list = std::vector<std::string>{"file_name_0"};

  string input_path1 = "./data/mindrecord/testCBGData/data/image_raw_meta.data";
  string input_path3 = "./data/mindrecord/testCBGData/statistics/statistics.txt";
  std::string path_dir = "./data/mindrecord/testCBGData/data/pictures";

  std::vector<std::vector<uint8_t>> bin_data;

  // buffer init
  std::vector<json> json_buffer1;            // store the image_raw_meta.data
  std::vector<json> json_buffer3;            // store the pictures
  std::vector<json> json_buffer4;            // store the statistics data
  std::vector<std::string> image_filenames;  // save all files' path within path_dir

  // read image_raw_meta.data
  LoadData(input_path1, json_buffer1, kMaxNum);
  MS_LOG(INFO) << "Load Meta Data Already.";

  // get files' paths stored in vector<string> image_filenames
  mindrecord::GetAbsoluteFiles(path_dir, image_filenames);  // get all files whose path within path_dir
  MS_LOG(INFO) << "Only process 10 file names:";
  image_filenames.resize(kMaxNum);
  MS_LOG(INFO) << "Load Img Filenames Already.";

  // read pictures
  mindrecord::Img2DataUint8(image_filenames, bin_data);

  // init shardHeader
  mindrecord::ShardHeader header_data;
  MS_LOG(INFO) << "Init ShardHeader Already.";

  // create schema
  json image_schema_json = R"({"name":{"type":"string"},"size":{"type":"int32"},"type":{"type":"string"},
                               "dataset_id":{"type":"int32"},"creation_time":{"type":"string"}})"_json;

  std::shared_ptr<mindrecord::Schema> image_schema = mindrecord::Schema::Build("picture", image_schema_json);
  if (image_schema == nullptr) {
    MS_LOG(ERROR) << "Build image schema failed";
    return;
  }

  // add schema to shardHeader
  int image_schema_id = header_data.AddSchema(image_schema);
  MS_LOG(INFO) << "Init Schema Already.";

  // create/init statistics
  LoadData(input_path3, json_buffer4, 2);
  json static1_json = json_buffer4[0];
  json static2_json = json_buffer4[1];
  MS_LOG(INFO) << "Initial statistics 1 is: " << common::SafeCStr(static1_json.dump());
  MS_LOG(INFO) << "Initial statistics 2 is: " << common::SafeCStr(static2_json.dump());
  std::shared_ptr<mindrecord::Statistics> static1 =
    mindrecord::Statistics::Build(static1_json["description"], static1_json["statistics"]);
  std::shared_ptr<mindrecord::Statistics> static2 =
    mindrecord::Statistics::Build(static2_json["description"], static2_json["statistics"]);
  MS_LOG(INFO) << "Init Statistics Already.";

  // add statistics to shardHeader
  if (static1 == nullptr) {
    MS_LOG(ERROR) << "static1 is nullptr";
    return;
  } else {
    header_data.AddStatistic(static1);
  }
  if (static2 == nullptr) {
    MS_LOG(ERROR) << "static2 is nullptr";
    return;
  } else {
    header_data.AddStatistic(static2);
  }

  // create index field by schema
  std::pair<uint64_t, std::string> index_field1(image_schema_id, "name");
  std::vector<std::pair<uint64_t, std::string>> fields;
  fields.push_back(index_field1);

  // add index to shardHeader

  // merge imgBinaryData(json_buffer3) and imgShardHeader(json_buffer1) to imgBinaryData(json_buffer3)
  for (std::size_t i = 0; i < json_buffer1.size(); i++) {
    json_buffer3.push_back(json{});
  }
  for (std::size_t i = 0; i < json_buffer1.size(); i++) {
    json_buffer3[i] = json_buffer1[i];
  }

  // get json2bson size indicate image size
  json j_test = json_buffer3[0];

  // reference json variable
  std::vector<json> &images = json_buffer3;  // imgBinaryData && imgShardHeader

  // add data
  std::map<std::uint64_t, std::vector<json>> rawdatas;
  rawdatas.insert(pair<uint64_t, vector<json>>(image_schema_id, images));

  // init file_writer
  MS_LOG(INFO) << "Init Writer ...";
  std::vector<std::string> file_names;

  // std::vector<std::string> file_names = {"train_base64.mindrecord01", "train_base64.mindrecord02",
  // "train_base64.mindrecord03"};
  file_names.push_back("./train_base64.mindrecord01");
  file_names.push_back("./train_base64.mindrecord02");
  file_names.push_back("./train_base64.mindrecord03");
  mindrecord::ShardWriter fw;
  fw.Open(file_names);
  uint64_t header_size = 1 << 14;
  uint64_t page_size = 1 << 17;
  fw.SetHeaderSize(header_size);
  fw.SetPageSize(page_size);

  // set shardHeader
  fw.SetShardHeader(std::make_shared<mindrecord::ShardHeader>(header_data));

  // write rawdata
  fw.WriteRawData(rawdatas, bin_data);

  // close file_writer
  fw.Commit();
  MS_LOG(INFO) << "fw ok";
  std::string filename = "./train_base64.mindrecord01";
  mindrecord::ShardIndexGenerator sg{filename};
  sg.Build();
  sg.WriteToDatabase();
  MS_LOG(INFO) << "Done create index";
  for (const auto &filename : file_names) {
    auto filename_db = filename + ".db";
    remove(common::SafeCStr(filename_db));
    remove(common::SafeCStr(filename));
  }
}

TEST_F(TestShardWriter, DataCheck) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test write trial data"));
  int kMaxNum = 10;
  auto column_list = std::vector<std::string>{"file_name_0"};

  string input_path1 = "./data/mindrecord/testCBGData/data/image_raw_meta.data";
  std::string path_dir = "./data/mindrecord/testCBGData/data/pictures";

  std::vector<std::vector<uint8_t>> bin_data;

  // buffer init
  std::vector<json> json_buffer1;            // store the image_raw_meta.data
  std::vector<json> json_buffer3;            // store the pictures
  std::vector<std::string> image_filenames;  // save all files' path within path_dir

  // read image_raw_meta.data
  LoadData(input_path1, json_buffer1, kMaxNum);
  MS_LOG(INFO) << "Load Meta Data Already.";

  // get files' paths stored in vector<string> image_filenames
  mindrecord::GetAbsoluteFiles(path_dir, image_filenames);  // get all files whose path within path_dir
  MS_LOG(INFO) << "Only process 10 file names:";
  image_filenames.resize(kMaxNum);
  MS_LOG(INFO) << "Load Img Filenames Already.";

  // read pictures
  mindrecord::Img2DataUint8(image_filenames, bin_data);

  // init shardHeader
  mindrecord::ShardHeader header_data;
  MS_LOG(INFO) << "Init ShardHeader Already.";

  // create schema
  json image_schema_json = R"({"name":{"type":"string"},"size":{"type":"int32"},"type":{"type":"string"},
                                            "dataset_id":{"type":"int32"},"creation_time":{"type":"string"},
                                            "entity_instances":{"type":"int32","shape":[-1]}})"_json;
  std::shared_ptr<mindrecord::Schema> image_schema = mindrecord::Schema::Build("picture", image_schema_json);
  if (image_schema == nullptr) {
    MS_LOG(ERROR) << "Build image schema failed";
    return;
  }

  // add schema to shardHeader
  int image_schema_id = header_data.AddSchema(image_schema);
  MS_LOG(INFO) << "Init Schema Already.";

  // merge imgBinaryData(json_buffer3) and imgShardHeader(json_buffer1) to imgBinaryData(json_buffer3)
  for (std::size_t i = 0; i < json_buffer1.size(); i++) {
    json_buffer3.push_back(json{});
  }
  for (std::size_t i = 0; i < json_buffer1.size(); i++) {
    json_buffer3[i] = json_buffer1[i];  // add meta_data to json_buffer3's json variable
  }

  // get json2bson size indicate image size
  json j_test = json_buffer3[0];

  // reference json variable
  std::vector<json> &images = json_buffer3;  // imgBinaryData && imgShardHeader

  // add data
  std::map<std::uint64_t, std::vector<json>> rawdatas;
  rawdatas.insert(pair<uint64_t, vector<json>>(image_schema_id, images));

  // init file_writer
  MS_LOG(INFO) << "Init Writer ...";
  std::vector<std::string> file_names;

  // std::vector<std::string> file_names = {"train_base64.mindrecord01", "train_base64.mindrecord02",
  // "train_base64.mindrecord03"};
  file_names.push_back("./train_base64.mindrecord01");
  file_names.push_back("./train_base64.mindrecord02");
  file_names.push_back("./train_base64.mindrecord03");
  mindrecord::ShardWriter fw;
  fw.Open(file_names);
  uint64_t header_size = 1 << 14;
  uint64_t page_size = 1 << 17;
  fw.SetHeaderSize(header_size);
  fw.SetPageSize(page_size);

  // set shardHeader
  fw.SetShardHeader(std::make_shared<mindrecord::ShardHeader>(header_data));

  // write rawdata
  fw.WriteRawData(rawdatas, bin_data);

  // close file_writer
  fw.Commit();
  std::string filename = "./train_base64.mindrecord01";
  // std::string filename = "train_base64.mindrecord01";
  mindrecord::ShardIndexGenerator sg{filename};
  sg.Build();
  sg.WriteToDatabase();
  MS_LOG(INFO) << "Done create index";
  for (const auto &filename : file_names) {
    auto filename_db = filename + ".db";
    remove(common::SafeCStr(filename_db));
    remove(common::SafeCStr(filename));
  }
}

TEST_F(TestShardWriter, AllRawDataWrong) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test write trial data"));
  int kMaxNum = 10;
  auto column_list = std::vector<std::string>{"file_name_0"};

  string input_path1 = "./data/mindrecord/testCBGData/data/image_raw_meta.data";
  std::string path_dir = "./data/mindrecord/testCBGData/data/pictures";

  std::vector<std::vector<uint8_t>> bin_data;

  // buffer init
  std::vector<json> json_buffer1;            // store the image_raw_meta.data
  std::vector<json> json_buffer3;            // store the pictures
  std::vector<std::string> image_filenames;  // save all files' path within path_dir

  // read image_raw_meta.data
  LoadData(input_path1, json_buffer1, kMaxNum);
  MS_LOG(INFO) << "Load Meta Data Already.";

  // get files' paths stored in vector<string> image_filenames
  mindrecord::GetAbsoluteFiles(path_dir, image_filenames);  // get all files whose path within path_dir
  MS_LOG(INFO) << "Only process 10 file names:";
  image_filenames.resize(kMaxNum);
  MS_LOG(INFO) << "Load Img Filenames Already.";

  // read pictures
  mindrecord::Img2DataUint8(image_filenames, bin_data);

  // init shardHeader
  mindrecord::ShardHeader header_data;
  MS_LOG(INFO) << "Init ShardHeader Already.";

  // create schema
  json image_schema_json = R"({"name":{"type":"string"},"size":{"type":"int32"},"type":{"type":"string"},
                                            "id":{"type":"int32"},"creation_time":{"type":"string"},
                                            "entity_instances":{"type":"int32","shape":[-1]}})"_json;
  std::shared_ptr<mindrecord::Schema> image_schema = mindrecord::Schema::Build("picture", image_schema_json);
  if (image_schema == nullptr) {
    MS_LOG(ERROR) << "Build image schema failed";
    return;
  }

  // add schema to shardHeader
  int image_schema_id = header_data.AddSchema(image_schema);
  MS_LOG(INFO) << "Init Schema Already.";

  // merge imgBinaryData(json_buffer3) and imgShardHeader(json_buffer1) to imgBinaryData(json_buffer3)
  for (std::size_t i = 0; i < json_buffer1.size(); i++) {
    json_buffer3.push_back(json{});
  }
  for (std::size_t i = 0; i < json_buffer1.size(); i++) {
    json_buffer3[i] = json_buffer1[i];  // add meta_data to json_buffer3's json variable
  }

  // get json2bson size indicate image size
  json j_test = json_buffer3[0];

  // reference json variable
  std::vector<json> &images = json_buffer3;  // imgBinaryData && imgShardHeader

  // add data
  std::map<std::uint64_t, std::vector<json>> rawdatas;
  rawdatas.insert(pair<uint64_t, vector<json>>(image_schema_id, images));

  // init file_writer
  MS_LOG(INFO) << "Init Writer ...";
  std::vector<std::string> file_names;

  // std::vector<std::string> file_names = {"train_base64.mindrecord01", "train_base64.mindrecord02",
  // "train_base64.mindrecord03"};
  file_names.push_back("./train_base64.mindrecord01");
  file_names.push_back("./train_base64.mindrecord02");
  file_names.push_back("./train_base64.mindrecord03");
  mindrecord::ShardWriter fw;
  fw.Open(file_names);
  uint64_t header_size = 1 << 14;
  uint64_t page_size = 1 << 17;
  fw.SetHeaderSize(header_size);
  fw.SetPageSize(page_size);

  // set shardHeader
  fw.SetShardHeader(std::make_shared<mindrecord::ShardHeader>(header_data));

  // write rawdata
  MSRStatus res = fw.WriteRawData(rawdatas, bin_data);
  ASSERT_EQ(res, SUCCESS);
  for (const auto &filename : file_names) {
    auto filename_db = filename + ".db";
    remove(common::SafeCStr(filename_db));
    remove(common::SafeCStr(filename));
  }
}

TEST_F(TestShardWriter, TestShardReaderStringAndNumberColumnInIndex) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet string and int32 are in index"));

  // load binary data
  std::vector<std::vector<uint8_t>> bin_data;
  std::vector<std::string> filenames;
  ASSERT_NE(-1, mindrecord::GetAbsoluteFiles("./data/mindrecord/testImageNetData/images", filenames));
  ASSERT_NE(-1, mindrecord::Img2DataUint8(filenames, bin_data));

  // init shardHeader
  mindrecord::ShardHeader header_data;
  MS_LOG(INFO) << "Init ShardHeader Already.";

  // create schema
  json anno_schema_json =
    R"({"file_name": {"type": "string"}, "label": {"type": "int32"}, "data":{"type":"bytes"}})"_json;
  std::shared_ptr<mindrecord::Schema> anno_schema = mindrecord::Schema::Build("annotation", anno_schema_json);
  ASSERT_TRUE(anno_schema != nullptr);

  // add schema to shardHeader
  int anno_schema_id = header_data.AddSchema(anno_schema);
  ASSERT_EQ(anno_schema_id, 0);
  MS_LOG(INFO) << "Init Schema Already.";

  // create index
  std::pair<uint64_t, std::string> index_field1(anno_schema_id, "file_name");
  std::pair<uint64_t, std::string> index_field2(anno_schema_id, "label");
  std::vector<std::pair<uint64_t, std::string>> fields;
  fields.push_back(index_field1);
  fields.push_back(index_field2);

  // add index to shardHeader
  ASSERT_EQ(header_data.AddIndexFields(fields), SUCCESS);
  MS_LOG(INFO) << "Init Index Fields Already.";

  // load  meta data
  std::vector<json> annotations;
  LoadDataFromImageNet("./data/mindrecord/testImageNetData/annotation.txt", annotations, 10);

  // add data
  std::map<std::uint64_t, std::vector<json>> rawdatas;
  rawdatas.insert(pair<uint64_t, vector<json>>(anno_schema_id, annotations));
  MS_LOG(INFO) << "Init Images Already.";

  // init file_writer
  std::vector<std::string> file_names;
  for (int i = 1; i <= 4; i++) {
    file_names.emplace_back(std::string("./imagenet.shard0") + std::to_string(i));
    MS_LOG(INFO) << "shard name is: " << common::SafeCStr(file_names[i - 1]);
  }

  mindrecord::ShardWriter fw_init;
  ASSERT_TRUE(fw_init.Open(file_names) == SUCCESS);

  // set shardHeader
  ASSERT_TRUE(fw_init.SetShardHeader(std::make_shared<mindrecord::ShardHeader>(header_data)) == SUCCESS);

  // write raw data
  ASSERT_TRUE(fw_init.WriteRawData(rawdatas, bin_data) == SUCCESS);
  ASSERT_TRUE(fw_init.Commit() == SUCCESS);

  // create the index file
  std::string filename = "./imagenet.shard01";
  mindrecord::ShardIndexGenerator sg{filename};
  sg.Build();
  ASSERT_TRUE(sg.WriteToDatabase() == SUCCESS);
  MS_LOG(INFO) << "Done create index";

  // read the mindrecord file
  filename = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"label", "file_name", "data"};
  ShardReader dataset;
  MSRStatus ret = dataset.Open({filename}, true, 4, column_list);
  ASSERT_EQ(ret, SUCCESS);
  dataset.Launch();

  int count = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto &j : x) {
      count++;
      json resp = std::get<1>(j);
      MS_LOG(INFO) << resp.dump();
      ASSERT_EQ(resp.size(), 2);
      ASSERT_TRUE(resp.size() == 2);
      ASSERT_TRUE(std::string(resp["file_name"].type_name()) == "string");
      ASSERT_TRUE(std::string(resp["label"].type_name()) == "number");
    }
  }
  ASSERT_TRUE(count == 10);
  dataset.Close();

  for (const auto &filename : file_names) {
    auto filename_db = filename + ".db";
    remove(common::SafeCStr(filename_db));
    remove(common::SafeCStr(filename));
  }
}

TEST_F(TestShardWriter, TestShardNoBlob) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test no-blob"));

  // load binary data
  std::vector<std::vector<uint8_t>> bin_data;
  std::vector<std::string> filenames;

  // init shardHeader
  mindrecord::ShardHeader header_data;
  MS_LOG(INFO) << "Init ShardHeader Already.";

  // create schema
  json anno_schema_json = R"({"file_name": {"type": "string"}, "label": {"type": "int32"}})"_json;
  std::shared_ptr<mindrecord::Schema> anno_schema = mindrecord::Schema::Build("annotation", anno_schema_json);
  ASSERT_TRUE(anno_schema != nullptr);

  // add schema to shardHeader
  int anno_schema_id = header_data.AddSchema(anno_schema);
  ASSERT_EQ(anno_schema_id, 0);
  MS_LOG(INFO) << "Init Schema Already.";

  // load  meta data
  std::vector<json> annotations;
  LoadDataFromImageNet("./data/mindrecord/testImageNetData/annotation.txt", annotations, 10);

  // add data
  std::map<std::uint64_t, std::vector<json>> rawdatas;
  rawdatas.insert(pair<uint64_t, vector<json>>(anno_schema_id, annotations));
  MS_LOG(INFO) << "Init labels Already.";

  // init file_writer
  std::vector<std::string> file_names;
  for (int i = 1; i <= 4; i++) {
    file_names.emplace_back(std::string("./imagenet.shard0") + std::to_string(i));
    MS_LOG(INFO) << "shard name is: " << common::SafeCStr(file_names[i - 1]);
  }

  mindrecord::ShardWriter fw_init;
  ASSERT_TRUE(fw_init.Open(file_names) == SUCCESS);

  // set shardHeader
  ASSERT_TRUE(fw_init.SetShardHeader(std::make_shared<mindrecord::ShardHeader>(header_data)) == SUCCESS);

  // write raw data
  ASSERT_TRUE(fw_init.WriteRawData(rawdatas, bin_data) == SUCCESS);
  ASSERT_TRUE(fw_init.Commit() == SUCCESS);

  // create the index file
  std::string filename = "./imagenet.shard01";
  mindrecord::ShardIndexGenerator sg{filename};
  sg.Build();
  ASSERT_TRUE(sg.WriteToDatabase() == SUCCESS);
  MS_LOG(INFO) << "Done create index";

  // read the mindrecord file
  filename = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"label", "file_name"};
  ShardReader dataset;
  MSRStatus ret = dataset.Open({filename}, true, 4, column_list);
  ASSERT_EQ(ret, SUCCESS);
  dataset.Launch();

  int count = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto &j : x) {
      count++;
      json resp = std::get<1>(j);
      ASSERT_TRUE(resp.size() == 2);
      ASSERT_TRUE(std::string(resp["label"].type_name()) == "number");
    }
  }
  ASSERT_TRUE(count == 10);
  dataset.Close();
  for (const auto &filename : file_names) {
    auto filename_db = filename + ".db";
    remove(common::SafeCStr(filename_db));
    remove(common::SafeCStr(filename));
  }
}

TEST_F(TestShardWriter, TestShardReaderStringAndNumberNotColumnInIndex) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet int32 is in index"));

  // load binary data
  std::vector<std::vector<uint8_t>> bin_data;
  std::vector<std::string> filenames;
  ASSERT_NE(-1, mindrecord::GetAbsoluteFiles("./data/mindrecord/testImageNetData/images", filenames));
  ASSERT_NE(-1, mindrecord::Img2DataUint8(filenames, bin_data));

  // init shardHeader
  mindrecord::ShardHeader header_data;
  MS_LOG(INFO) << "Init ShardHeader Already.";

  // create schema
  json anno_schema_json =
    R"({"file_name": {"type": "string"}, "label": {"type": "int32"}, "data":{"type":"bytes"}})"_json;
  std::shared_ptr<mindrecord::Schema> anno_schema = mindrecord::Schema::Build("annotation", anno_schema_json);
  ASSERT_TRUE(anno_schema != nullptr);

  // add schema to shardHeader
  int anno_schema_id = header_data.AddSchema(anno_schema);
  ASSERT_EQ(anno_schema_id, 0);
  MS_LOG(INFO) << "Init Schema Already.";

  // create index
  std::pair<uint64_t, std::string> index_field1(anno_schema_id, "label");
  std::vector<std::pair<uint64_t, std::string>> fields;
  fields.push_back(index_field1);

  // add index to shardHeader
  ASSERT_EQ(header_data.AddIndexFields(fields), SUCCESS);
  MS_LOG(INFO) << "Init Index Fields Already.";

  // load  meta data
  std::vector<json> annotations;
  LoadDataFromImageNet("./data/mindrecord/testImageNetData/annotation.txt", annotations, 10);

  // add data
  std::map<std::uint64_t, std::vector<json>> rawdatas;
  rawdatas.insert(pair<uint64_t, vector<json>>(anno_schema_id, annotations));
  MS_LOG(INFO) << "Init Images Already.";

  // init file_writer
  std::vector<std::string> file_names;
  for (int i = 1; i <= 4; i++) {
    file_names.emplace_back(std::string("./imagenet.shard0") + std::to_string(i));
    MS_LOG(INFO) << "shard name is: " << common::SafeCStr(file_names[i - 1]);
  }

  mindrecord::ShardWriter fw_init;
  ASSERT_TRUE(fw_init.Open(file_names) == SUCCESS);

  // set shardHeader
  ASSERT_TRUE(fw_init.SetShardHeader(std::make_shared<mindrecord::ShardHeader>(header_data)) == SUCCESS);

  // write raw data
  ASSERT_TRUE(fw_init.WriteRawData(rawdatas, bin_data) == SUCCESS);
  ASSERT_TRUE(fw_init.Commit() == SUCCESS);

  // create the index file
  std::string filename = "./imagenet.shard01";
  mindrecord::ShardIndexGenerator sg{filename};
  sg.Build();
  ASSERT_TRUE(sg.WriteToDatabase() == SUCCESS);
  MS_LOG(INFO) << "Done create index";

  // read the mindrecord file
  filename = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"label", "data"};
  ShardReader dataset;
  MSRStatus ret = dataset.Open({filename}, true, 4, column_list);
  ASSERT_EQ(ret, SUCCESS);
  dataset.Launch();

  int count = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto &j : x) {
      count++;
      json resp = std::get<1>(j);
      ASSERT_TRUE(resp.size() == 1);
      ASSERT_TRUE(std::string(resp["label"].type_name()) == "number");
    }
  }
  ASSERT_TRUE(count == 10);
  dataset.Close();
  for (const auto &filename : file_names) {
    auto filename_db = filename + ".db";
    remove(common::SafeCStr(filename_db));
    remove(common::SafeCStr(filename));
  }
}

TEST_F(TestShardWriter, TestShardWriter10Sample40Shard) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test write imageNet int32 of sample less than num of shards"));

  int num_sample = 10;
  int num_shard = 40;

  // load binary data
  std::vector<std::vector<uint8_t>> bin_data;
  std::vector<std::string> filenames;
  ASSERT_NE(-1, mindrecord::GetAbsoluteFiles("./data/mindrecord/testImageNetData/images", filenames));

  mindrecord::Img2DataUint8(filenames, bin_data);

  // init shardHeader
  mindrecord::ShardHeader header_data;
  MS_LOG(INFO) << "Init ShardHeader Already.";

  // create schema
  json anno_schema_json =
    R"({"file_name": {"type": "string"}, "label": {"type": "int32"}, "data":{"type":"bytes"}})"_json;
  std::shared_ptr<mindrecord::Schema> anno_schema = mindrecord::Schema::Build("annotation", anno_schema_json);
  if (anno_schema == nullptr) {
    MS_LOG(ERROR) << "Build annotation schema failed";
    return;
  }

  // add schema to shardHeader
  int anno_schema_id = header_data.AddSchema(anno_schema);
  MS_LOG(INFO) << "Init Schema Already.";

  // create index
  std::pair<uint64_t, std::string> index_field1(anno_schema_id, "file_name");
  std::pair<uint64_t, std::string> index_field2(anno_schema_id, "label");
  std::vector<std::pair<uint64_t, std::string>> fields;
  fields.push_back(index_field1);
  fields.push_back(index_field2);

  // add index to shardHeader
  header_data.AddIndexFields(fields);
  MS_LOG(INFO) << "Init Index Fields Already.";

  // load  meta data
  std::vector<json> annotations;
  LoadDataFromImageNet("./data/mindrecord/testImageNetData/annotation.txt", annotations, num_sample);

  // add data
  std::map<std::uint64_t, std::vector<json>> rawdatas;
  rawdatas.insert(pair<uint64_t, vector<json>>(anno_schema_id, annotations));
  MS_LOG(INFO) << "Init Images Already.";

  // init file_writer
  std::vector<std::string> file_names;
  for (int i = 1; i <= num_shard; i++) {
    file_names.emplace_back(std::string("./TenSampleFortyShard.shard0") + std::to_string(i));
    MS_LOG(INFO) << "shard name is: " << common::SafeCStr(file_names[i - 1]);
  }

  MS_LOG(INFO) << "Init Output Files Already.";
  {
    mindrecord::ShardWriter fw_init;
    fw_init.Open(file_names);
    // set shardHeader
    fw_init.SetShardHeader(std::make_shared<mindrecord::ShardHeader>(header_data));
    // close file_writer
    fw_init.Commit();
  }
  std::string filename = "./TenSampleFortyShard.shard01";
  {
    MS_LOG(INFO) << "=============== images " << bin_data.size() << " ============================";
    mindrecord::ShardWriter fw;
    fw.OpenForAppend(filename);
    bin_data = std::vector<std::vector<uint8_t>>(bin_data.begin(), bin_data.begin() + num_sample);
    fw.WriteRawData(rawdatas, bin_data);
    fw.Commit();
  }

  mindrecord::ShardIndexGenerator sg{filename};
  sg.Build();
  sg.WriteToDatabase();
  MS_LOG(INFO) << "Done create index";

  filename = "./TenSampleFortyShard.shard01";
  ShardReader dataset;
  MSRStatus ret = dataset.Open({filename}, true, 4);
  ASSERT_EQ(ret, SUCCESS);
  dataset.Launch();

  int count = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto &j : x) {
      MS_LOG(INFO) << "item size: " << std::get<0>(j).size();
      for (auto &item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << common::SafeCStr(item.key()) << ", value: " << common::SafeCStr(item.value().dump());
      }
    }
    count++;
  }
  ASSERT_TRUE(count == 10);
  dataset.Close();
  for (const auto &filename : file_names) {
    auto filename_db = filename + ".db";
    remove(common::SafeCStr(filename_db));
    remove(common::SafeCStr(filename));
  }
}

TEST_F(TestShardWriter, TestWriteOpenFileName) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test write imageNet with error filename contain invalid utf-8 data"));
  mindrecord::ShardHeader header_data;

  // create schema
  json anno_schema_json = R"({"file_name": {"type": "string"}, "label": {"type": "int32"}})"_json;
  std::shared_ptr<mindrecord::Schema> anno_schema = mindrecord::Schema::Build("annotation", anno_schema_json);
  if (anno_schema == nullptr) {
    MS_LOG(ERROR) << "Build annotation schema failed";
    return;
  }

  // add schema to shardHeader
  int anno_schema_id = header_data.AddSchema(anno_schema);
  MS_LOG(INFO) << "Init Schema Already.";

  // create index
  std::pair<uint64_t, std::string> index_field1(anno_schema_id, "file_name");
  std::pair<uint64_t, std::string> index_field2(anno_schema_id, "label");
  std::vector<std::pair<uint64_t, std::string>> fields;
  fields.push_back(index_field1);
  fields.push_back(index_field2);

  // add index to shardHeader
  header_data.AddIndexFields(fields);
  MS_LOG(INFO) << "Init Index Fields Already.";

  string filename = "./ä\xA9ü";
  MS_LOG(INFO) << "filename: " << common::SafeCStr(filename);

  std::vector<std::string> file_names;
  for (int i = 1; i <= 4; i++) {
    // file_names.emplace_back(std::string(filename).substr(0, std::string(filename).length()-1) + std::to_string(i));
    file_names.emplace_back(std::string(filename) + "0" + std::to_string(i));
    MS_LOG(INFO) << "shard name is: " << common::SafeCStr(file_names[i - 1]);
  }

  MS_LOG(INFO) << "Init Output Files Already.";
  {
    mindrecord::ShardWriter fw_init;
    fw_init.Open(file_names);
    // set shardHeader
    fw_init.SetShardHeader(std::make_shared<mindrecord::ShardHeader>(header_data));
    // close file_writer
    fw_init.Commit();
  }
}

TEST_F(TestShardWriter, TestOpenForAppend) {
  MS_LOG(INFO) << "start ---- TestOpenForAppend\n";
  string filename = "./";
  ShardWriterImageNetOpenForAppend(filename);

  string filename1 = "./▒AppendSample.shard01";
  ShardWriterImageNetOpenForAppend(filename1);
  string filename2 = "./ä\xA9ü";

  ShardWriterImageNetOpenForAppend(filename2);

  MS_LOG(INFO) << "end ---- TestOpenForAppend\n";
  for (int i = 1; i <= 4; i++) {
    string filename = std::string("./OpenForAppendSample.shard0") + std::to_string(i);
    string db_name = std::string("./OpenForAppendSample.shard0") + std::to_string(i) + ".db";
    remove(common::SafeCStr(filename));
    remove(common::SafeCStr(db_name));
  }
}

}  // namespace mindrecord
}  // namespace mindspore
