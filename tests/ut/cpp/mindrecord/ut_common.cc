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

#include "ut_common.h"

namespace mindspore {
namespace mindrecord {
namespace UT {
#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

void Common::SetUp() {}

void Common::TearDown() {}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
}  // namespace UT

const std::string FormatInfo(const std::string &message, uint32_t message_total_length) {
  // if the message is larger than message_total_length
  std::string part_message = "";
  if (message_total_length < message.length()) {
    part_message = message.substr(0, message_total_length);
  } else {
    part_message = message;
  }
  int padding_length = static_cast<int>(message_total_length - part_message.length());
  std::string left_padding(static_cast<uint64_t>(ceil(padding_length / 2.0)), '=');
  std::string right_padding(static_cast<uint64_t>(floor(padding_length / 2.0)), '=');
  return left_padding + part_message + right_padding;
}

void LoadData(const std::string &directory, std::vector<json> &json_buffer, const int max_num) {
  int count = 0;
  string input_path = directory;
  ifstream infile(input_path);
  if (!infile.is_open()) {
    MS_LOG(ERROR) << "can not open the file ";
    return;
  }
  string temp;
  while (getline(infile, temp) && count != max_num) {
    count++;
    json j = json::parse(temp);
    json_buffer.push_back(j);
  }
  infile.close();
}

void LoadDataFromImageNet(const std::string &directory, std::vector<json> &json_buffer, const int max_num) {
  int count = 0;
  string input_path = directory;
  ifstream infile(input_path);
  if (!infile.is_open()) {
    MS_LOG(ERROR) << "can not open the file ";
    return;
  }
  string temp;
  string filename;
  string label;
  json j;
  while (getline(infile, temp) && count != max_num) {
    count++;
    std::size_t pos = temp.find(",", 0);
    if (pos != std::string::npos) {
      j["file_name"] = temp.substr(0, pos);
      j["label"] = atoi(common::SafeCStr(temp.substr(pos + 1, temp.length())));
      json_buffer.push_back(j);
    }
  }
  infile.close();
}

int Img2DataUint8(const std::vector<std::string> &img_absolute_path, std::vector<std::vector<uint8_t>> &bin_data) {
  for (auto &file : img_absolute_path) {
    // read image file
    std::ifstream in(common::SafeCStr(file), std::ios::in | std::ios::binary | std::ios::ate);
    if (!in) {
      MS_LOG(ERROR) << common::SafeCStr(file) << " is not a directory or not exist!";
      return -1;
    }

    // get the file size
    uint64_t size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> file_data(size);
    in.read(reinterpret_cast<char *>(&file_data[0]), size);
    in.close();
    bin_data.push_back(file_data);
  }
  return 0;
}

int GetAbsoluteFiles(std::string directory, std::vector<std::string> &files_absolute_path) {
  DIR *dir = opendir(common::SafeCStr(directory));
  if (dir == nullptr) {
    MS_LOG(ERROR) << common::SafeCStr(directory) << " is not a directory or not exist!";
    return -1;
  }
  struct dirent *d_ent = nullptr;
  char dot[3] = ".";
  char dotdot[6] = "..";
  while ((d_ent = readdir(dir)) != nullptr) {
    if ((strcmp(d_ent->d_name, dot) != 0) && (strcmp(d_ent->d_name, dotdot) != 0)) {
      if (d_ent->d_type == DT_DIR) {
        std::string new_directory = directory + std::string("/") + std::string(d_ent->d_name);
        if (directory[directory.length() - 1] == '/') {
          new_directory = directory + string(d_ent->d_name);
        }
        if (-1 == GetAbsoluteFiles(new_directory, files_absolute_path)) {
          closedir(dir);
          return -1;
        }
      } else {
        std::string absolute_path = directory + std::string("/") + std::string(d_ent->d_name);
        if (directory[directory.length() - 1] == '/') {
          absolute_path = directory + std::string(d_ent->d_name);
        }
        files_absolute_path.push_back(absolute_path);
      }
    }
  }
  closedir(dir);
  return 0;
}

void ShardWriterImageNet() {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Write imageNet"));

  // load binary data
  std::vector<std::vector<uint8_t>> bin_data;
  std::vector<std::string> filenames;
  if (-1 == mindrecord::GetAbsoluteFiles("./data/mindrecord/testImageNetData/images", filenames)) {
    MS_LOG(INFO) << "-- ATTN -- Missed data directory. Skip this case. -----------------";
    return;
  }
  mindrecord::Img2DataUint8(filenames, bin_data);

  // init shardHeader
  ShardHeader header_data;
  MS_LOG(INFO) << "Init ShardHeader Already.";

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
  // load  meta data
  std::vector<json> annotations;
  LoadDataFromImageNet("./data/mindrecord/testImageNetData/annotation.txt", annotations, 10);

  // add data
  std::map<std::uint64_t, std::vector<json>> rawdatas;
  rawdatas.insert(pair<uint64_t, vector<json>>(anno_schema_id, annotations));
  MS_LOG(INFO) << "Init Images Already.";

  // init file_writer
  std::vector<std::string> file_names;
  int file_count = 4;
  for (int i = 1; i <= file_count; i++) {
    file_names.emplace_back(std::string("./imagenet.shard0") + std::to_string(i));
    MS_LOG(INFO) << "shard name is: " << common::SafeCStr(file_names[i - 1]);
  }

  MS_LOG(INFO) << "Init Output Files Already.";
  {
    ShardWriter fw_init;
    fw_init.Open(file_names);

    // set shardHeader
    fw_init.SetShardHeader(std::make_shared<mindrecord::ShardHeader>(header_data));

    // close file_writer
    fw_init.Commit();
  }
  std::string filename = "./imagenet.shard01";
  {
    MS_LOG(INFO) << "=============== images " << bin_data.size() << " ============================";
    mindrecord::ShardWriter fw;
    fw.OpenForAppend(filename);
    fw.WriteRawData(rawdatas, bin_data);
    fw.Commit();
  }
  mindrecord::ShardIndexGenerator sg{filename};
  sg.Build();
  sg.WriteToDatabase();

  MS_LOG(INFO) << "Done create index";
}

void ShardWriterImageNetOneSample() {
  // load binary data
  std::vector<std::vector<uint8_t>> bin_data;
  std::vector<std::string> filenames;
  if (-1 == mindrecord::GetAbsoluteFiles("./data/mindrecord/testImageNetData/images", filenames)) {
    MS_LOG(INFO) << "-- ATTN -- Missed data directory. Skip this case. -----------------";
    return;
  }
  mindrecord::Img2DataUint8(filenames, bin_data);

  // init shardHeader
  mindrecord::ShardHeader header_data;
  MS_LOG(INFO) << "Init ShardHeader Already.";

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

  // load  meta data
  std::vector<json> annotations;
  LoadDataFromImageNet("./data/mindrecord/testImageNetData/annotation.txt", annotations, 1);

  // add data
  std::map<std::uint64_t, std::vector<json>> rawdatas;
  rawdatas.insert(pair<uint64_t, vector<json>>(anno_schema_id, annotations));
  MS_LOG(INFO) << "Init Images Already.";

  // init file_writer
  std::vector<std::string> file_names;
  for (int i = 1; i <= 4; i++) {
    file_names.emplace_back(std::string("./OneSample.shard0") + std::to_string(i));
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

  std::string filename = "./OneSample.shard01";
  {
    MS_LOG(INFO) << "=============== images " << bin_data.size() << " ============================";
    mindrecord::ShardWriter fw;
    fw.OpenForAppend(filename);
    bin_data = std::vector<std::vector<uint8_t>>(bin_data.begin(), bin_data.begin() + 1);
    fw.WriteRawData(rawdatas, bin_data);
    fw.Commit();
  }

  mindrecord::ShardIndexGenerator sg{filename};
  sg.Build();
  sg.WriteToDatabase();
  MS_LOG(INFO) << "Done create index";
}

void ShardWriterImageNetOpenForAppend(string filename) {
  for (int i = 1; i <= 4; i++) {
    string filename = std::string("./OpenForAppendSample.shard0") + std::to_string(i);
    string db_name = std::string("./OpenForAppendSample.shard0") + std::to_string(i) + ".db";
    remove(common::SafeCStr(filename));
    remove(common::SafeCStr(db_name));
  }

  // load binary data
  std::vector<std::vector<uint8_t>> bin_data;
  std::vector<std::string> filenames;
  if (-1 == mindrecord::GetAbsoluteFiles("./data/mindrecord/testImageNetData/images", filenames)) {
    MS_LOG(INFO) << "-- ATTN -- Missed data directory. Skip this case. -----------------";
    return;
  }
  mindrecord::Img2DataUint8(filenames, bin_data);

  // init shardHeader
  mindrecord::ShardHeader header_data;
  MS_LOG(INFO) << "Init ShardHeader Already.";

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

  // load  meta data
  std::vector<json> annotations;
  LoadDataFromImageNet("./data/mindrecord/testImageNetData/annotation.txt", annotations, 1);

  // add data
  std::map<std::uint64_t, std::vector<json>> rawdatas;
  rawdatas.insert(pair<uint64_t, vector<json>>(anno_schema_id, annotations));
  MS_LOG(INFO) << "Init Images Already.";

  // init file_writer
  std::vector<std::string> file_names;
  for (int i = 1; i <= 4; i++) {
    file_names.emplace_back(std::string("./OpenForAppendSample.shard0") + std::to_string(i));
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
  {
    MS_LOG(INFO) << "=============== images " << bin_data.size() << " ============================";
    mindrecord::ShardWriter fw;
    auto status = fw.OpenForAppend(filename);
    if (status.IsError()) {
      return;
    }

    bin_data = std::vector<std::vector<uint8_t>>(bin_data.begin(), bin_data.begin() + 1);
    fw.WriteRawData(rawdatas, bin_data);
    fw.Commit();
  }

  ShardIndexGenerator sg{filename};
  sg.Build();
  sg.WriteToDatabase();
  MS_LOG(INFO) << "Done create index";
}


}  // namespace mindrecord
}  // namespace mindspore
