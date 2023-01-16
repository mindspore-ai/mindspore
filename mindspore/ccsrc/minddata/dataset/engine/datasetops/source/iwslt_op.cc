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

#include "minddata/dataset/engine/datasetops/source/iwslt_op.h"

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/util/status.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
IWSLTOp::IWSLTOp(int32_t num_workers, int64_t num_samples, int32_t worker_connector_size, int32_t op_connector_size,
                 bool shuffle_files, int32_t num_devices, int32_t device_id, std::unique_ptr<DataSchema> data_schema,
                 IWSLTType type, const std::string &dataset_dir, const std::string &usage,
                 const std::vector<std::string> &language_pair, const std::string &valid_set,
                 const std::string &test_set)
    : NonMappableLeafOp(num_workers, worker_connector_size, num_samples, op_connector_size, shuffle_files, num_devices,
                        device_id),
      iwslt_type_(type),
      data_schema_(std::move(data_schema)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      language_pair_(language_pair),
      valid_set_(valid_set),
      test_set_(test_set) {}

Status IWSLTOp::Init() {
  RETURN_IF_NOT_OK(this->GetFiles());
  RETURN_IF_NOT_OK(filename_index_->insert(src_target_file_list_));

  int32_t safe_queue_size = static_cast<int32_t>(std::ceil(src_target_file_list_.size() / num_workers_) + 1);
  io_block_queues_.Init(num_workers_, safe_queue_size);

  jagged_rows_connector_ = std::make_unique<JaggedConnector>(num_workers_, 1, worker_connector_size_);
  return Status::OK();
}

std::vector<std::string> IWSLTOp::Split(const std::string &s, const std::string &delim) {
  std::vector<std::string> res;
  std::string::size_type pos1 = 0;
  std::string::size_type pos2 = s.find(delim);
  while (std::string::npos != pos2) {
    res.push_back(s.substr(pos1, pos2 - pos1));

    pos1 = pos2 + delim.size();
    pos2 = s.find(delim, pos1);
  }
  if (pos1 != s.length()) {
    res.push_back(s.substr(pos1));
  }
  return res;
}

Status IWSLTOp::Trim(std::string *text, const std::string &character) {
  RETURN_UNEXPECTED_IF_NULL(text);
  CHECK_FAIL_RETURN_UNEXPECTED(!text->empty(), "Invalid file, read an empty line.");
  (void)text->erase(0, text->find_first_not_of(character));
  (void)text->erase(text->find_last_not_of(character) + 1);
  return Status::OK();
}

Status IWSLTOp::LoadTensor(const std::string &line, TensorRow *out_row, size_t index) {
  RETURN_UNEXPECTED_IF_NULL(out_row);
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(Tensor::CreateScalar(line, &tensor));
  (*out_row)[index] = std::move(tensor);
  return Status::OK();
}

Status IWSLTOp::LoadFile(const std::string &file, int64_t start_offset, int64_t end_offset, int32_t worker_id) {
  std::ifstream handle(file);
  std::string line;
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open " + DatasetName() + " file: " + file);
  }

  int64_t rows_total = 0;
  while (getline(handle, line)) {
    if (line.empty()) {
      continue;
    }
    // If read to the end offset of this file, break.
    if (rows_total >= end_offset) {
      break;
    }
    // Skip line before start offset.
    if (rows_total < start_offset) {
      rows_total++;
      continue;
    }

    const int kColumnSize = 2;
    TensorRow tRow(kColumnSize, nullptr);
    tRow.setPath({file, file});

    // Remove the newline character.
    RETURN_IF_NOT_OK(Trim(&line, "\n"));
    RETURN_IF_NOT_OK(Trim(&line, "\r"));
    std::vector<std::string> sentence_list = Split(line, "#*$");
    if (!sentence_list.empty() && sentence_list.size() == kColumnSize) {
      RETURN_IF_NOT_OK(LoadTensor(sentence_list[0], &tRow, 0));
      RETURN_IF_NOT_OK(LoadTensor(sentence_list[1], &tRow, 1));
      RETURN_IF_NOT_OK(jagged_rows_connector_->Add(worker_id, std::move(tRow)));
      rows_total++;
    }
  }
  handle.close();
  return Status::OK();
}

Status IWSLTOp::FillIOBlockQueue(const std::vector<int64_t> &i_keys) {
  int32_t queue_index = 0;
  int64_t pre_count = 0;
  int64_t start_offset = 0;
  int64_t end_offset = 0;
  bool finish = false;
  while (!finish) {
    std::vector<std::pair<std::string, int64_t>> file_index;
    if (!i_keys.empty()) {
      for (auto it = i_keys.begin(); it != i_keys.end(); ++it) {
        {
          if (!GetLoadIoBlockQueue()) {
            break;
          }
        }
        file_index.emplace_back(std::pair<std::string, int64_t>((*filename_index_)[*it], *it));
      }
    } else {
      for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
        {
          if (!GetLoadIoBlockQueue()) {
            break;
          }
        }
        file_index.emplace_back(std::pair<std::string, int64_t>(it.value(), it.key()));
      }
    }
    for (auto file_info : file_index) {
      if (NeedPushFileToBlockQueue(file_info.first, &start_offset, &end_offset, pre_count)) {
        auto ioBlock =
          std::make_unique<FilenameBlock>(file_info.second, start_offset, end_offset, IOBlock::kDeIoBlockNone);
        RETURN_IF_NOT_OK(PushIoBlockQueue(queue_index, std::move(ioBlock)));
        queue_index = (queue_index + 1) % num_workers_;
      }

      pre_count += filename_numrows_[file_info.first];
    }

    if (pre_count < (static_cast<int64_t>(device_id_) + 1) * num_rows_per_shard_) {
      finish = false;
    } else {
      finish = true;
    }
  }

  RETURN_IF_NOT_OK(PostEndOfEpoch(queue_index));
  return Status::OK();
}

void IWSLTOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op.
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff.
    out << "\nSample count: " << total_rows_ << "\nDevice id: " << device_id_ << "\nNumber of devices: " << num_devices_
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\nIWSLT files list:\n";
    for (int i = 0; i < src_target_file_list_.size(); ++i) {
      out << " " << src_target_file_list_[i];
    }
    out << "\nData Schema:\n";
    out << *data_schema_ << "\n\n";
  }
}

int64_t IWSLTOp::CountFileRows(const std::string &file) {
  std::ifstream handle(file);
  if (!handle.is_open()) {
    MS_LOG(ERROR) << "Invalid file, failed to open file: " << file;
    return 0;
  }

  std::string line;
  int64_t count = 0;
  while (getline(handle, line)) {
    if (!line.empty()) {
      count++;
    }
  }
  handle.close();
  return count;
}

Status IWSLTOp::CalculateNumRowsPerShard() {
  for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
    int64_t count = CountFileRows(it.value());
    filename_numrows_[it.value()] = count;
    num_rows_ += count;
  }
  if (num_rows_ == 0) {
    std::stringstream ss;
    for (int i = 0; i < src_target_file_list_.size(); ++i) {
      ss << " " << src_target_file_list_[i];
    }
    std::string file_list = ss.str();
    RETURN_STATUS_UNEXPECTED("Invalid data, " + DatasetName(true) +
                             "Dataset API can't read the data file (interface mismatch or no data found). Check " +
                             DatasetName() + ": " + file_list);
  }

  num_rows_per_shard_ = static_cast<int64_t>(std::ceil(num_rows_ * 1.0 / num_devices_));
  MS_LOG(DEBUG) << "Number rows per shard is " << num_rows_per_shard_;
  return Status::OK();
}

Status IWSLTOp::ComputeColMap() {
  // Set the column name mapping (base class field).
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->Column(i).Name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

Status IWSLTOp::CountTotalRows(IWSLTType type, const std::string &dataset_dir, const std::string &usage,
                               const std::vector<std::string> &language_pair, const std::string &valid_set,
                               const std::string &test_set, int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  int32_t num_workers = GlobalContext::config_manager()->num_parallel_workers();
  int32_t connector_que_size = GlobalContext::config_manager()->op_connector_size();
  int32_t worker_connector_size = GlobalContext::config_manager()->worker_connector_size();
  const int32_t shard_id = 0;
  const int32_t num_shards = 1;
  const int64_t num_samples = 0;
  bool shuffle_files = false;
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();

  // Create and initialize.
  std::shared_ptr<IWSLTOp> op = std::make_shared<IWSLTOp>(
    num_workers, num_samples, worker_connector_size, connector_que_size, shuffle_files, num_shards, shard_id,
    std::move(schema), type, dataset_dir, usage, language_pair, valid_set, test_set);
  RETURN_IF_NOT_OK(op->Init());

  *count = 0;
  std::vector<std::string> file_list = op->FileNames();
  for (auto file : file_list) {
    *count += op->CountFileRows(file);
  }
  return Status::OK();
}

Status LoadXmlDocument(XMLDocument *xml_document, const std::string &file_path, XMLElement **doc) {
  RETURN_UNEXPECTED_IF_NULL(xml_document);
  XMLError e = xml_document->LoadFile(common::SafeCStr(file_path));
  if (e != XMLError::XML_SUCCESS) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to load xml file: " + file_path);
  }
  XMLElement *root = xml_document->RootElement();
  if (root == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid data, failed to load root element for xml file.");
  }
  XMLElement *firstChild = root->FirstChildElement();
  if (firstChild == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid data, no first child found in " + file_path);
  }
  *doc = firstChild->FirstChildElement("doc");
  if (*doc == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid data, no doc found in " + file_path);
  }
  return Status::OK();
}

Status IWSLTOp::CleanXmlFile(const std::string &src_file_path, const std::string &target_file_path,
                             const std::string &new_file_path) {
  XMLDocument xml_document1, xml_document2;
  XMLElement *src_doc = nullptr;
  XMLElement *target_doc = nullptr;

  RETURN_IF_NOT_OK(LoadXmlDocument(&xml_document1, src_file_path, &src_doc));
  RETURN_IF_NOT_OK(LoadXmlDocument(&xml_document2, target_file_path, &target_doc));
  std::string src_content, target_content;
  std::ofstream new_file(new_file_path);
  CHECK_FAIL_RETURN_UNEXPECTED(new_file.is_open(), "Invalid file, failed to open file: " + new_file_path);

  while (src_doc != nullptr && target_doc != nullptr) {
    XMLElement *src_seg = src_doc->FirstChildElement("seg");
    XMLElement *target_seg = target_doc->FirstChildElement("seg");
    while (src_seg != nullptr && target_seg != nullptr) {
      src_content = src_seg->GetText();
      target_content = target_seg->GetText();
      RETURN_IF_NOT_OK(Trim(&src_content, " "));
      RETURN_IF_NOT_OK(Trim(&target_content, " "));
      src_seg = src_seg->NextSiblingElement();
      target_seg = target_seg->NextSiblingElement();
      new_file << (src_content + "#*$" + target_content + "\n");
    }
    src_doc = src_doc->NextSiblingElement();
    target_doc = target_doc->NextSiblingElement();
  }

  new_file.close();
  return Status::OK();
}

bool IWSLTOp::IsContainTags(const std::string &content) {
  std::vector<std::string> xml_tags = {"<url",        "<keywords", "<talkid",  "<description", "<reviewer",
                                       "<translator", "<title",    "<speaker", "<doc",         "</doc"};
  int i = 0;
  int size = xml_tags.size();
  while (i < size) {
    if (content.find(xml_tags[i]) != std::string::npos) {
      return true;
    }
    i++;
  }
  return false;
}

Status IWSLTOp::CleanTagFile(const std::string &src_file_path, const std::string &target_file_path,
                             const std::string &new_file_path) {
  std::ifstream src_handle(src_file_path);
  std::ifstream target_handle(target_file_path);

  std::ofstream new_file(new_file_path, std::ios::trunc);
  std::string src_content, target_content;
  while (getline(src_handle, src_content)) {
    while (getline(target_handle, target_content)) {
      if (!IsContainTags(src_content) && !IsContainTags(target_content)) {
        RETURN_IF_NOT_OK(Trim(&src_content, " "));
        RETURN_IF_NOT_OK(Trim(&target_content, " "));
        new_file << (src_content + "#*$" + target_content + "\n");
      }
      break;
    }
  }
  new_file.close();
  src_handle.close();
  target_handle.close();
  return Status::OK();
}

Status IWSLTOp::GenerateNewFile(const std::vector<std::string> &src_file_list,
                                const std::vector<std::string> &target_file_list,
                                std::vector<std::string> *src_target_file_list) {
  RETURN_UNEXPECTED_IF_NULL(src_target_file_list);
  std::string::size_type position;
  std::string new_path;
  std::string src_path, target_path;
  for (int i = 0; i < src_file_list.size(); i++) {
    src_path = src_file_list[i];
    target_path = target_file_list[i];

    // Add new train file name.
    position = src_path.find(".tags");
    if (position != std::string::npos) {
      new_path = src_path;
      const int kTagSize = 5;
      const int kSuffixSize = 3;
      new_path = new_path.replace(new_path.find(".tags"), kTagSize, "");
      new_path = new_path.substr(0, new_path.length() - kSuffixSize);

      // Write data to the new file path.
      RETURN_IF_NOT_OK(CleanTagFile(src_path, target_path, new_path));
      src_target_file_list->push_back(new_path);
    } else {
      // Add new valid or test file name.
      // Delete suffix.
      const int kSuffixXMLSize = 7;
      new_path = src_path;
      new_path = new_path.substr(0, new_path.length() - kSuffixXMLSize);
      // Write data to the new file path.
      RETURN_IF_NOT_OK(CleanXmlFile(src_path, target_path, new_path));
      src_target_file_list->push_back(new_path);
    }
  }
  return Status::OK();
}

std::string IWSLTOp::GenerateIWSLT2016TagsFileName(Path dir, const std::string &src_language,
                                                   const std::string &target_language, const std::string &suffix) {
  Path src_language_path(src_language);
  Path target_language_path(target_language);
  Path sub_dir(src_language + "-" + target_language);
  Path file_name("train.tags." + src_language + "-" + target_language + "." + suffix);
  Path file_path = dir / "texts" / src_language_path / target_language_path / sub_dir / file_name;
  return file_path.ToString();
}

std::string IWSLTOp::GenerateIWSLT2016XMLFileName(Path dir, const std::string &src_language,
                                                  const std::string &target_language, const std::string &set_type,
                                                  const std::string &suffix) {
  Path src_language_path(src_language);
  Path target_language_path(target_language);
  Path sub_dir(src_language + "-" + target_language);
  Path file_name("IWSLT16.TED." + set_type + "." + src_language + "-" + target_language + "." + suffix + ".xml");
  Path file_path = dir / "texts" / src_language_path / target_language_path / sub_dir / file_name;
  return file_path.ToString();
}

std::string IWSLTOp::GenerateIWSLT2017TagsFileName(Path dir, const std::string &src_language,
                                                   const std::string &target_language, const std::string &suffix) {
  Path sub_const_dir("texts");
  Path sub_src_language_dir("DeEnItNlRo");
  Path sub_tgt_language_dir("DeEnItNlRo");
  Path sub_src_tgt_dir("DeEnItNlRo-DeEnItNlRo");
  Path file_name("train.tags." + src_language + "-" + target_language + "." + suffix);
  Path file_path = dir / sub_const_dir / sub_src_language_dir / sub_tgt_language_dir / sub_src_tgt_dir / file_name;
  return file_path.ToString();
}

std::string IWSLTOp::GenerateIWSLT2017XMLFileName(Path dir, const std::string &src_language,
                                                  const std::string &target_language, const std::string &set_type,
                                                  const std::string &suffix) {
  Path sub_const_dir("texts");
  Path sub_src_language_dir("DeEnItNlRo");
  Path sub_tgt_language_dir("DeEnItNlRo");
  Path sub_src_tgt_dir("DeEnItNlRo-DeEnItNlRo");
  Path file_name("IWSLT17.TED." + set_type + "." + src_language + "-" + target_language + "." + suffix + ".xml");
  Path file_path = dir / sub_const_dir / sub_src_language_dir / sub_tgt_language_dir / sub_src_tgt_dir / file_name;
  return file_path.ToString();
}

Status IWSLTOp::GetFiles() {
  std::vector<std::string> src_path_list;
  std::vector<std::string> target_path_list;
  auto real_dataset_dir = FileUtils::GetRealPath(dataset_dir_.c_str());
  CHECK_FAIL_RETURN_UNEXPECTED(real_dataset_dir.has_value(), "Get real path failed: " + dataset_dir_);
  Path root_dir(real_dataset_dir.value());

  if (iwslt_type_ == kIWSLT2016) {
    if (usage_ == "train" || usage_ == "all") {
      src_path_list.push_back(
        GenerateIWSLT2016TagsFileName(root_dir, language_pair_[0], language_pair_[1], language_pair_[0]));
      target_path_list.push_back(
        GenerateIWSLT2016TagsFileName(root_dir, language_pair_[0], language_pair_[1], language_pair_[1]));
    }
    if (usage_ == "valid" || usage_ == "all") {
      src_path_list.push_back(
        GenerateIWSLT2016XMLFileName(root_dir, language_pair_[0], language_pair_[1], valid_set_, language_pair_[0]));
      target_path_list.push_back(
        GenerateIWSLT2016XMLFileName(root_dir, language_pair_[0], language_pair_[1], valid_set_, language_pair_[1]));
    }
    if (usage_ == "test" || usage_ == "all") {
      src_path_list.push_back(
        GenerateIWSLT2016XMLFileName(root_dir, language_pair_[0], language_pair_[1], test_set_, language_pair_[0]));
      target_path_list.push_back(
        GenerateIWSLT2016XMLFileName(root_dir, language_pair_[0], language_pair_[1], test_set_, language_pair_[1]));
    }
  } else {
    if (usage_ == "train" || usage_ == "all") {
      src_path_list.push_back(
        GenerateIWSLT2017TagsFileName(root_dir, language_pair_[0], language_pair_[1], language_pair_[0]));
      target_path_list.push_back(
        GenerateIWSLT2017TagsFileName(root_dir, language_pair_[0], language_pair_[1], language_pair_[1]));
    }
    if (usage_ == "valid" || usage_ == "all") {
      src_path_list.push_back(
        GenerateIWSLT2017XMLFileName(root_dir, language_pair_[0], language_pair_[1], valid_set_, language_pair_[0]));
      target_path_list.push_back(
        GenerateIWSLT2017XMLFileName(root_dir, language_pair_[0], language_pair_[1], valid_set_, language_pair_[1]));
    }
    if (usage_ == "test" || usage_ == "all") {
      src_path_list.push_back(
        GenerateIWSLT2017XMLFileName(root_dir, language_pair_[0], language_pair_[1], test_set_, language_pair_[0]));
      target_path_list.push_back(
        GenerateIWSLT2017XMLFileName(root_dir, language_pair_[0], language_pair_[1], test_set_, language_pair_[1]));
    }
  }
  RETURN_IF_NOT_OK(GenerateNewFile(src_path_list, target_path_list, &src_target_file_list_));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
