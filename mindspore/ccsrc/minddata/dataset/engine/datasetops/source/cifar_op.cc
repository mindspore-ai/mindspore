/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/source/cifar_op.h"

#include <algorithm>
#include <fstream>
#include <utility>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {

constexpr uint32_t kCifarImageHeight = 32;
constexpr uint32_t kCifarImageWidth = 32;
constexpr uint32_t kCifarImageChannel = 3;
constexpr uint32_t kCifarBlockImageNum = 5;
constexpr uint32_t kCifarImageSize = kCifarImageHeight * kCifarImageWidth * kCifarImageChannel;
CifarOp::CifarOp(CifarType type, const std::string &usage, int32_t num_works, const std::string &file_dir,
                 int32_t queue_size, std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler)
    : MappableLeafOp(num_works, queue_size, std::move(sampler)),
      cifar_type_(type),
      usage_(usage),
      folder_path_(file_dir),
      data_schema_(std::move(data_schema)) {
  constexpr uint64_t kUtilQueueSize = 512;
  cifar_raw_data_block_ = std::make_unique<Queue<std::vector<unsigned char>>>(kUtilQueueSize);
}

Status CifarOp::RegisterAndLaunchThreads() {
  RETURN_IF_NOT_OK(ParallelOp::RegisterAndLaunchThreads());
  RETURN_IF_NOT_OK(cifar_raw_data_block_->Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask(
    "Get cifar data block", std::bind(&CifarOp::ReadCifarBlockDataAsync, this), nullptr, id()));
  return Status::OK();
}

// Load 1 TensorRow (image,label). 1 function call produces 1 TensorTow
Status CifarOp::LoadTensorRow(row_id_type index, TensorRow *trow) {
  std::shared_ptr<Tensor> label;
  std::shared_ptr<Tensor> fine_label;
  std::shared_ptr<Tensor> ori_image = cifar_image_label_pairs_[index].first;
  std::shared_ptr<Tensor> copy_image;
  uint64_t path_index = static_cast<uint64_t>(std::ceil(index / kCifarBlockImageNum));
  RETURN_IF_NOT_OK(Tensor::CreateFromTensor(ori_image, &copy_image));
  RETURN_IF_NOT_OK(Tensor::CreateScalar(cifar_image_label_pairs_[index].second[0], &label));

  if (cifar_image_label_pairs_[index].second.size() > 1) {
    RETURN_IF_NOT_OK(Tensor::CreateScalar(cifar_image_label_pairs_[index].second[1], &fine_label));
    (*trow) = TensorRow(index, {copy_image, std::move(label), std::move(fine_label)});
    // Add file path info
    trow->setPath({path_record_[path_index], path_record_[path_index], path_record_[path_index]});
  } else {
    (*trow) = TensorRow(index, {copy_image, std::move(label)});
    // Add file path info
    trow->setPath({path_record_[path_index], path_record_[path_index]});
  }

  return Status::OK();
}

void CifarOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nNumber of rows:" << num_rows_ << "\nCifar directory: " << folder_path_ << "\n\n";
  }
}

Status CifarOp::ReadCifarBlockDataAsync() {
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(GetCifarFiles());
  if (cifar_type_ == kCifar10) {
    RETURN_IF_NOT_OK(ReadCifar10BlockData());
  } else {
    RETURN_IF_NOT_OK(ReadCifar100BlockData());
  }

  return Status::OK();
}

Status CifarOp::ReadCifar10BlockData() {
  // CIFAR 10 has 6 bin files. data_batch_1.bin ... data_batch_5.bin and 1 test_batch.bin file
  // each of the file has exactly 10K images and labels and size is 30,730 KB
  // each image has the dimension of 32 x 32 x 3 = 3072 plus 1 label (label has 10 classes) so each row has 3073 bytes
  constexpr uint32_t num_cifar10_records = 10000;
  uint32_t block_size = (kCifarImageSize + 1) * kCifarBlockImageNum;  // about 2M
  std::vector<unsigned char> image_data(block_size * sizeof(unsigned char), 0);
  for (auto &file : cifar_files_) {
    // check the validity of the file path
    Path file_path(file);
    CHECK_FAIL_RETURN_UNEXPECTED(file_path.Exists() && !file_path.IsDirectory(),
                                 "Invalid cifar10 file, " + file + " does not exist or is a directory.");
    std::string file_name = file_path.Basename();

    if (usage_ == "train") {
      if (file_name.find("data_batch") == std::string::npos) {
        continue;
      }
    } else if (usage_ == "test") {
      if (file_name.find("test_batch") == std::string::npos) {
        continue;
      }
    } else {  // get all the files that contain the word batch, aka any cifar 100 files
      if (file_name.find("batch") == std::string::npos) {
        continue;
      }
    }

    std::ifstream in(file, std::ios::binary);
    CHECK_FAIL_RETURN_UNEXPECTED(
      in.is_open(), "Invalid cifar10 file, failed to open " + file + ", the file is damaged or permission denied.");

    for (uint32_t index = 0; index < num_cifar10_records / kCifarBlockImageNum; ++index) {
      (void)in.read(reinterpret_cast<char *>(&(image_data[0])), block_size * sizeof(unsigned char));
      CHECK_FAIL_RETURN_UNEXPECTED(!in.fail(), "Invalid cifar10 file, failed to read data from: " + file +
                                                 ", re-download dataset(make sure it is CIFAR-10 binary version).");
      (void)cifar_raw_data_block_->EmplaceBack(image_data);
      // Add file path info
      path_record_.push_back(file);
    }
    in.close();
  }
  (void)cifar_raw_data_block_->EmplaceBack(std::vector<unsigned char>());  // end block

  return Status::OK();
}

Status CifarOp::ReadCifar100BlockData() {
  // CIFAR 100 has 2 bin files. train.bin (60K imgs)  153,700KB and test.bin (30,740KB) (10K imgs)
  // each img has two labels. Each row then is 32 * 32 *5 + 2 = 3,074 Bytes
  uint32_t num_cifar100_records = 0;  // test:10000, train:50000
  constexpr uint32_t num_cifar100_test_records = 10000;
  constexpr uint32_t num_cifar100_train_records = 50000;
  uint32_t block_size = (kCifarImageSize + 2) * kCifarBlockImageNum;  // about 2M
  std::vector<unsigned char> image_data(block_size * sizeof(unsigned char), 0);
  for (auto &file : cifar_files_) {
    // check the validity of the file path
    Path file_path(file);
    CHECK_FAIL_RETURN_UNEXPECTED(file_path.Exists() && !file_path.IsDirectory(),
                                 "Invalid cifar100 file, " + file + " does not exist or is a directory.");
    std::string file_name = file_path.Basename();

    // if usage is train/test, get only these 2 files
    if (usage_ == "train" && file_name.find("train") == std::string::npos) {
      continue;
    }
    if (usage_ == "test" && file_name.find("test") == std::string::npos) {
      continue;
    }

    if (file_name.find("test") != std::string::npos) {
      num_cifar100_records = num_cifar100_test_records;
    } else if (file_name.find("train") != std::string::npos) {
      num_cifar100_records = num_cifar100_train_records;
    } else {
      RETURN_STATUS_UNEXPECTED("Invalid cifar100 file, Cifar100 train/test file is missing in: " + file_name);
    }

    std::ifstream in(file, std::ios::binary);
    CHECK_FAIL_RETURN_UNEXPECTED(
      in.is_open(), "Invalid cifar100 file, failed to open " + file + ", the file is damaged or permission denied.");

    for (uint32_t index = 0; index < num_cifar100_records / kCifarBlockImageNum; index++) {
      (void)in.read(reinterpret_cast<char *>(&(image_data[0])), block_size * sizeof(unsigned char));
      CHECK_FAIL_RETURN_UNEXPECTED(!in.fail(), "Invalid cifar100 file, failed to read data from: " + file +
                                                 ", re-download dataset(make sure it is CIFAR-100 binary version).");
      (void)cifar_raw_data_block_->EmplaceBack(image_data);
      // Add file path info
      path_record_.push_back(file);
    }
    in.close();
  }
  (void)cifar_raw_data_block_->EmplaceBack(std::vector<unsigned char>());  // block end
  return Status::OK();
}

Status CifarOp::GetCifarFiles() {
  const std::string extension = ".bin";
  Path dir_path(folder_path_);
  auto dirIt = Path::DirIterator::OpenDirectory(&dir_path);
  if (dirIt) {
    while (dirIt->HasNext()) {
      Path file = dirIt->Next();
      if (file.Extension() == extension) {
        cifar_files_.push_back(file.ToString());
      }
    }
  } else {
    RETURN_STATUS_UNEXPECTED("Invalid directory, " + dir_path.ToString() + " is not a directory or permission denied.");
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!cifar_files_.empty(),
                               "Invalid cifar folder, cifar(.bin) files are missing under " + folder_path_);
  std::sort(cifar_files_.begin(), cifar_files_.end());
  return Status::OK();
}

Status CifarOp::PrepareData() {
  std::vector<unsigned char> block;
  RETURN_IF_NOT_OK(cifar_raw_data_block_->PopFront(&block));
  uint32_t cur_block_index = 0;
  while (!block.empty()) {
    for (uint32_t index = 0; index < kCifarBlockImageNum; ++index) {
      std::vector<uint32_t> labels;
      uint32_t label = block[cur_block_index++];
      labels.push_back(label);
      if (cifar_type_ == kCifar100) {
        uint32_t fine_label = block[cur_block_index++];
        labels.push_back(fine_label);
      }

      std::shared_ptr<Tensor> image_tensor;
      RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({kCifarImageHeight, kCifarImageWidth, kCifarImageChannel}),
                                           data_schema_->Column(0).Type(), &image_tensor));
      auto itr = image_tensor->begin<uint8_t>();
      uint32_t total_pix = kCifarImageHeight * kCifarImageWidth;
      for (uint32_t pix = 0; pix < total_pix; ++pix) {
        for (uint32_t ch = 0; ch < kCifarImageChannel; ++ch) {
          *itr = block[cur_block_index + ch * total_pix + pix];
          ++itr;
        }
      }
      cur_block_index += total_pix * kCifarImageChannel;
      cifar_image_label_pairs_.emplace_back(std::make_pair(image_tensor, labels));
    }
    RETURN_IF_NOT_OK(cifar_raw_data_block_->PopFront(&block));
    cur_block_index = 0;
  }
  cifar_image_label_pairs_.shrink_to_fit();
  num_rows_ = cifar_image_label_pairs_.size();
  if (num_rows_ == 0) {
    std::string api = cifar_type_ == kCifar10 ? "Cifar10Dataset" : "Cifar100Dataset";
    RETURN_STATUS_UNEXPECTED("Invalid data, " + api +
                             " API can't read the data file (interface mismatch or no data found). "
                             "Check file in directory:" +
                             folder_path_);
  }
  cifar_raw_data_block_->Reset();
  return Status::OK();
}

// Derived from RandomAccessOp
Status CifarOp::GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const {
  if (cls_ids == nullptr || !cls_ids->empty()) {
    RETURN_STATUS_UNEXPECTED(
      "[Internal ERROR] Map for containing image-index pair is nullptr or has been set in other place,"
      "it must be empty before using GetClassIds.");
  }

  for (uint64_t index = 0; index < cifar_image_label_pairs_.size(); ++index) {
    uint32_t label = (cifar_image_label_pairs_[index].second)[0];
    (*cls_ids)[label].push_back(index);
  }

  for (auto &pair : (*cls_ids)) {
    pair.second.shrink_to_fit();
  }
  return Status::OK();
}

Status CifarOp::CountTotalRows(const std::string &dir, const std::string &usage, bool isCIFAR10, int64_t *count) {
  // the logic of counting the number of samples is copied from ReadCifar100Block() and ReadCifar10Block()
  // Note that this count logic is flawed, should be able to copy the sampler of original CifarOp without state
  *count = 0;
  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto new_sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);

  CifarType type = isCIFAR10 ? kCifar10 : kCifar100;
  // build a new unique schema object
  auto new_schema = std::make_unique<DataSchema>();
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    new_schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  if (type == kCifar10) {
    RETURN_IF_NOT_OK(
      new_schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  } else {
    RETURN_IF_NOT_OK(new_schema->AddColumn(
      ColDescriptor("coarse_label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
    TensorShape another_scalar = TensorShape::CreateScalar();
    RETURN_IF_NOT_OK(new_schema->AddColumn(
      ColDescriptor("fine_label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &another_scalar)));
  }
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connect_size = cfg->op_connector_size();
  std::shared_ptr<CifarOp> op = std::make_shared<CifarOp>(type, usage, num_workers, dir, op_connect_size,
                                                          std::move(new_schema), std::move(new_sampler));

  RETURN_IF_NOT_OK(op->GetCifarFiles());
  if (op->cifar_type_ == kCifar10) {
    constexpr int64_t num_cifar10_records = 10000;
    for (auto &file : op->cifar_files_) {
      Path file_path(file);
      CHECK_FAIL_RETURN_UNEXPECTED(file_path.Exists() && !file_path.IsDirectory(),
                                   "Invalid cifar10 file, " + file + " does not exist or is a directory.");
      std::string file_name = file_path.Basename();

      if (op->usage_ == "train") {
        if (file_name.find("data_batch") == std::string::npos) {
          continue;
        }
      } else if (op->usage_ == "test") {
        if (file_name.find("test_batch") == std::string::npos) {
          continue;
        }
      } else {  // get all the files that contain the word batch, aka any cifar 100 files
        if (file_name.find("batch") == std::string::npos) {
          continue;
        }
      }

      std::ifstream in(file, std::ios::binary);

      CHECK_FAIL_RETURN_UNEXPECTED(
        in.is_open(), "Invalid cifar10 file, failed to open " + file + ", the file is damaged or permission denied.");
      *count = *count + num_cifar10_records;
    }
    return Status::OK();
  } else {
    const uint32_t kCifar100RecordsPerTestFile = 10000;
    const uint32_t kCifar100RecordsPerTrainFile = 50000;
    int64_t num_cifar100_records = 0;
    for (auto &file : op->cifar_files_) {
      Path file_path(file);
      std::string file_name = file_path.Basename();

      CHECK_FAIL_RETURN_UNEXPECTED(file_path.Exists() && !file_path.IsDirectory(),
                                   "Invalid cifar100 file, " + file + " does not exist or is a directory.");

      if (op->usage_ == "train" && file_path.Basename().find("train") == std::string::npos) {
        continue;
      }
      if (op->usage_ == "test" && file_path.Basename().find("test") == std::string::npos) {
        continue;
      }

      if (file_name.find("test") != std::string::npos) {
        num_cifar100_records += kCifar100RecordsPerTestFile;
      } else if (file_name.find("train") != std::string::npos) {
        num_cifar100_records += kCifar100RecordsPerTrainFile;
      }
      std::ifstream in(file, std::ios::binary);
      CHECK_FAIL_RETURN_UNEXPECTED(
        in.is_open(), "Invalid cifar100 file, failed to open " + file + ", the file is damaged or permission denied.");
    }
    *count = num_cifar100_records;
    return Status::OK();
  }
}

Status CifarOp::ComputeColMap() {
  // set the column name map (base class field)
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->Column(i).Name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

Status CifarOp::InitPullMode() {
  RETURN_IF_NOT_OK(cifar_raw_data_block_->Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask(
    "Get cifar data block", std::bind(&CifarOp::ReadCifarBlockDataAsync, this), nullptr, id()));
  return PrepareData();
}
}  // namespace dataset
}  // namespace mindspore
