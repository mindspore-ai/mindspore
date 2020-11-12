/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "MDToDApi.h"

#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/execute.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/include/vision.h"
#include "minddata/dataset/util/data_helper.h"
#if defined(__ANDROID__) || defined(ANDROID)
#include <android/log.h>
#include <android/asset_manager.h>
#endif

using mindspore::dataset::Path;
using mindspore::dataset::Tensor;

using mindspore::dataset;

using mindspore::LogStream;
using mindspore::MsLogLevel::DEBUG;
using mindspore::MsLogLevel::ERROR;
using mindspore::MsLogLevel::INFO;

using mindspore::dataset::BorderType;
using mindspore::dataset::InterpolationMode;
using mindspore::dataset::Status;

class MDToDApi {
 public:
  std::shared_ptr<Dataset> _ds;
  std::shared_ptr<Iterator> _iter;
  std::vector<std::shared_ptr<TensorOperation>> _augs;
  std::string _storage_folder;
  std::string _folder_path;
  bool _hasBatch;
  int64_t _file_id;

  MDToDApi() : _ds(nullptr), _iter(nullptr), _augs({}), _storage_folder(""), _file_id(-1), _hasBatch(false) {
    MS_LOG(WARNING) << "MDToDAPI Call constructor";
  }
  ~MDToDApi() {
    MS_LOG(WARNING) << "MDToDAPI Call destructor";
    _augs.clear();
    _ds = nullptr;
    _iter = nullptr;
  }
};

std::vector<std::string> MDToDBuffToVector(MDToDBuff_t StrBuff) {
  std::vector<std::string> strVector;
  if (StrBuff.DataSize > 0) {
    const char *p = reinterpret_cast<char *>(StrBuff.Buff);
    do {
      strVector.push_back(std::string(p));
      p += strVector.back().size() + 1;
    } while (p < reinterpret_cast<char *>(StrBuff.Buff) + StrBuff.DataSize);
  }
  return strVector;
}

extern "C" int MDToDApi_pathTest(const char *path) {
  Path f(path);
  MS_LOG(WARNING) << f.Exists() << f.IsDirectory() << f.ParentPath();
  // Print out the first few items in the directory
  auto dir_it = Path::DirIterator::OpenDirectory(&f);
  MS_LOG(WARNING) << dir_it.get();
  int i = 0;
  while (dir_it->hasNext()) {
    Path v = dir_it->next();
    MS_LOG(WARNING) << v.toString() << "\n";
    i++;
    if (i > 5) break;
  }
  return 0;
}

extern "C" MDToDApi *MDToDApi_createPipeLine(MDToDConf_t MDConf) {
  MS_LOG(WARNING) << "Start createPipeLine";
  std::string folder_path(MDConf.pFolderPath);
  std::string schema_file(MDConf.pSchemFile);
  std::vector<std::string> column_names = MDToDBuffToVector(MDConf.columnsToReadBuff);
  if (std::find(column_names.begin(), column_names.end(), "id") == column_names.end()) {
    MS_LOG(WARNING) << "Column id not foud adding it ";
    column_names.push_back("id");
  }
  std::vector<std::shared_ptr<TensorOperation>> mapOperations;
  if (std::find(column_names.begin(), column_names.end(), "image") != column_names.end()) {
    MS_LOG(WARNING) << "Found column image create map with:";
    MS_LOG(WARNING) << "resize: { " << MDConf.ResizeSizeWH[0] << ", " << MDConf.ResizeSizeWH[1] << " }";
    MS_LOG(WARNING) << "crop: { " << MDConf.CropSizeWH[0] << ", " << MDConf.CropSizeWH[1] << " }";
    MS_LOG(WARNING) << "MEAN: { " << MDConf.MEAN[0] << ", " << MDConf.MEAN[1] << ", " << MDConf.MEAN[2] << " }";
    MS_LOG(WARNING) << "STD: { " << MDConf.STD[0] << ", " << MDConf.STD[1] << ", " << MDConf.STD[2] << " }";

    if ((MDConf.ResizeSizeWH[0] != 0) && (MDConf.ResizeSizeWH[1] != 0)) {
      std::vector<int> Resize(MDConf.ResizeSizeWH, MDConf.ResizeSizeWH + 2);
      std::shared_ptr<TensorOperation> resize_op = vision::Resize(Resize);
      assert(resize_op != nullptr);
      MS_LOG(WARNING) << "Push back resize";
      mapOperations.push_back(resize_op);
    }
    if ((MDConf.CropSizeWH[0] != 0) && (MDConf.CropSizeWH[1] != 0)) {
      std::vector<int> Crop(MDConf.CropSizeWH, MDConf.CropSizeWH + 2);
      std::shared_ptr<TensorOperation> center_crop_op = vision::CenterCrop(Crop);
      assert(center_crop_op != nullptr);
      MS_LOG(WARNING) << "Push back crop";
      mapOperations.push_back(center_crop_op);
    }
  }
  std::shared_ptr<Dataset> ds = nullptr;
  MS_LOG(INFO) << "Read id =" << MDConf.fileid << " (-1) for all";
  if (MDConf.fileid > -1) {
    // read specific image using SequentialSampler
    ds = Album(folder_path, schema_file, column_names, true, SequentialSampler(MDConf.fileid, 1L));
  } else {
    // Distributed sampler takes num_shards then shard_id
    ds = Album(folder_path, schema_file, column_names, true, SequentialSampler());
  }
  ds = ds->SetNumWorkers(1);

  assert(ds != nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 1;
  ds = ds->Repeat(repeat_num);
  assert(ds != nullptr);

  // Create objects for the tensor ops
  MS_LOG(INFO) << " Create pipline parameters";
  MS_LOG(INFO) << "floder path: " << folder_path << " , schema json: " << schema_file;
  MS_LOG(INFO) << "Reading columns:";
  for (auto str : column_names) {
    MS_LOG(INFO) << str << " ";
  }
  bool hasBatch = false;

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  if (nullptr == iter) {
    MS_LOG(ERROR) << "Iterator creation failed";
    return nullptr;
  }
  assert(iter != nullptr);
  MDToDApi *pMDToDApi = new MDToDApi;
  pMDToDApi->_ds = ds;
  pMDToDApi->_iter = iter;
  pMDToDApi->_augs = mapOperations;
  pMDToDApi->_storage_folder = std::string(MDConf.pStoragePath);
  pMDToDApi->_folder_path = folder_path;
  pMDToDApi->_hasBatch = hasBatch;
  return pMDToDApi;
}

template <typename T>
void MDBuffToVector(MDToDBuff_t MDBuff, std::vector<T> *vec) {
  vec.clear();
  if (MDBuff.DataSize > 0) {
    int nofElements = MDBuff.DataSize / sizeof(T);
    *vec.assign(reinterpret_cast<T *>(MDBuff.Buff), reinterpret_cast<T *>(MDBuff.Buff) + nofElements);
  }
}

template <typename T>
void GetValue(std::unordered_map<std::string, std::shared_ptr<Tensor>> row, std::string columnName, T *o) {
  auto column = row[columnName];
  if (NULL != column) {
    MS_LOG(INFO) << "Tensor " << columnName << " shape: " << column->shape() << " type: " << column->type()
                 << " bytes: " << column->SizeInBytes();
    column->GetItemAt<T>(o, {});
    MS_LOG(INFO) << columnName << ": " << +*o;
  } else {
    MS_LOG(INFO) << "Tensor " << columnName << " Not found"
                 << ".";
    *o = 0;
  }
}

void GetTensorToBuff(std::unordered_map<std::string, std::shared_ptr<Tensor>> row, std::string columnName,
                     bool hasBatch, MDToDBuff_t *resBuff) {
  auto column = row[columnName];
  resBuff->TensorSize[0] = resBuff->TensorSize[1] = resBuff->TensorSize[2] = resBuff->TensorSize[3] =
    0;  // Mark all dims do not exist in tensor
  int firstDim = (hasBatch) ? 1 : 0;
  if (NULL != column) {
    MS_LOG(INFO) << "Tensor " << columnName << " shape: " << column->shape() << " type: " << column->type()
                 << " bytes: " << column->SizeInBytes() << "nof elements: " << column->shape()[firstDim];
    auto tesoreShape = column->shape().AsVector();
    for (int ix = 0; ix < tesoreShape.size(); ix++) {
      MS_LOG(INFO) << "Tensor " << columnName << " shape[" << ix << "] = " << tesoreShape[ix];
      resBuff->TensorSize[ix] = tesoreShape[ix];
    }
    if (!hasBatch) {
      for (int ix = 3; ix > 0; ix--) {
        resBuff->TensorSize[ix] = resBuff->TensorSize[ix - 1];
      }
      resBuff->TensorSize[0] = 1;
    }
    if (column->shape()[firstDim] > 0) {
      if (DataType::DE_STRING == column->type()) {
        std::string str;
        for (int ix = 0; ix < column->shape()[firstDim]; ix++) {
          std::string_view strView;
          if (hasBatch) {
            column->GetItemAt(&strView, {0, ix});
          } else {
            column->GetItemAt(&strView, {ix});
          }
          MS_LOG(INFO) << "string " << columnName << "[" << ix << "]:" << strView << " (size: " << strView.size()
                       << ")";
          str.append(strView);
          str.push_back('\0');
        }
        resBuff->DataSize = str.size();
        errno_t ret = memcpy_s(resBuff->Buff, resBuff->MaxBuffSize, str.data(), resBuff->DataSize);
        if (ret != 0) {
          resBuff->DataSize = 0;  // memcpy fail amount of data copied is 0
          MS_LOG(ERROR) << "memcpy_s return: " << ret;
        }
      } else {
        DataHelper dh;
        resBuff->DataSize =
          dh.DumpData(column->GetBuffer(), column->SizeInBytes(), resBuff->Buff, resBuff->MaxBuffSize);
      }
      MS_LOG(INFO) << columnName << " " << resBuff->DataSize
                   << " bytesCopyed to buff (MaxBuffSize: " << resBuff->MaxBuffSize << ") ";
      if (0 == resBuff->DataSize) {
        MS_LOG(ERROR) << "Copy Failed!!!! " << columnName << " Too large"
                      << ".";  // memcpy failed
      }
    } else {
      MS_LOG(INFO) << "Tensor " << columnName << " is empty (has size 0)";
    }
  } else {
    MS_LOG(INFO) << "Tensor " << columnName << " was not read.";
  }
}

extern "C" int MDToDApi_GetNext(MDToDApi *pMDToDApi, MDToDResult_t *results) {
  MS_LOG(INFO) << "Start GetNext";
  if (pMDToDApi == nullptr) {
    MS_LOG(ERROR) << "GetNext called with nullptr. Abort";
    assert(pMDToDApi != nullptr);
  }

  // Set defualt
  results->fileid = -1;
  results->embeddingBuff.DataSize = 0;
  results->imageBuff.DataSize = 0;
  MS_LOG(INFO) << "Start GetNext [1]" << pMDToDApi;
  // get next row for dataset
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  if (pMDToDApi->_iter == nullptr) {
    MS_LOG(ERROR) << "GetNext called with no iterator. abort";
    return -1;
  }
  // create Execute functions, this replaces Map in Pipeline
  pMDToDApi->_iter->GetNextRow(&row);
  if (row.size() != 0) {
    if ((pMDToDApi->_augs).size() > 0) {
      // String and Tensors
      GetTensorToBuff(row, "image_filename", pMDToDApi->_hasBatch, &results->fileNameBuff);
      // for each operation, run eager mode, single threaded operation, will have to memcpy
      // regardless
      for (int i = 0; i < (pMDToDApi->_augs).size(); i++) {
        // each Execute call will invoke a memcpy, this cannot really be optimized further
        // for this use case, std move is added for fail save.
        row["image"] = Execute((pMDToDApi->_augs)[i])(std::move(row["image"]));
        if (row["image"] == nullptr) {
          // nullptr means that the eager mode image processing failed, we fail in this case
          return -1;
        }
      }
    }
    // FILE ID
    GetValue<int64_t>(row, "id", &results->fileid);
    pMDToDApi->_file_id = results->fileid;  // hold current file id to enable embeddings update (no itr->getCurrent)
    // IS FOR TRAIN
    GetValue<int32_t>(row, "_isForTrain", &results->isForTrain);
    GetValue<int32_t>(row, "_noOfFaces", &results->noOfFaces);
    // String and Tensors
    GetTensorToBuff(row, "image_filename", pMDToDApi->_hasBatch, &results->fileNameBuff);
    GetTensorToBuff(row, "image", pMDToDApi->_hasBatch, &results->imageBuff);
    GetTensorToBuff(row, "_embedding", pMDToDApi->_hasBatch, &results->embeddingBuff);
    GetTensorToBuff(row, "label", pMDToDApi->_hasBatch, &results->labelBuff);
    GetTensorToBuff(row, "_boundingBoxes", pMDToDApi->_hasBatch, &results->boundingBoxesBuff);
    GetTensorToBuff(row, "_confidences", pMDToDApi->_hasBatch, &results->confidencesBuff);
    GetTensorToBuff(row, "_landmarks", pMDToDApi->_hasBatch, &results->landmarksBuff);
    GetTensorToBuff(row, "_faceFileNames", pMDToDApi->_hasBatch, &results->faceFileNamesBuff);
    GetTensorToBuff(row, "_imageQualities", pMDToDApi->_hasBatch, &results->imageQualitiesBuff);
    GetTensorToBuff(row, "_faceEmbeddings", pMDToDApi->_hasBatch, &results->faceEmbeddingsBuff);
    return 0;
  }
  return -1;
}

extern "C" int MDToDApi_Stop(MDToDApi *pMDToDApi) {
  // Manually terminate the pipeline
  pMDToDApi->_iter->Stop();
  MS_LOG(WARNING) << "pipline stoped";
  return 0;
}

extern "C" int MDToDApi_Destroy(MDToDApi *pMDToDApi) {
  MS_LOG(WARNING) << "pipeline deleted start";
  pMDToDApi->_iter->Stop();
  delete pMDToDApi;
  MS_LOG(WARNING) << "pipeline deleted end";
  return 0;
}

int GetJsonFullFileName(MDToDApi *pMDToDApi, std::string *filePath) {
  int64_t file_id = pMDToDApi->_file_id;
  if (file_id < 0) {
    MS_LOG(ERROR) << "Illigal file ID to update: " << file_id << ".";
    return -1;
  }
  std::string converted = std::to_string(pMDToDApi->_file_id);
  *filePath = pMDToDApi->_folder_path + "/" + converted + ".json";
  return 0;
}

extern "C" int MDToDApi_UpdateEmbeding(MDToDApi *pMDToDApi, const char *column, float *emmbeddings,
                                       size_t emmbeddingsSize) {
  auto columnName = std::string(column);
  MS_LOG(INFO) << "Start update " << columnName;

  std::string converted = std::to_string(pMDToDApi->_file_id);
  std::string embedding_file_path = pMDToDApi->_storage_folder + "/" + converted + columnName + ".bin";
  DataHelper dh;
  MS_LOG(INFO) << "Try to save file " << embedding_file_path;
  std::vector<float> bin_content(emmbeddings, emmbeddings + emmbeddingsSize);
  Status rc = dh.template WriteBinFile<float>(embedding_file_path, bin_content);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Fail to write embedding file: " << embedding_file_path << ".";
    return -1;
  }
  MS_LOG(INFO) << "Saved file " << embedding_file_path;

  std::string file_path;
  if (0 != GetJsonFullFileName(pMDToDApi, &file_path)) {
    MS_LOG(ERROR) << "Failed to update " << columnName;
    return -1;
  }

  MS_LOG(INFO) << "Updating json file: " << file_path;
  rc = dh.UpdateValue(file_path, std::string(column), embedding_file_path);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Fail to update json: " << file_path << ".";
    return -1;
  }
  return 0;
}

extern "C" int MDToDApi_UpdateStringArray(MDToDApi *pMDToDApi, const char *column, MDToDBuff_t MDbuff) {
  auto columnName = std::string(column);
  std::string file_path;
  if (0 != GetJsonFullFileName(pMDToDApi, &file_path)) {
    MS_LOG(ERROR) << "Failed to update " << columnName;
    return -1;
  }
  MS_LOG(INFO) << "Start Update string array column: " << columnName << " in file " << file_path;
  DataHelper dh;
  std::vector<std::string> strVec;
  if (MDbuff.DataSize > 0) {
    const char *p = reinterpret_cast<char *>(MDbuff.Buff);
    do {
      strVec.push_back(std::string(p));
      p += strVec.back().size() + 1;
    } while (p < reinterpret_cast<char *>(MDbuff.Buff) + MDbuff.DataSize);
  }
  Status rc = dh.UpdateArray(file_path, columnName, strVec);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Fail to update json: " << file_path << ".";
    return -1;
  }
  return 0;
}

extern "C" int MDToDApi_UpdateFloatArray(MDToDApi *pMDToDApi, const char *column, MDToDBuff_t MDBuff) {
  auto columnName = std::string(column);
  std::string file_path;
  if (0 != GetJsonFullFileName(pMDToDApi, &file_path)) {
    MS_LOG(ERROR) << "Faile to updaet " << columnName;
    return -1;
  }
  MS_LOG(INFO) << "Start Update float Array column: " << columnName << " in file " << file_path;
  DataHelper dh;
  std::vector<float> vec;
  MDBuffToVector<float>(MDBuff, &vec);
  Status rc = dh.UpdateArray<float>(file_path, columnName, vec);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Fail to update json: " << file_path << ".";
    return -1;
  }
  return 0;
}

extern "C" int MDToDApi_UpdateIsForTrain(MDToDApi *pMDToDApi, int32_t isForTrain) {
  int64_t file_id = pMDToDApi->_file_id;
  MS_LOG(INFO) << "Start Update isForTRain for id: " << file_id << " To " << isForTrain;

  if (file_id < 0) return -1;
  std::string converted = std::to_string(pMDToDApi->_file_id);
  std::string file_path = pMDToDApi->_folder_path + "/" + converted + ".json";
  DataHelper dh;
  MS_LOG(INFO) << "Updating file: " << file_path;
  Status rc = dh.UpdateValue<int32_t>(file_path, "_isForTrain", isForTrain, "");
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Fail to update json: " << file_path << ".";
    return -1;
  }
  return 0;
}

extern "C" int MDToDApi_UpdateNoOfFaces(MDToDApi *pMDToDApi, int32_t noOfFaces) {
  int64_t file_id = pMDToDApi->_file_id;
  MS_LOG(INFO) << "Start Update noOfFaces for id: " << file_id << " To " << noOfFaces;

  if (file_id < 0) return -1;
  std::string converted = std::to_string(pMDToDApi->_file_id);
  std::string file_path = pMDToDApi->_folder_path + "/" + converted + ".json";
  DataHelper dh;
  MS_LOG(INFO) << "Updating file: " << file_path;
  Status rc = dh.UpdateValue<int32_t>(file_path, "_noOfFaces", noOfFaces, "");
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Fail to update json: " << file_path << ".";
    return -1;
  }
  return 0;
}
