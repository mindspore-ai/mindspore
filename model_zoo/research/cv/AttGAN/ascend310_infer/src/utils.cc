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
#include "inc/utils.h"

#include <fstream>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <sstream>

using mindspore::MSTensor;
using mindspore::DataType;

std::vector<std::string> GetAllFiles(std::string_view dirName) {
    struct dirent *filename;
    DIR *dir = OpenDir(dirName);
    if (dir == nullptr) {
        return {};
    }
    std::vector<std::string> res;
    while ((filename = readdir(dir)) != nullptr) {
        std::string dName = std::string(filename->d_name);
        if (dName == "." || dName == ".." || filename->d_type != DT_REG) {
            continue;
        }
        res.emplace_back(std::string(dirName) + "/" + filename->d_name);
    }
    std::sort(res.begin(), res.end());
    for (auto &f : res) {
        std::cout << "image file: " << f << std::endl;
    }
    return res;
}

int WriteResult(const std::string& imageFile, const std::vector<MSTensor> &outputs) {
    std::string homePath = "./result_Files";
    for (size_t i = 0; i < outputs.size(); ++i) {
        size_t outputSize;
        std::shared_ptr<const void> netOutput;
        netOutput = outputs[i].Data();
        outputSize = outputs[i].DataSize();
        int pos = imageFile.rfind('/');
        std::string fileName(imageFile, pos + 1);
        std::cout << fileName << std::endl;
        fileName.replace(fileName.find('.'), fileName.size() - fileName.find('.'), ".bin");
        std::string outFileName = homePath + "/" + fileName;
        FILE * outputFile = fopen(outFileName.c_str(), "wb");
        fwrite(netOutput.get(), sizeof(char), outputSize, outputFile);
        fclose(outputFile);
        outputFile = nullptr;
    }
    return 0;
}

mindspore::MSTensor ReadFileToTensor(const std::string &file) {
    if (file.empty()) {
        std::cout << "Pointer file is nullptr" << std::endl;
        return mindspore::MSTensor();
    }

    std::ifstream ifs(file);
    if (!ifs.good()) {
        std::cout << "File: " << file << " is not exist" << std::endl;
        return mindspore::MSTensor();
    }

    if (!ifs.is_open()) {
        std::cout << "File: " << file << "open failed" << std::endl;
        return mindspore::MSTensor();
    }

    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    mindspore::MSTensor buffer(file, mindspore::DataType::kNumberTypeUInt8,
                               {static_cast<int64_t>(size)}, nullptr, size);
    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char *>(buffer.MutableData()), size);
    ifs.close();
    return buffer;
}

std::vector<std::string> split(std::string inputs) {
    std::vector<std::string> line;
    std::stringstream stream(inputs);
    std::string result;
    while ( stream >> result ) {
        line.push_back(result);
    }
    return line;
}

std::vector<mindspore::MSTensor> ReadCfgToTensor(const std::string &file, size_t *n_ptr) {
    std::vector<mindspore::MSTensor> res;
    if (file.empty()) {
        std::cout << "Pointer file is nullptr." << std::endl;
        exit(1);
    }

    std::ifstream ifs(file);
    if (!ifs.good()) {
        std::cout << "File: " << file << " is not exist." << std::endl;
        exit(1);
    }

    if (!ifs.is_open()) {
        std::cout << "File: " << file << " open failed." << std::endl;
        exit(1);
    }

    std::string n_images;
    std::string n_attrs;
    getline(ifs, n_images);
    getline(ifs, n_attrs);

    auto n_images_ = std::stoi(n_images);
    auto n_attrs_ = std::stoi(n_attrs);
    *n_ptr = n_attrs_;
    std::cout << "Image number is " << n_images << std::endl;
    std::cout << "Attribute number is " << n_attrs << std::endl;

    auto all_lines = n_images_ * n_attrs_;
    for (auto i = 0; i < all_lines; i++) {
        std::string val;
        getline(ifs, val);
        std::vector<std::string> val_split = split(val);
        void *data = malloc(sizeof(float)*n_attrs_);
        float *elements = reinterpret_cast<float *>(data);
        for (auto j = 0; j < n_attrs_; j++) elements[j] = atof(val_split[j].c_str());
        auto size = sizeof(float) * n_attrs_;
        mindspore::MSTensor buffer(file + std::to_string(i), mindspore::DataType::kNumberTypeFloat32,
                                   {static_cast<int64_t>(size)}, nullptr, size);
        memcpy(buffer.MutableData(), elements, size);
        res.emplace_back(buffer);
    }
    ifs.close();
    return res;
}

DIR *OpenDir(std::string_view dirName) {
    if (dirName.empty()) {
        std::cout << " dirName is null ! " << std::endl;
        return nullptr;
    }
    std::string realPath = RealPath(dirName);
    struct stat s;
    lstat(realPath.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        std::cout << "dirName is not a valid directory !" << std::endl;
        return nullptr;
    }
    DIR *dir;
    dir = opendir(realPath.c_str());
    if (dir == nullptr) {
        std::cout << "Can not open dir " << dirName << std::endl;
        return nullptr;
    }
    std::cout << "Successfully opened the dir " << dirName << std::endl;
    return dir;
}

std::string RealPath(std::string_view path) {
    char realPathMem[PATH_MAX] = {0};
    char *realPathRet = nullptr;
    realPathRet = realpath(path.data(), realPathMem);

    if (realPathRet == nullptr) {
        std::cout << "File: " << path << " is not exist.";
        return "";
    }

    std::string realPath(realPathMem);
    std::cout << path << " realpath is: " << realPath << std::endl;
    return realPath;
}

void Denorm(std::vector<MSTensor> *outputs) {
    for (size_t i = 0; i < outputs->size(); ++i) {
        size_t outputSize = (*outputs)[i].DataSize();
        float* netOutput = reinterpret_cast<float *>((*outputs)[i].MutableData());
        size_t outputLen = outputSize / sizeof(float);

        for (size_t j = 0; j < outputLen; ++j) {
            netOutput[j] = (netOutput[j] + 1) / 2 * 255;
            netOutput[j] = (netOutput[j] < 0) ? 0 : netOutput[j];
            netOutput[j] = (netOutput[j] > 255) ? 255 : netOutput[j];
        }
    }
}
