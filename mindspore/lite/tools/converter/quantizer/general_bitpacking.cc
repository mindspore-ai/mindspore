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

#include "tools/converter/quantizer/general_bitpacking.h"

namespace mindspore {
namespace lite {
BitPack::BitPack(const uint8_t& bitnum) {this->bitnum = bitnum;}
void BitPack::UnPackFromUint8ToOrigin(uint8_t& n, std::queue<bool>& unpackBitData) {
    int bitCount = 0;
    while (bitCount < 8) {
        bool a = n % 2;
        n = n >> 1;
        bitCount++;
        unpackBitData.push(a);
    }
}
void BitPack::UnPack(uint8_t bitnum, uint8_t& packedData,
                     std::vector<uint8_t> &originData, std::queue<bool>& unpackBitData) {
    UnPackFromUint8ToOrigin(packedData, unpackBitData);
    // std::queue<bool> unpackBitTmpData;

    while (unpackBitData.size() > bitnum) {
        uint32_t result = 0;
        for (int k = 0; k < bitnum; k++) {
            bool bitTmp = unpackBitData.front();
            result = (result << 1) + static_cast<int>(bitTmp);
            unpackBitData.pop();
        }
        originData.push_back(result);
    }
}
void BitPack::PackFromOriginToUint8(std::stack<bool>& ans, std::vector<uint8_t>& packedDataVec) {
    uint32_t result = 0;
    for (size_t i = 0; i < 8; i++) {
        bool bit_tmp = ans.top();
        result = (result << 1) + static_cast<int>(bit_tmp);
        ans.pop();
    }
    packedDataVec.push_back(result);
}
void BitPack::DoBinary(uint8_t& n, std::stack<bool>& ans, std::vector<uint8_t>& packedDataVec) {
    int bitCount = 0;
    while (bitCount < bitnum) {
        bool a = n / (1 << (unsigned int)(bitnum - bitCount - 1));
        n = n - a * (1 << (unsigned int)(bitnum - bitCount - 1));
        bitCount++;
        ans.push(a);
        if (ans.size() == 8) {
            PackFromOriginToUint8(ans, packedDataVec);
        }
    }
}

void BitPack::BitPacking(const std::vector<uint8_t>& originDataVec, std::vector<uint8_t>& packedDataVec) {
    std::stack<bool> bitDataVec;
    for (size_t i = 0; i < originDataVec.size(); i++) {
        uint8_t tmp = originDataVec[i];
        DoBinary(tmp, bitDataVec, packedDataVec);
    }

    size_t remainBitData = bitDataVec.size();
    if ( 8 > remainBitData && remainBitData > 0 ) {
        for ( int i = 0; i < 8 - remainBitData; i++ ) {
            bitDataVec.push(0);
        }
        PackFromOriginToUint8(bitDataVec, packedDataVec);
    }
}

}  // namespace lite
}  // namespace mindspore

