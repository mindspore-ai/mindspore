/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

package com.mindspore.flclient.cipher;

import com.mindspore.flclient.Common;

import java.security.SecureRandom;
import java.util.List;
import java.util.logging.Logger;

/**
 * Define the basic method for generating pairwise mask and individual mask.
 *
 * @since 2021-06-30
 */
public class Masking {
    private static final Logger LOGGER = Logger.getLogger(Masking.class.toString());

    /**
     * Random generate RNG algorithm name.
     */
    private static final String RNG_ALGORITHM = "SHA1PRNG";

    /**
     * Generate individual mask.
     *
     * @param secret used to store individual mask.
     * @return the int value, 0 indicates success, -1 indicates failed .
     */
    public int getRandomBytes(byte[] secret) {
        if (secret == null || secret.length == 0) {
            LOGGER.severe(Common.addTag("[Masking] the input argument <secret> is null, please check!"));
            return -1;
        }
        SecureRandom secureRandom = Common.getSecureRandom();
        secureRandom.nextBytes(secret);
        return 0;
    }

    /**
     * Generate pairwise mask.
     *
     * @param noise  used to store pairwise mask.
     * @param length the length of individual mask.
     * @param seed   the seed for generate pairwise mask.
     * @param iVec   the IV value.
     * @return the int value, 0 indicates success, -1 indicates failed .
     */
    public int getMasking(List<Float> noise, int length, byte[] seed, byte[] iVec) {
        if (length <= 0) {
            LOGGER.severe(Common.addTag("[Masking] the input argument <length> is not valid: <= 0, please check!"));
            return -1;
        }
        int intV = Integer.SIZE / 8;
        int size = length * intV;
        byte[] data = new byte[size];
        for (int i = 0; i < size; i++) {
            data[i] = 0;
        }
        AESEncrypt aesEncrypt = new AESEncrypt(seed, "CTR");
        byte[] encryptCtr = aesEncrypt.encryptCTR(seed, data, iVec);
        if (encryptCtr == null || encryptCtr.length == 0) {
            LOGGER.severe(Common.addTag("[Masking] the return byte[] is null, please check!"));
            return -1;
        }
        for (int i = 0; i < length; i++) {
            int[] sub = new int[intV];
            for (int j = 0; j < 4; j++) {
                sub[j] = (int) encryptCtr[i * intV + j] & 0xff;
            }
            int subI = byte2int(sub, 4);
            if (subI == -1) {
                LOGGER.severe(Common.addTag("[Masking] the the returned <subI> is not valid: -1, please check!"));
                return -1;
            }
            Float fNoise = Float.valueOf(Float.valueOf(subI) / Integer.MAX_VALUE);
            noise.add(fNoise);
        }
        return 0;
    }

    private static int byte2int(int[] data, int number) {
        if (data.length < 4) {
            LOGGER.severe(Common.addTag("[Masking] the input argument <data> is not valid: length < 4, please check!"));
            return -1;
        }
        switch (number) {
            case 1:
                return (int) data[0];
            case 2:
                return (int) (data[0] & 0xff) | (data[1] << 8 & 0xff00);
            case 3:
                return (int) (data[0] & 0xff) | (data[1] << 8 & 0xff00) | (data[2] << 16 & 0xff0000);
            case 4:
                return (int) (data[0] & 0xff) | (data[1] << 8 & 0xff00) | (data[2] << 16 & 0xff0000)
                        | (data[3] << 24 & 0xff000000);
            default:
                return 0;
        }
    }
}