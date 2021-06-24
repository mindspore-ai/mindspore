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

package com.mindspore.flclient.cipher;

import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.List;
import java.util.logging.Logger;

public class Random {
    /**
     * random generate RNG algorithm name
     */
    private static final Logger LOGGER = Logger.getLogger(Random.class.toString());
    private static final String RNG_ALGORITHM = "SHA1PRNG";

    private static final int RANDOM_LEN = 128 / 8;

    public void getRandomBytes(byte[] secret) {
        try {
            SecureRandom secureRandom = SecureRandom.getInstance("SHA1PRNG");
            secureRandom.nextBytes(secret);
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
    }

    public void randomAESCTR(List<Float> noise, int length, byte[] seed) throws Exception {
        int intV = Integer.SIZE / 8;
        int size = length * intV;
        byte[] data = new byte[size];
        for (int i = 0; i < size; i++) {
            data[i] = 0;
        }
        byte[] ivec = new byte[RANDOM_LEN];
        AESEncrypt aesEncrypt = new AESEncrypt(seed, ivec, "CTR");
        byte[] encryptCtr = aesEncrypt.encryptCTR(seed, data);
        for (int i = 0; i < length; i++) {
            int[] sub = new int[intV];
            for (int j = 0; j < 4; j++) {
                sub[j] = (int) encryptCtr[i * intV + j] & 0xff;
            }
            int subI = byte2int(sub, 4);

            Float f = Float.valueOf(Float.valueOf(subI) / Integer.MAX_VALUE);
            noise.add(f);
        }
    }

    public static int byte2int(int[] data, int n) {
        switch (n) {
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


