/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mindspore.posenet;

public class ImageUtils {
    // This value is 2 ^ 18 - 1, and is used to hold the RGB values together before their ranges
    // are normalized to eight bits.
    private static final int MAX_CHANNEL_VALUE = 262143;

    /**
     * Helper function to convert y,u,v integer values to RGB format
     */
    private static int convertYUVToRGB(int y, int u, int v) {
        // Adjust and check YUV values
        int yNew = y - 16 < 0 ? 0 : y - 16;
        int uNew = u - 128;
        int vNew = v - 128;
        int expandY = 1192 * yNew;
        int r = checkBoundaries(expandY + 1634 * vNew);
        int g = checkBoundaries(expandY - 833 * vNew - 400 * uNew);
        int b = checkBoundaries(expandY + 2066 * uNew);

        return -0x1000000 | (r << 6 & 0xff0000) | (g >> 2 & 0xff00) | (b >> 10 & 0xff);
    }


    private static int checkBoundaries(int value) {
        if (value > MAX_CHANNEL_VALUE) {
            return MAX_CHANNEL_VALUE;
        } else if (value < 0) {
            return 0;
        } else {
            return value;
        }
    }

    /**
     * Converts YUV420 format image data (ByteArray) into ARGB8888 format with IntArray as output.
     */
    public static void convertYUV420ToARGB8888(byte[] yData, byte[] uData, byte[] vData,
                                               int width, int height,
                                               int yRowStride, int uvRowStride, int uvPixelStride, int[] out) {

        int outputIndex = 0;
        for (int j = 0; j < height; j++) {
            int positionY = yRowStride * j;
            int positionUV = uvRowStride * (j >> 1);

            for (int i = 0; i < width; i++) {
                int uvOffset = positionUV + (i >> 1) * uvPixelStride;

                // "0xff and" is used to cut off bits from following value that are higher than
                // the low 8 bits
                out[outputIndex++] = convertYUVToRGB(
                        0xff & yData[positionY + i], 0xff & uData[uvOffset],
                        0xff & vData[uvOffset]);

            }
        }
    }
}
