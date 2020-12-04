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
package com.mindspore.imagesegmentation.help;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.media.ExifInterface;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class BitmapUtils {
    private static final String TAG = "BitmapUtils";

    public static void recycleBitmap(Bitmap... bitmaps) {
        for (Bitmap bitmap : bitmaps) {
            if (bitmap != null && !bitmap.isRecycled()) {
                bitmap.recycle();
                bitmap = null;
            }
        }
    }

    private static String getImagePath(Activity activity, Uri uri) {
        String[] projection = {MediaStore.Images.Media.DATA};
        Cursor cursor = activity.managedQuery(uri, projection, null, null, null);
        int columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
        cursor.moveToFirst();
        return cursor.getString(columnIndex);
    }

    public static Bitmap loadFromPath(Activity activity, int id, int width, int height) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        InputStream is = activity.getResources().openRawResource(id);
        int sampleSize = calculateInSampleSize(options, width, height);
        options.inSampleSize = sampleSize;
        options.inJustDecodeBounds = false;
        return zoomImage(BitmapFactory.decodeStream(is), width, height);
    }

    public static Bitmap loadFromPath(Activity activity, Uri uri, int width, int height) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;

        String path = getImagePath(activity, uri);
        BitmapFactory.decodeFile(path, options);
        int sampleSize = calculateInSampleSize(options, width, height);
        options.inSampleSize = sampleSize;
        options.inJustDecodeBounds = false;

        Bitmap bitmap = zoomImage(BitmapFactory.decodeFile(path, options), width, height);
        return rotateBitmap(bitmap, getRotationAngle(path));
    }

    private static int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
        final int width = options.outWidth;
        final int height = options.outHeight;
        int inSampleSize = 1;

        if (height > reqHeight || width > reqWidth) {
            // Calculate height and required height scale.
            final int heightRatio = Math.round((float) height / (float) reqHeight);
            // Calculate width and required width scale.
            final int widthRatio = Math.round((float) width / (float) reqWidth);
            // Take the larger of the values.
            inSampleSize = heightRatio > widthRatio ? heightRatio : widthRatio;
        }
        return inSampleSize;
    }

    // Scale pictures to screen width.
    private static Bitmap zoomImage(Bitmap imageBitmap, int targetWidth, int maxHeight) {
        float scaleFactor =
                Math.max(
                        (float) imageBitmap.getWidth() / (float) targetWidth,
                        (float) imageBitmap.getHeight() / (float) maxHeight);
        Bitmap resizedBitmap =
                Bitmap.createScaledBitmap(
                        imageBitmap,
                        (int) (imageBitmap.getWidth() / scaleFactor),
                        (int) (imageBitmap.getHeight() / scaleFactor),
                        true);

        return resizedBitmap;
    }

    /**
     * Get the rotation angle of the photo.
     *
     * @param path photo path.
     * @return angle.
     */
    public static int getRotationAngle(String path) {
        int rotation = 0;
        try {
            ExifInterface exifInterface = new ExifInterface(path);
            int orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    rotation = 90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    rotation = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    rotation = 270;
                    break;
                default:
                    break;
            }
        } catch (IOException e) {
            Log.e(TAG, "Failed to get rotation: " + e.getMessage());
        }
        return rotation;
    }

    public static Bitmap rotateBitmap(Bitmap bitmap, int angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        Bitmap result = null;
        try {
            result = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        } catch (OutOfMemoryError e) {
            Log.e(TAG, "Failed to rotate bitmap: " + e.getMessage());
        }
        if (result == null) {
            return bitmap;
        }
        return result;
    }

    private static Matrix decodeExifOrientation(int orientation) {
        Matrix matrix = new Matrix();

        switch (orientation) {
            case androidx.exifinterface.media.ExifInterface.ORIENTATION_NORMAL:
            case androidx.exifinterface.media.ExifInterface.ORIENTATION_UNDEFINED:
                break;
            case androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_90:
                matrix.postRotate(90F);
                break;
            case androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_180:
                matrix.postRotate(180F);
                break;
            case androidx.exifinterface.media.ExifInterface.ORIENTATION_ROTATE_270:
                matrix.postRotate(270F);
                break;
            case androidx.exifinterface.media.ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                matrix.postScale(-1F, 1F);
                break;
            case androidx.exifinterface.media.ExifInterface.ORIENTATION_FLIP_VERTICAL:
                matrix.postScale(1F, -1F);
                break;
            case androidx.exifinterface.media.ExifInterface.ORIENTATION_TRANSPOSE:
                matrix.postScale(-1F, 1F);
                matrix.postRotate(270F);
                break;
            case androidx.exifinterface.media.ExifInterface.ORIENTATION_TRANSVERSE:
                matrix.postScale(-1F, 1F);
                matrix.postRotate(90F);
                break;

            default:
                try {
                    new IllegalArgumentException("Invalid orientation: " + orientation);
                } catch (Throwable throwable) {
                    throwable.printStackTrace();
                }
        }
        return matrix;
    }

    public static Bitmap scaleBitmapAndKeepRatio(Bitmap targetBmp, int reqHeightInPixels, int reqWidthInPixels) {
        if (targetBmp.getHeight() == reqHeightInPixels && targetBmp.getWidth() == reqWidthInPixels) {
            return targetBmp;
        }

        Matrix matrix = new Matrix();
        matrix.setRectToRect(new RectF(0f, 0f,
                targetBmp.getWidth(),
                targetBmp.getHeight()
        ), new RectF(0f, 0f,
                reqWidthInPixels,
                reqHeightInPixels
        ), Matrix.ScaleToFit.FILL);

        return Bitmap.createBitmap(
                targetBmp, 0, 0,
                targetBmp.getWidth(),
                targetBmp.getHeight(), matrix, true
        );
    }

    public static ByteBuffer bitmapToByteBuffer(Bitmap bitmapIn, int width, int height, float mean, float std) {
        Bitmap bitmap = scaleBitmapAndKeepRatio(bitmapIn, width, height);
        ByteBuffer inputImage = ByteBuffer.allocateDirect(1 * width * height * 3 * 4);
        inputImage.order(ByteOrder.nativeOrder());
        inputImage.rewind();
        int[] intValues = new int[width * height];
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height);
        int pixel = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int value = intValues[pixel++];
                inputImage.putFloat(((float) (value >> 16 & 255) - mean) / std);
                inputImage.putFloat(((float) (value >> 8 & 255) - mean) / std);
                inputImage.putFloat(((float) (value & 255) - mean) / std);
            }
        }
        inputImage.rewind();
        return inputImage;
    }

    // Save the picture to the system album and refresh it.
    public static void saveToAlbum(final Context context, Bitmap bitmap) {
        File file = null;
        String fileName = System.currentTimeMillis() + ".jpg";
        File root = new File(Environment.getExternalStorageDirectory().getAbsoluteFile(), context.getPackageName());
        File dir = new File(root, "image");
        if (dir.mkdirs() || dir.isDirectory()) {
            file = new File(dir, fileName);
        }
        FileOutputStream os = null;
        try {
            os = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, os);
            os.flush();

        } catch (FileNotFoundException e) {
            Log.e(TAG, e.getMessage());
        } catch (IOException e) {
            Log.e(TAG, e.getMessage());
        } finally {
            try {
                if (os != null) {
                    os.close();
                }
            } catch (IOException e) {
                Log.e(TAG, e.getMessage());
            }
        }
        if (file == null) {
            return;
        }
        // Gallery refresh.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            String path = null;
            try {
                path = file.getCanonicalPath();
            } catch (IOException e) {
                Log.e(TAG, e.getMessage());
            }
            MediaScannerConnection.scanFile(context, new String[]{path}, null,
                    new MediaScannerConnection.OnScanCompletedListener() {
                        @Override
                        public void onScanCompleted(String path, Uri uri) {
                            Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
                            mediaScanIntent.setData(uri);
                            context.sendBroadcast(mediaScanIntent);
                        }
                    });
        } else {
            String relationDir = file.getParent();
            File file1 = new File(relationDir);
            context.sendBroadcast(new Intent(Intent.ACTION_MEDIA_MOUNTED, Uri.fromFile(file1.getAbsoluteFile())));
        }
    }
}
