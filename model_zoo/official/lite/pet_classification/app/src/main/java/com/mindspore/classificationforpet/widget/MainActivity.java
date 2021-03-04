package com.mindspore.classificationforpet.widget;

import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.Settings;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.UiThread;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.mindspore.classificationforpet.R;
import com.mindspore.classificationforpet.gallery.classify.BitmapUtils;
import com.mindspore.classificationforpet.gallery.classify.ImageTrackingMobile;
import com.mindspore.classificationforpet.gallery.classify.RecognitionImageBean;
import com.mindspore.classificationforpet.gallery.classify.TrackingMobile;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity implements OnBackgroundImageListener {

    private static final String TAG = "MainActivity";

    private static final String[] PERMISSIONS = {Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_PHONE_STATE, Manifest.permission.CAMERA};
    private static final int REQUEST_PERMISSION = 0;

    private static final int[] IMAGES = {R.drawable.style4, R.drawable.style8, R.drawable.style1, R.drawable.style6, R.drawable.style3,
            R.drawable.style7, R.drawable.style5, R.drawable.style9, R.drawable.style0, R.drawable.style2};

    private static final int RC_CHOOSE_PHOTO = 1;
    private static final int RC_CHOOSE_CAMERA = 2;
    private static final String IMAGE_SCENE_MS = "model/mobilenetv2.ms";

    private boolean isAllGranted;
    private static final String Pet_MS = "pet.ms";
    private File ROOT_FILE = new File(Environment.getExternalStorageDirectory().getAbsoluteFile(), "PetClassification");
    private File DIR_FILE = new File(ROOT_FILE, Pet_MS);

    private ImageView imgPreview;
    private Uri imageUri;
    private TextView textResult;
    private ProgressBar progressBar;
    private RecyclerView recyclerView;
    private TrackingMobile trackingMobile;
    private ImageTrackingMobile imageTrackingMobile;
    private List<RecognitionImageBean> recognitionObjectBeanList;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        init();
        requestPermissions();
    }

    private void init() {
        imgPreview = findViewById(R.id.img_origin);
        textResult = findViewById(R.id.tv_image);
        progressBar = findViewById(R.id.progress);
        recyclerView = findViewById(R.id.recyclerview);
        recyclerView.setLayoutManager(new GridLayoutManager(this, 3));
        recyclerView.setAdapter(new RecyclerViewAdapter(this, IMAGES, this));
        trackingMobile = new TrackingMobile(this);
        imageTrackingMobile = new ImageTrackingMobile(this);
    }

    private void requestPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            isAllGranted = checkPermissionAllGranted(PERMISSIONS);
            if (!isAllGranted) {
                ActivityCompat.requestPermissions(this, PERMISSIONS, REQUEST_PERMISSION);
            }
        } else {
            isAllGranted = true;
        }
    }

    private boolean checkPermissionAllGranted(String[] permissions) {
        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    /**
     * Authority application result callback
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (REQUEST_PERMISSION == requestCode) {
            isAllGranted = true;
            for (int grant : grantResults) {
                if (grant != PackageManager.PERMISSION_GRANTED) {
                    isAllGranted = false;
                    break;
                }
            }
            if (!isAllGranted) {
                openAppDetails();
            }
        }
    }

    private void openAppDetails() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setMessage(getString(R.string.app_choose_authority));
        builder.setPositiveButton(getString(R.string.app_choose_authority_manual), (dialog, which) -> {
            Intent intent = new Intent();
            intent.setAction(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
            intent.addCategory(Intent.CATEGORY_DEFAULT);
            intent.setData(Uri.parse("package:" + getPackageName()));
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            intent.addFlags(Intent.FLAG_ACTIVITY_NO_HISTORY);
            intent.addFlags(Intent.FLAG_ACTIVITY_EXCLUDE_FROM_RECENTS);
            startActivity(intent);
        });
        builder.setNegativeButton(getString(R.string.app_choose_cancle), null);
        builder.show();
    }

    public boolean isHasPetModelFile() {
        if (DIR_FILE.exists()) {
            return true;
        } else {
            if (!ROOT_FILE.exists()) {
                ROOT_FILE.mkdirs();
            }
            return false;
        }
    }

    public void onClickPhoto(View view) {
        if (isAllGranted) {
            openGallay();
        } else {
            requestPermissions();
        }
    }

    public void onClickCamera(View view) {
        if (isAllGranted) {
            openCamera();
        } else {
            requestPermissions();
        }
    }

    public void onClickScene(View view) {
        Intent intent = new Intent(MainActivity.this, CameraActivity.class);
        intent.putExtra("FILEPATH", DIR_FILE.getPath());
        intent.putExtra("ISHASPetMODELFILE", isHasPetModelFile());
        startActivity(intent);
    }

    @Override
    public void onBackImageSelected(int position) {
        imgPreview.setImageResource(IMAGES[position]);
        initMindspore(BitmapFactory.decodeResource(getResources(), IMAGES[position]));
    }

    private void openGallay() {
        Intent intentToPickPic = new Intent(Intent.ACTION_PICK, null);
        intentToPickPic.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intentToPickPic, RC_CHOOSE_PHOTO);
    }

    private void openCamera() {
        Intent intentToTakePhoto = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        String mTempPhotoPath = Environment.getExternalStorageDirectory() + File.separator + "photo22.jpeg";
        imageUri = FileProvider.getUriForFile(this, getApplicationContext().getPackageName() + ".fileprovider", new File(mTempPhotoPath));
        intentToTakePhoto.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
        startActivityForResult(intentToTakePhoto, RC_CHOOSE_CAMERA);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (RC_CHOOSE_PHOTO == requestCode) {
                if (null != data && null != data.getData()) {
                    this.imageUri = data.getData();
                    showOriginImage();
                } else {
                    finish();
                }
            } else if (RC_CHOOSE_CAMERA == requestCode) {
                showOriginCamera();
            }
        }
    }

    private void showOriginImage() {
        File file = BitmapUtils.getFileFromMediaUri(this, imageUri);
        Bitmap photoBmp = BitmapUtils.getBitmapFormUri(this, Uri.fromFile(file));
        int degree = BitmapUtils.getBitmapDegree(file.getAbsolutePath());
        Bitmap originBitmap = BitmapUtils.rotateBitmapByDegree(photoBmp, degree);
        if (originBitmap != null) {
            imgPreview.setImageBitmap(originBitmap);
            initMindspore(originBitmap.copy(Bitmap.Config.ARGB_8888, true));
        } else {
            Toast.makeText(this, R.string.image_invalid, Toast.LENGTH_LONG).show();
        }
    }

    private void showOriginCamera() {
        try {
            Bitmap originBitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
            if (originBitmap != null) {
                imgPreview.setImageBitmap(originBitmap);
                initMindspore(originBitmap.copy(Bitmap.Config.ARGB_8888, true));
            } else {
                Toast.makeText(this, R.string.image_invalid, Toast.LENGTH_LONG).show();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private void initMindspore(Bitmap bitmap) {
        progressBar.setVisibility(View.VISIBLE);

        if (isHasPetModelFile()) {
            boolean ret = trackingMobile.loadModelFromBuf(DIR_FILE.getPath());
            if (!ret) {
                textResult.setText("Load model error.");
                Log.e(TAG, "Load model error.");
                return;
            }
            // run net.
            long startTime = System.currentTimeMillis();
            String result = trackingMobile.MindSpore_runnet(bitmap);
            long endTime = System.currentTimeMillis();
            String[] IMAGECONTENT = getResources().getStringArray(R.array.image_category_pet);
            int nameIndex = Integer.parseInt(result);
            progressBar.setVisibility(View.GONE);
            textResult.setText(IMAGECONTENT[nameIndex]);
            Log.d(TAG, "RUNNET CONSUMING：" + (endTime - startTime) + "ms");
            Log.d(TAG, "result：" + result);
        } else {
            if (recognitionObjectBeanList != null) {
                recognitionObjectBeanList.clear();
            } else {
                recognitionObjectBeanList = new ArrayList<>();
            }

            boolean ret = imageTrackingMobile.loadModelFromBuf(IMAGE_SCENE_MS);
            if (!ret) {
                textResult.setText("Load model error.");
                Log.e(TAG, "Load model error.");
                return;
            }
            // run net.
            long startTime = System.currentTimeMillis();
            String result = imageTrackingMobile.MindSpore_runnet(bitmap);
            long endTime = System.currentTimeMillis();
            progressBar.setVisibility(View.GONE);
            Log.d(TAG, "RUNNET CONSUMING：" + (endTime - startTime) + "ms");
            Log.d(TAG, "result：" + result);
            String[] IMAGECONTENT = getResources().getStringArray(R.array.image_category);

            if (!TextUtils.isEmpty(result)) {
                String[] resultArray = result.split(";");
                for (String singleRecognitionResult : resultArray) {
                    String[] singleResult = singleRecognitionResult.split(":");
                    int nameIndex = Integer.parseInt(singleResult[0]);
                    float score = Float.parseFloat(singleResult[1]);
                    if (score > 0.5) {
                        recognitionObjectBeanList.add(new RecognitionImageBean(IMAGECONTENT[nameIndex], score));
                    }
                }
                Collections.sort(recognitionObjectBeanList, (t1, t2) -> Float.compare(t2.getScore(), t1.getScore()));
                showResultsInBottomSheet(recognitionObjectBeanList, (endTime - startTime) + "ms");
            }
        }
        if (!bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }

    @UiThread
    protected void showResultsInBottomSheet(List<RecognitionImageBean> list, String time) {
        if (list == null || list.size() < 1) {
            return;
        }
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < list.size(); i++) {
            RecognitionImageBean bean = list.get(i);
            stringBuilder.append(bean.getName()).append("\r:\r").append(String.format("%.2f", (100 * bean.getScore())) + "%").append("\r\n");
            if (i > 3) { // set maximum display is 3.
                break;
            }
        }
        textResult.setText(stringBuilder);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (trackingMobile != null) {
            trackingMobile.unloadModel();
        }
        if (imageTrackingMobile != null) {
            imageTrackingMobile.unloadModel();
        }
    }


}