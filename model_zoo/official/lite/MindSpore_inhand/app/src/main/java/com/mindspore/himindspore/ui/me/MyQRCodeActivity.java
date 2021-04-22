package com.mindspore.himindspore.ui.me;

import android.content.ContentResolver;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.PopupWindow;
import android.widget.RelativeLayout;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.mindspore.himindspore.R;

@Route(path = "/app/MyQRCodeActivity")
public class MyQRCodeActivity extends AppCompatActivity {

    private static final String TAG = "MyQRCodeActivity";
    private PopupWindow popupWindow;
    private View save_and_cancels_image;
    private RelativeLayout mMy_layout;

    private ImageView mQr_code_image;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_my_qr_code);
        init();
    }

    private void init() {
        Toolbar mToolbar = findViewById(R.id.segmentation_toolbar);
        setSupportActionBar(mToolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());
        mMy_layout = (RelativeLayout) findViewById(R.id.my_relativelayout);
        mQr_code_image = findViewById(R.id.img_origin);

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "onCreateOptionsMenu info");
        getMenuInflater().inflate(R.menu.menu_setting_qr, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        findViewById(R.id.item_more_qr).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                LayoutInflater layoutInflater = (LayoutInflater) getSystemService(Context.LAYOUT_INFLATER_SERVICE);
                save_and_cancels_image = layoutInflater.inflate(R.layout.save_qr_code_popupwindo, null);
                DisplayMetrics dm = new DisplayMetrics();
                getWindowManager().getDefaultDisplay().getMetrics(dm);
                popupWindow = new PopupWindow(save_and_cancels_image, dm.widthPixels, WindowManager.LayoutParams.WRAP_CONTENT);
                popupWindow.setAnimationStyle(R.style.anims);
                popupWindow.setFocusable(true);
                popupWindow.setOutsideTouchable(true);
                popupWindow.setBackgroundDrawable(new BitmapDrawable());
                backgroundAlpha(0.5f);
                popupWindow.setOnDismissListener(new PopupWindow.OnDismissListener() {
                    @Override
                    public void onDismiss() {
                        backgroundAlpha(1f);

                    }
                });
                popupWindow.showAtLocation(mMy_layout, Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL, 0, 0);
                save_and_cancels_image.findViewById(R.id.my_save_picture).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        SaveImageToSysAlbum();
                    }
                });
                save_and_cancels_image.findViewById(R.id.my_cancel).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        popupWindow.dismiss();
                    }
                });
            }
        });

        return super.onOptionsItemSelected(item);
    }

    public void backgroundAlpha(float bgAlpha) {
        WindowManager.LayoutParams lp = getWindow().getAttributes();
        lp.alpha = bgAlpha;
        getWindow().setAttributes(lp);
    }

    private void SaveImageToSysAlbum() {
        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)) {
            mQr_code_image.setDrawingCacheEnabled(true);
            mQr_code_image.buildDrawingCache();
            Bitmap bitmap = Bitmap.createBitmap(mQr_code_image.getDrawingCache());
            if (bitmap != null) {
                try {
                    ContentResolver cr = getContentResolver();
                    String url = MediaStore.Images.Media.insertImage(cr, bitmap,
                            String.valueOf(System.currentTimeMillis()), "");
                    Toast.makeText(this.getApplicationContext(), R.string.save_success, Toast.LENGTH_SHORT).show();
                    popupWindow.dismiss();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            } else {
                Toast.makeText(this.getApplicationContext(), R.string.save_failure, Toast.LENGTH_SHORT).show();
            }
        } else {
            Toast.makeText(this.getApplicationContext(), R.string.save_success, Toast.LENGTH_SHORT).show();
        }
    }
}