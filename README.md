# Forenseek

Screens face images and short videos for deepfake manipulation, returning a probability score and a Grad-CAM heatmap of the regions that drove the decision, to help fact-checkers decide what needs verification.

Prototipe skripsi — Arie Pratama Yuliansyah, Sistem Informasi UMN.

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud
1. Push folder ini sebagai repositori GitHub (mis. `veritas-lens`).
2. share.streamlit.io → **Create app** → pilih repo, branch `main`, file `app.py`.
3. **Advanced settings → Python 3.12** (wajib; TensorFlow belum mendukung 3.14). Versi Python tidak bisa diubah setelah deploy — hapus app lalu deploy ulang.
4. Model: commit `models/model.keras` + `models/model_meta.json`, atau isi Secrets:
   ```toml
   MODEL_URL = "https://.../model.keras"
   META_URL  = "https://.../model_meta.json"
   ```
Tanpa model, aplikasi tetap berjalan dan menampilkan status "Model belum tersedia".

## Catatan
- Deteksi wajah: OpenCV Haar cascade; jika wajah tak terdeteksi, seluruh citra dipakai.
- Skor video = rata-rata probabilitas frame (Pers. 3.8).
- Berkas unggahan tidak disimpan. Dataset FaceForensics++ berlisensi nonkomersial.
