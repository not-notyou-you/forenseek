Taruh model hasil notebook bagian 9 di sini:

- `model.keras` — model terlatih (Keras 3)
- `model_meta.json` — contoh:

```json
{"model_name": "xception", "threshold": 0.43, "gradcam_layer": "block14_sepconv2_act", "img_size": 224}
```

`model_name`: `xception` | `efficientnetb0` | `mesonet` (menentukan pra-pemrosesan).
Jika file > 100 MB, unggah ke Hugging Face / Google Drive (tautan unduh langsung) lalu isi
`MODEL_URL` dan `META_URL` di Settings → Secrets pada Streamlit Cloud.
