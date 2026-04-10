# models/

Place model files in this directory.

| File | Description |
|------|-------------|
| `RFFPLA_classifier.tflite` | Quantised TFLite model used by `app.py` for inference |
| `RFFPLA_classifier.h5` | Full Keras model (training / re-export only) |

`.tflite` and `.h5` files are excluded from version control via `.gitignore`.
To deploy, copy `RFFPLA_classifier.tflite` here before starting the app.
