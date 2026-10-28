# Architecture
Modular monolith. `pages/*.py` only render widgets and call `src/`. `src/ui.py` is the only `src` module importing Streamlit.

Data flow: bytes → `data_validation.load_csv_bytes` (`raw` strings + `clean` typed frame, kept separate) → `data_profile` → `train.train_model` (fit/cal/test splits + sklearn `Pipeline`) → `evaluate`, `explain`, `calibration`, `uncertainty` consume the `TrainResult`. `drift` works on any two cleaned frames. `student_demo` is independent of the Model Lab.

State: `st.session_state` holds `df`, `df_fp` (content hash), `result`, `model_data_fp`. `set_dataset` clears derived results; `require_model` refuses a model trained on a different dataset. Nothing is cached across sessions and nothing is written to disk.

Decisions: csv-module parsing (strict field counts); three-way split so calibration/conformal/OOD thresholds never touch test labels; model-agnostic explanations aligned to original columns.
