# Student Performance Predictor
Attendance + test-score trends use panni at-risk students-a early-ah flag pannum ML project (Python, Pandas, Scikit-learn).

## Run steps
```bash
pip install -r requirements.txt
python generate_data.py     # sample data (skip pannalam, own data irundha)
python train.py             # model train + evaluate
python predict.py           # at-risk list -> outputs/at_risk_students.csv
```

## Own data format (CSV)
`student_id, attendance_w1..attendance_wN, test1..testN, assignment_completion, at_risk (train-kku mattum, 1/0)`

## Dashboard (Streamlit + Matplotlib)
```bash
streamlit run app.py
```
Browser-la http://localhost:8501 open aagum. Overview charts, student-wise detail, at-risk list + CSV download irukku.

## Jupyter Notebook
```bash
pip install -r requirements.txt
jupyter notebook
```
Open `Student_Performance_Predictor.ipynb` and run the cells top to bottom (Shift + Enter).
