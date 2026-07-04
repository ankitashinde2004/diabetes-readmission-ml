# Healthcare Analytics: Predicting Hospital Readmissions for Diabetes Patients

An end-to-end Python machine learning pipeline to identify high-risk diabetic patients and predict 30-day hospital readmissions using a dataset of over 100,000 medical records.

## Project Highlights
* **Data Engineering & Cleaning:** Processed missing clinical data placeholders (`?` characters), eliminated low-variance features, and applied One-Hot Encoding to categorical healthcare attributes.
* **Class Imbalance Resolution:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the skewed target class, scaling minority samples from 9,000 to 71,195 records for stable model training.
* **Predictive Modeling:** Trained an optimized Random Forest Classifier to isolate key patient risk parameters.
* **Key Visual Insights:** Generated feature importance visualizations highlighting that previous inpatient stays, number of lab procedures, and time spent in the hospital are the primary drivers of readmission risk.

## Tech Stack Used
* **Environment:** Google Colab / Jupyter Notebook
* **Libraries:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn, Matplotlib, Seaborn
