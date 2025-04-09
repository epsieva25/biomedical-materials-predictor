# AI Techniques for Predicting Properties of Sustainable Materials for Biomedical Applications

**Overview:** This project applies AI-based regression models—Support Vector Regression (SVR) and Decision Tree Regression—to predict the bio-compatibility score of sustainable materials used in biomedical applications. It enables intelligent selection of materials for implants through accurate prediction of material behavior using machine learning.

**Features:** The system includes synthetic dataset generation, training and evaluation of SVR and Decision Tree models, a clean Streamlit-based web interface, real-time prediction based on user inputs, and persistent model storage using Joblib.

**Project Structure:** The repository includes: `app/app.py` for the Streamlit frontend, `backend/model_train.py` for data creation and model training, `data/sustainable_materials.csv` containing synthetic data, `models/` folder with saved models, and `utils/helper.py` for additional functions.

**Technologies Used:** Python, NumPy, Pandas, Scikit-learn, Streamlit, Joblib.

**Input Features:** The model accepts `biodegradable_index` (0.2–0.9), `tensile_strength` (30–120 MPa), `porosity` (0.1–0.6), and `water_absorption` (1–10%) to predict a bio-compatibility score on a scale of 0–100.

**How to Use:** 1. Clone the repo using `git clone https://github.com/epsieva25/biomedical-materials-predictor.git` and navigate to the folder. 2. Install dependencies using `pip install -r requirements.txt`. 3. Run the training script with `python backend/model_train.py` to generate data and train models. 4. Launch the app by navigating to the `app` folder and executing `streamlit run app.py`.

**Highlights:** This tool enables rapid prediction of sustainable biomedical material properties, features dual-model comparison, and provides an intuitive UI for research and educational use. The modular design allows future integration with real datasets and deployment options.

**Future Improvements:** Plans include incorporating real-world material datasets, adding explainable AI features like SHAP, deploying on Streamlit Cloud or AWS, and enhancing the UI with material visualizations.

**Contributing:** Contributions are welcome—fork the repository, create a feature branch, commit your changes, push, and submit a Pull Request.

**Author:** Developed by Mary Jasper Epsibha R ([GitHub](https://github.com/epsieva25)).


