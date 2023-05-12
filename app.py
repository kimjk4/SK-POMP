#Core Pkgs
import streamlit as st

#EDA Pkgs
import pandas as pd
import numpy as np
import math

#Utils
import os
import joblib
import hashlib

#passlib,bcrypt

#Data Viz Pckgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

feature_names_best = ['age_baseline', 'gender', 'prenatal_hn', 'laterality', 'length_1_1',
       'apd_us1', 'sfu_us1', 'max_dilation1']

Gender_dict = {"Male":1, "Female":2}
Laterality_dict = {"Left":1, "Right":2, "Bilateral":3}
feature_dict = {"Yes":1, "No":0}

def get_value(val,my_dict):
    return new_func(val, my_dict)

def new_func(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def get_key(val):
    for key, value in my_dict.items():
        if val == key:
            return value
        

# Load ML models        
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

stats_container = st.container()	
header_container = st.container()


about_container1 = st.container()
about_container2 = st.container()	
about_container3 = st.container()
about_container4 = st.container()
about_container5 = st.container()
about_container6 = st.container()
about_container7 = st.container()	
about_container8 = st.container()
about_container9 = st.container()
about_container10 = st.container()

col_1, col_2, col_3 = st.columns(3)

col_4, col_5, col_6 = st.columns(3)

col_7, col_8 = st.columns(2)
col_9, col_10 = st.columns(2)
col_11, col_12 = st.columns(2)
col_13, col_14 = st.columns(2)

col_15, col_16 = st.columns(2)
col_17, col_18 = st.columns(2)
col_19, col_20 = st.columns(2)
col_21, col_22 = st.columns(2)

with stats_container:
    def main():
        
        activity = st.selectbox ('Select Activity', ['SK-POMP: Prediction for obstruction', 'SK-PNMS: Prediction for surgical intervention', 'About SK-POMP', 'About SK-PNMS'])
        if activity == 'SK-POMP: Prediction for obstruction':
            
            with col_1:
            
                st.subheader ('Clinical Characteristics')
                age_baseline = st.number_input ("Age (days)", 0, 30)
                gender = st.radio ("Sex", tuple(Gender_dict.keys()))
                laterality = st.radio ("Side of hydronephrosis", tuple(Laterality_dict.keys()))
                st.caption('If bilateral, ultrasound findings should be focused on the side with worse hydroureter.')

        
            with col_2:

                st.subheader ('Ultrasound findings')
                length_1_1 = st.number_input ("Length (mm)", 0, 110)
                apd_us1 = st.number_input ("AP diameter (mm)", 0, 45)
                sfu_us1 = st.number_input ("SFU Grading", 0, 4)
                max_dilation1 = st.number_input ("maximum ureter diameter (mm)", 0, 50)
            
            with col_3:

                feature_list = [age_baseline, get_value(gender, Gender_dict),  get_value(laterality, Laterality_dict), length_1_1, 
                                apd_us1, sfu_us1, max_dilation1]
                single_sample = np.array(feature_list).reshape(1,-1)

                model_choice = st.selectbox("Select Model", ["Calibrated logistic regression"])
                if st.button("Predict"):    
                    if model_choice == "Calibrated logistic regression":
                        loaded_model = load_model("calibratedlogmodel.pkl")
                        prediction = loaded_model.predict(single_sample)
                        proba = loaded_model.predict_proba(single_sample)

                    if prediction == 1:
                        st.success("The patient is likely to have obstruction on MAG3 scan.")
                    else:
                        st.success("The patient is unlikely to have obstruction on MAG3 scan.")

                    st.write("Prediction: ", prediction)
                    st.write("Probability of obstruction on MAG3: ", proba)
                    st.caption("Our decision curve analysis suggests patients with 10-65 percent likelihood of obstruction are likely to benefit from this model.")
                    st.image('decision_curve_obstruction.png')

        if activity == 'SK-PNMS: Prediction for surgical intervention':

            with col_4:
            
                st.subheader ('Clinical Characteristics')
                gender = st.radio ("Sex", tuple(Gender_dict.keys()))
                prenatal_hn = st.radio ("Prenatal hydronephrosis", tuple(feature_dict.keys()))
                laterality = st.radio ("Side of hydronephrosis", tuple(Laterality_dict.keys()))
                st.caption('If bilateral, ultrasound findings should be focused on the side with worse hydroureter.')

        
            with col_5:

                st.subheader ('Ultrasound findings')
                length_1_1 = st.number_input ("Length (mm)", 0, 110)
                apd_us1 = st.number_input ("AP diameter (mm)", 0, 45)
                sfu_us1 = st.number_input ("SFU Grading", 0, 4)
                max_dilation1 = st.number_input ("maximum ureter diameter (mm)", 0, 50)
            
            with col_6:

                feature_list = [get_value(gender, Gender_dict), get_value(prenatal_hn, feature_dict), get_value(laterality, Laterality_dict), length_1_1, 
                                apd_us1, sfu_us1, max_dilation1]
                single_sample = np.array(feature_list).reshape(1,-1)

                model_choice = st.selectbox("Select Model", ["Calibrated logistic regression"])
                if st.button("Predict"):    
                    if model_choice == "Calibrated logistic regression":
                        loaded_model = load_model("calibratedlogmodel_surgery.pkl")
                        prediction = loaded_model.predict(single_sample)
                        proba = loaded_model.predict_proba(single_sample)

                    if prediction == 1:
                        st.success("The patient is likely to require surgical intervention.")
                    else:
                        st.success("The patient is unlikely to require surgical intervention.")

                    st.write("Prediction: ", prediction)
                    st.write("Probability of requiring surgical intervention: ", proba)
                    st.caption("Our decision curve analysis suggests patients with 10-75 percent likelihood of surgery are likely to benefit from this model.")
                    st.image('decision_curve_surgery.png')

        if activity == 'About SK-POMP':
            
                   with about_container1:
                       st.subheader ('SK-POMP is a prediction of primary obstructive megaureter for patients with hydroureter, developed at The Hospital for Sick Children (SickKids), Toronto, Ontario, Canada.')
                       st.write ('The tool is based on 183 infants identified to have primary non-refluxing megaureter.')
                       st.caption('Among infants identified to have hydronephrosis, those with primary non-refluxing megaureter accounts for the minority. Without a mercaptoacetyltriglycine-3 (MAG-3) diuretic renal scan, it is difficult to discern whether the cause of the megaureter is due to obstruction. Hence, we aim to develop a prediction model, specifically for the megaureter population, to predict the likelihood of detecting obstruction on MAG-3 scan based on clinical and ultrasound characteristics.')

                   with about_container2:
                       with col_7:
                           st.write('Using plot densities of the variables, we identified variables that had potential to differentiate patients who are likely to have obstruction')
                       with col_8:
                           st.image('SFU.png')
                           st.caption('Example plot of SFU grading demonstrating potential to differentiate patients who are likely to have obstruction.')

                   with about_container3:
                       with col_9:
                           st.write('Following this, we developed a logistic regression model with L2 regularization and was able to calibrate it to provide better prediction.')
                       with col_10:
                           st.image('calibration.png')
                           st.caption('Calibration curve of the pre- and post-calibration logistic regression model.')

                   with about_container4:
                       with col_11:
                           st.write('The final model had a area under receiving operating characteristics curve (AUROC) of 0.817 and area under precision-recall curve (AUPRC) of 0.736. The f1 score was 0.700.')
                           st.write('The tool was last updated on May 10, 2023 and may be updated with new data as they become available.')
                           st.caption('For questions regarding model hyperparameters and training, please contact Jin Kyu (Justin) Kim at: jjk.kim@mail.utoronto.ca')
                       with col_12:
                           st.image('ROC.png')
                           st.image('PRC.png')
                           st.image('CM.png')
                           st.caption('Final model evaluation using ROC, PRC, and confusion matrix.')

                   with about_container5:
                       with col_13:
                           st.subheader ('Reference')
                       with col_14:
                           st.write('Predicting the likelihood of obstruction for non-refluxing primary megaureter using a calibrated ridge regression model: SK-POMP (SickKids-Primary Obstructive Megaureter Prediction)')
                           st.caption('Kim JK, Chua ME, Khondker A, ... Richter J, Lorenzo AJ, Rickard M')
                           st.caption('Pending peer-reviewed publication')

        if activity == 'About SK-PNMS':
            
                   with about_container6:
                       st.subheader ('SK-PNMS is a prediction of surgical intervention in for patients with hydroureter, developed at The Hospital for Sick Children (SickKids), Toronto, Ontario, Canada.')
                       st.write ('The tool is based on 183 infants identified to have primary non-refluxing megaureter.')
                       st.caption('Among infants identified to have hydronephrosis, those with primary non-refluxing megaureter accounts for the minority. Patients with obstruction or worsening clinical or imagine features, as well as decreasing renal function, are candidates for surgical intervention. This model aims to identify patients at high risk of requiring surgical intervention at presentation, using clinical and ultrasound features.')

                   with about_container7:
                       with col_15:
                           st.write('Using plot densities of the variables, we identified variables that had potential to differentiate patients who are likely to require surgical intervention')
                       with col_16:
                           st.image('SFU_surgery.png')
                           st.caption('Example plot of SFU grading demonstrating potential to differentiate patients who are likely to require surgical intervention.')

                   with about_container8:
                       with col_17:
                           st.write('Following this, we developed a logistic regression model with L2 regularization and was able to calibrate it to provide better prediction.')
                       with col_18:
                           st.image('calibration_surgery.png')
                           st.caption('Calibration curve of the pre- and post-calibration logistic regression model.')

                   with about_container9:
                       with col_19:
                           st.write('The final model had a area under receiving operating characteristics curve (AUROC) of 0.815 and area under precision-recall curve (AUPRC) of 0.832. The f1 score was 0.750.')
                           st.write('The tool was last updated on May 10, 2023 and may be updated with new data as they become available.')
                           st.caption('For questions regarding model hyperparameters and training, please contact Jin Kyu (Justin) Kim at: jjk.kim@mail.utoronto.ca')
                       with col_20:
                           st.image('ROC_surgery.png')
                           st.image('PRC_surgery.png')
                           st.image('CM_surgery.png')
                           st.caption('Final model evaluation using ROC, PRC, and confusion matrix.')

                   with about_container10:
                       with col_21:
                           st.subheader ('Reference')
                       with col_22:
                           st.write('Predicting the likelihood of surgery for non-refluxing primary megaureter using a calibrated ridge regression model: SK-POMP (SickKids-Primary Obstructive Megaureter Prediction for Surgery)')
                           st.caption('Kim JK, Chua ME, Khondker A, ... Richter J, Lorenzo AJ, Rickard M')
                           st.caption('Pending peer-reviewed publication')

with header_container:
    st.title('SK-POMP & SK-PNMS')
    st.caption("SK-POMP (SickKids Primary Obstructive Megaureter Prediction) is a web app to predict the likelihood of identfying obstruction on MAG3 scan (defined as t1/2 > 20 minutes) based on an infant's baseline clinical and ultrasound characteristics.")
    st.caption("SK-PNMS (SickKids Prediction for Non-refluxing Megaureter Surgical interention) This is a web app to predict the likelihood of requiring surgery based on an infant's baseline clinical and ultrasound characteristics. The need for surgery was determined based on: presence of obstruction, decreased differential function <40%, decreasing differential renal function >5%, pain or recurrent UTIs, or worsening hydroureteronephrosis on follow-up imaging.")
    st.caption('These models are currently in development. Further external validation is required before wide use in clinical decision making. Please use at your own risk.')

if __name__ == '__main__':
    main()
