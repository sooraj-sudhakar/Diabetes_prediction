# Diabetes prediction
Diabetes prediction using neural network created using keras with tensorflow backend.

### Dependancy
```sh
pip install h5py==2.8.0  
pip install Keras==2.2.0  
pip install Keras-Applications==1.0.2  
pip install Keras-Preprocessing==1.0.1  
pip install numpy==1.14.5  
pip install PyYAML==3.12  
pip install scikit-learn==0.19.1  
pip install scipy==1.1.0  
pip install six==1.11.0  
pip install sklearn==0.0 `
```
### Dataset
This work is used to predict the diabetes in a patient. The dataset used here is the *Pima Indians diabetes* database. The dataset consists of 768 entries having 9 features. The entires correspond to the test on each patient. 

The 9 features are :
- Pregnancies - Number of times pregnant
- GlucosePlasma - glucose concentration a 2 hours in an oral glucose tolerance test
- BloodPressure - Diastolic blood pressure (mm Hg)
- SkinThickness - Triceps skin fold thickness (mm)
- Insulin - 2-Hour serum insulin (mu U/ml)
- BMI - Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction - Diabetes pedigree function
- Age - Age (years)
- Outcome - Class variable (0 or 1) 268 of 768 are 1, the others are 0

#### Sample

| Pregnancies | GlucosePlasma |BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction |Age|Outcome
| ------ | ------ | ------ | ------ | ------ | ------ |------ |------ |------ |
|6|	148|	72|	35|	0|	33.6|	0.627|	50|	1|
|1|	85|	66|	29|	0|	26.6|	0.351|	31|	0|

### Working
There is a main *training.py* file which contains the ANN defined using the keras. The input *Pima Indians diabetes* csv file is splitted into train & test. The trained model is saved as *model.h5*. This saved model will be then used for the single as well as the bulk prediction programs. 

![Train vs validation loss](https://github.com/sooraj-sudhakar/Diabetes_prediction/blob/master/graph.png) ![Bulk prediction](https://github.com/sooraj-sudhakar/Diabetes_prediction/blob/master/Bulk_prediction_out.png)
