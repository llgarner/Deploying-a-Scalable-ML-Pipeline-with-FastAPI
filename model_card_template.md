# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model was created by Lori Garner

Model uses `RandomForestClassifier` from `sklearn.model.RandomForestClassifier`.

The default parameters are used.

## Intended Use
This model is used to predict whether an individual makes more or less than 50K a year based on US census data.

## Training Data
More details about the training data: https://archive.ics.uci.edu/ml/datasets/census+income

Data is a cleaned dataset extracted from 1994 US Census data with the following features:
 - age: continuous
 - workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
 - fnlwgt: continuous
 - education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool
 - education-num: continuous
 - marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
 - occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces
 - relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
 - race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
 - sex: Female, Male
 - capital-gain: continuous
 - capital-loss: continuous
 - hours-per-week: continuous
 - native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands


## Evaluation Data
The data processed and split into training and test sets.
For both, categorical features are encoded using `OneHotEncoder` and the target is transformed using `LabelBinarizer`

## Metrics
Model Performance:
- Precision: 0.7437
- Recall: 0.6257
- F1: 0.6796

## Ethical Considerations
The model was trained using census data, which should not hold any particula bias toward any specific groups.

## Caveats and Recommendations
Checks on the data used for predictions should be done to ensure no inherent bias in the data collection process.