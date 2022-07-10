# Classification-project
This repository contains the code for the classification project completed as part of the Codeup Data Science curriculum.

## Repo contents:

### <summary>1. Readme File</summary>
<details>

```- project description with goals
- initial hypotheses and/or questions you have of the data, ideas
- data dictionary
- project planning (lay out your process through the data science pipeline)
- instructions or an explanation of how someone else can reproduce your project and findings (What would someone need to be able to recreate your project on their own?)
- key findings, recommendations, and takeaways from your project,
```
</p>
</details>

### 2. Final report (churn_report.ipynb)
### 3. Acquire module (acquire.py)
### 4. Prepare module (prepare.py)
### 5. predictions.csv
### 6. Exploration & modeling notebooks (model_testing.ipynb, explore.ipynb)
### 7. Functions to support modeling work (model.py)

## Project Goals

## Initial Questions and Hypotheses

## Data Dictionary
<details>

|Feature Name|	Description|	Data Type| Updated to|
|:---|:---|---:|:----|
|payment_type_id| Numerical version of payment_type|categorical| deleted|
|internet_service_type_id| numercical version of internet service type| categorical| deleted|
|customer_id|	Contains customer ID|	categorical|
|gender|	whether the customer female or male|	categorical| deleted after encoding|
|senior_citizen|	Whether the customer is a senior citizen or not (1, 0)|	numeric, int|is_senior_citizen|
|partner|	Whether the customer has a partner or not (Yes, No)|	categorical| deleted after encoding|
|dependents|	Whether the customer has dependents or not (Yes, No)|	categorical|deleted after encoding|
|tenure|	Number of months the customer has stayed with the company|	numeric, int|
|phone_service|	Whether the customer has a phone service or not (Yes, No)|	categorical|deleted after encoding|
|multiple_lines|	Whether the customer has multiple lines r not (Yes, No, No phone service)|	categorical|
|internet_service_type|	Customer’s internet service provider (DSL, Fiber optic, No)|categorical|
|online_security|	Whether the customer has online security or not (Yes, No, No internet service)|	categorical|
|online_backup|	Whether the customer has online backup or not (Yes, No, No internet service)| categorical|
|device_protection|	Whether the customer has device protection or not (Yes, No, No internet service)|	categorical|
|tech_support|	Whether the customer has tech support or not (Yes, No, No internet service)|	categorical|
|streaming_tv|	Whether the customer has streaming TV or not (Yes, No, No internet service)| categorical|
|streaming_movies|	Whether the customer has streaming movies or not (Yes, No, No internet service)|	categorical|
|contract_type|	The contract term of the customer (Month-to-month, One year, Two year)|	categorical| deleted|
|paperless_billing|	Whether the customer has paperless billing or not (Yes, No)|	categorical| deleted after encoding|
|payment_type| The customer’s payment method (Electronic check, Mailed check, Bank transfer, Credit card)|categorical|
|monthly_charges|	The amount charged to the customer monthly|	numeric , int|
|total_charges|	The total amount charged to the customer|	object| numerical, int|
|churn|	Whether the customer churned or not (Yes or No)|	categorical| Yes:1 No:0|
|* is_male| gender converted to Male:1 Female:0| categorical|
|* has_phone| phone_service updted to Yes:1, No:0| categorical
|* has_internet_service|internet_service_type updated to show Fiber and DSL as 1, others as 0| categorical|
|* has_partner| partner updated to Yes:1, No:0|categorical|
|* has_dependent|dependents updated to Yeas:1, No:0|categorical|
|* is_paperless| paperless_billing updated to Yes:1, No:0| categorical|
|* is_month_to_month|contract type month-to-month:1, others: 0|categorical|
|* is_autopay| E-check and mailed:0 automatic transfers:1|categorical|
|* has_streaming|streaming movies or TV:1 others :0| categorical

* indicated row added through python
</details>

## Project Plan

- Planning:

- Acquire:

- Prepare:

- Explore:

- Model:

- Delivery:

## Steps to Reproduce

## Key Findings and Recommendations


## Limitations


## Future Work 

