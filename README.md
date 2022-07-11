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
</details>

### 2. Final report (churn_report.ipynb)
### 3. Acquire module (acquire.py)
### 4. Prepare module (prepare.py)
### 5. Predictions.csv
### 6. Exploration & modeling notebooks (model_testing.ipynb, explore.ipynb)
### 7. Functions to support modeling work (model.py)

## Project Goals
The project goals focus on identifying customers and drivers of churn.  Statistical testing and machine learning models are used to find factors of churn and make recommendations to Telco for reducing churn.

## Initial Questions and Hypotheses

Churn is customer turnover.  Understanding customer and business factors of churn is key to keeping and retaining customers. The purpose of this project is to identify which factors influence churn and identify ways to prevent it.  

Data Questions:
  - What customer factors are available in the data?
  - What business factors are available in the data?
  - What factors contribute the most to churn from the customers?
  - What factors from the business have the highest contribution to churn?

Project Questions:
1. Is churn customer or feature driven?
2. Are any groups of churn higher than the average rate?
    a. Of those groups, are there any subgroups who churn?
    b. What services lead to higher churn?
3. Is there a significant difference between the internet and phone services for those who churn? 
4. Is there a significant difference between payment methods that drive churn?


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
    - [x] Review project expectations from Codeup & Rubric
    - [x] Draft project goal to include measures of success
    - [x] Create questions related to the project
    - [x] Create questions related to the data
    - [x] Create a plan for completing the project using the data science pipeline
    - [x] Create a data dictionary to define variables and data context
    - [x] Draft starting hypothesis

- Acquire:
   - [x] Create .gitignore
   - [x] Create env file with log-in credentials
   - [x] Store env file in .gitignore to ensure security of sensitive data
   - [x] Create acquire.py module
   - [x] Store functions needed to acquire the Telco dataset from mySQL
   - [x] Ensure all imports needed to run the functions are inside the acquire.py document
   - [x] Using Jupyter Notebook
   - [x] Run all required imports
   - [x] Import functions from aquire.py module
   - [x] Summarize dataset using methods and document observations

- Prepare:
   - [x] Create prepare.py module
   - [x] Store functions needed to prepare the Telco data such as:
          - [x] Split Function: to split data into train, validate, and test
          - [x] Cleaning Function: to clean data for exploration
          - [x] Encoding Function: to create numeric columns for object column
          - [x] Feature Engineering Function: to create new features
   - [x] Ensure all imports needed to run the functions are inside the prepare.py document Using Jupyter Notebook
   - [x] Import functions from prepare.py module
   - [x] Summarize dataset using methods and document observations
   - [x] Clean data
   - [x] Features need to be turned into numbers
   - [x] Categorical features or discrete features need to be numbers that represent those categories
   - [x] Continuous features may need to be standardized to compare like datatypes
   - [x] Address missing values, data errors, unnecessary data, renaming
   - [x] Split data into train, validate, and test samples
   
- Explore:
  - [x] Answer key questions about hypotheses and find drivers of churn
      - Run at least two statistical tests
      - Document findings
  - [x] Create visualizations with intent to discover variable relationships
      - Identify variables related to churn
      - Identify any business features related to churn
  - [x] Summarize conclusions, provide clear answers, and summarize takeaways
  - [x] Explain plan of action as deduced from work to this point

- Model:
** Using Jupter Notebook:
  - [x] Establish Baseline Accuracy
  - [ ] Train and fit 3 Models 
  - [ ] Remove unnecessary features
  - [ ] Evaluate Best Performing Models
  - [ ] Choose Best performing model for test
  - [ ] Test Final Model on out-of-sample dataset
  - [ ] Summarize Performance
  - [ ] Interpret and document findings

- Delivery:
  - [ ] 5 min presentation in jupyter notebook
      - Introduction and Proj. Goals
      - Executive summary of findings, takeaways, and recommendations
      - Analysis Walkthrough
          -  Visualize relationships
          -  Document Takeaways
          -  Highlight where Project questions are answered
      -  Final takeaway and COAs 
  - [ ] Ready for questions 

## Steps to Reproduce

 - [x] You need an env.py file with hostname, username, and password for mySQL database that contains the telco_churn database
 - [x] Store env file in local repository
 - [x] Make .gitignore and validate env.py file is part of .gitignore
 - [ ] Clone my repo
 - [ ] Import python libraries: pandas, matplotlib, seaborn, numpy, scipy, and sklearn
 - [ ] follow steps outlined in README.md and churn_report.ipynb
 

## Key Findings and Recommendations
 - [ ] Key finding to come when analysis is complete

## Future Work 
 - 
