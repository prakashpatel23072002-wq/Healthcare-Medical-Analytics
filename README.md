# CareScope Analytics

CareScope Analytics is a deployable starter project for the Healthcare & Medical Analytics problem domain. It demonstrates:

- Exploratory Data Analysis on a clinical cohort
- Disease-risk prediction with logistic regression
- Patient-level decision support
- Simple hospital admission and staffing forecasting

## What this project includes

- A responsive analytics dashboard
- A deterministic starter dataset generated in the browser
- In-browser training and evaluation of a logistic regression model
- A patient risk prediction simulator
- Static deployment configuration for Vercel

## Why this starter is useful

This version works immediately without a database, Python runtime, or external APIs. That makes it ideal for:

- Final-year student projects
- Hackathon demos
- MVP validation
- Portfolio deployment

Once you are ready, you can replace the starter dataset with a real public dataset such as:

- UCI Heart Disease
- Pima Indians Diabetes
- MIMIC-derived public subsets
- Hospital readmission CSV datasets

## Project structure

```text
.
|-- package.json
|-- server.js
|-- vercel.json
|-- README.md
`-- src
    |-- app.js
    |-- index.html
    `-- styles.css
```

## Run locally

1. Install Node.js 18 or newer
2. Run:

```bash
npm start
```

3. Open `http://localhost:3000`

## Deploy live on Vercel

1. Create a GitHub repository and push this project
2. Sign in to [Vercel](https://vercel.com/)
3. Import the repository
4. Deploy using the default settings

Because this app is static, deployment is simple and usually finishes in under a minute.

## Suggested next upgrades

### Data

- Replace generated data with a CSV upload flow
- Connect a real dataset and persist it in PostgreSQL or Supabase
- Add patient segmentation and longitudinal tracking

### Machine learning

- Move training to Python with scikit-learn, XGBoost, or PyTorch
- Add SHAP explanations
- Train separate models for diabetes, cardiovascular risk, and readmission

### Productization

- Add login and clinician roles
- Store prediction history
- Create a FastAPI backend
- Add audit logs and model versioning

## Important note

This project is for academic, demonstration, and analytics prototyping use only. It is not a medical device and must not be used as a substitute for clinical judgment.
