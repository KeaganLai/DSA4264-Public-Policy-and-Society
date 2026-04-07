<p align="center">
  <strong style="font-size:64px;">DSA4264-Public-Policy-and-Society</strong> <br>
  <em>HDB Resale Prices & Proximity to Good Primary Schools</em> <br>
</p>

## Project Overview
Housing demand in Singapore is closely tied to access to quality education, particularly at the primary school level. Under the Ministry of Education's admission framework, priority is given to children living within 1 km of a primary school, creating a strong incentive for families to purchase homes near desirable schools.

As a result, proximity to "good" primary schools is widely believed to influence HDB resale flat prices. In this project, we take the perspective of data scientists in the Ministry of National Development, tasked with evaluating how school proximity affects resale prices.

We use:
- Hedonic pricing models to estimate proximity premiums.
- A Regression Discontinuity Design (RDD) around the 1 km school cutoff to estimate local causal effects.

## Project Structure
```text
DSA4264-Public-Policy-and-Society/
├── artifacts/                         # Trained model artifacts + metadata for API
├── data/                              # Input data files (download from Google Drive)
├── outputs/                           # Generated outputs (RDD tables/plots/summaries)
├── service/                           # FastAPI + chat/prediction web UI + LLM integration
├── src/                               # Notebooks (data cleaning, hedonic, RDD)
├── archive/                           # Older versions / archival files
├── .gitignore
└── README.md
```

## Full Pipeline 
1) Clone repository
```
git clone https://github.com/KeaganLai/DSA4264-Public-Policy-and-Society.git 
cd DSA4264-Public-Policy-and-Society
```
2) Create and activate a virtual environment
```
python -m venv .venv
source venv/bin/activate
```
3) Install dependencies
```
pip install -r requirements.txt
```
4) Download data from Google Drive and place into data/
   
<https://drive.google.com/drive/folders/1ldE2xA_QpdttSjHPtE6T04_KMsEhhTAx>

Then, create data/ folder locally
```
mkdir data
```
Next, put all provided files into data/
```text
data/
├── final_df.csv                         
├── hdb_nearest_sch.csv                            
├── Good_School_index.csv
```
           
5) Run notebooks first (required)
Run all cells in this order:
```
src/hedonic.ipynb
src/rdd.ipynb
```
# Acknowledgements
- Singapore data.gov.sg
- OneMap API
