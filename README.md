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
  ├── clean/
  ├── raw/                              
├── outputs/                           # Generated outputs (RDD tables/plots/summaries)
├── service/                           # FastAPI + chat/prediction web UI + LLM integration
├── src/                               # Notebooks (data cleaning, hedonic, RDD)
├── archive/                           # Older versions / archival files
├── .env                               # Create locally when running LLM
├── .gitignore
└── README.md
```


## Full Pipeline 
1) Clone repository
```
git clone https://github.com/KeaganLai/DSA4264-Public-Policy-and-Society.git 
cd DSA4264-Public-Policy-and-Society
```

&nbsp;


2) (a) Create and activate a virtual environment (on Windows)
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

&nbsp;

2) (b) Create and activate a virtual environment (on Mac)
```
python -m venv .venv
source .venv/bin/activate
```
&nbsp;


3) Install dependencies in venv kernel
```
pip install -r requirements.txt
```

&nbsp;

4) Create `.env` file and add your LTA DataMall API key
```env
LTA_KEY=your_lta_api_key_here
```

&nbsp;


5) Create data/ folder locally, with the clean/ and raw/ subgroups
```
mkdir data
mkdir data\clean
mkdir data\raw
```

&emsp;&emsp;Then, verify you can view the files from Google Drive: 

&emsp;&emsp;<https://drive.google.com/drive/folders/1ldE2xA_QpdttSjHPtE6T04_KMsEhhTAx>

&nbsp;


6) (a) If you would like to directly work with our clean data, download and set the paths as displayed below:
```text
data/
├── clean/
    ├── final_df.csv        
    ├── final_df_robust_1.csv
    ├── final_df_robust_2.csv                                      
    ├── Good_School_index.csv
    .
    .
    .
├── raw/
```

&emsp;&emsp;Then, run all cells in this order:
```
src/hedonic.ipynb
src/rdd.ipynb
```

&nbsp;


6) (b) Else, if you would like to download the raw data and verify our data cleaning and calculation process, set the paths as displayed below:
```text
data/
├── clean/                           
├── raw/
    ├── 4Q2025 RPI Table
    ├── Resale Flat Prices (Based on Approval Date), 1990 - 1999
    ├── Resale Flat Prices (Based on Approval Date), 2000 - Feb 2012
    .
    .
    .
```

&emsp;&emsp;Make sure the repository-root `.env` file contains `LTA_KEY`, since `src/data cleaning/1_bus_data.ipynb` reads it before calling the LTA DataMall API.

&emsp;&emsp;Then, run all cells in this order:
```
src/data cleaning/1_bus_data.ipynb
src/data cleaning/2_hawker_data.ipynb
src/data cleaning/3_mall_data.ipynb
src/data cleaning/4_mrt_data.ipynb
src/data cleaning/5_rpi_data.ipynb
src/data cleaning/6_hdb_data.ipynb
src/data cleaning/7_SDI_calculation.ipynb
src/data cleaning/7_SDI_robust_1_calculation.ipynb
src/data cleaning/7_SDI_robust_2_calculation.ipynb
src/data cleaning/8_hdb_sch_features.ipynb
src/hedonic.ipynb
src/rdd.ipynb
```





# Setting up the LLM
1) Install Ollama from official installer:
   
    <https://ollama.com/download/windows>

&nbsp;

2) Create a `.env` file in the repo root. It should contain:
```
LTA_KEY=your_lta_api_key_here
CHAT_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_TIMEOUT_SECONDS=180
REPORT_DOCX_AUTO_DISCOVER=0
```

&nbsp;

3) Run PATH script to detect Ollama.exe
```
$ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not (($userPath -split ';') -contains $ollamaPath)) {
  [Environment]::SetEnvironmentVariable("Path", "$userPath;$ollamaPath", "User")
}
```

&nbsp;

4) Start Ollama in the background, or run
```
ollama serve
```

&nbsp;

5) Verify Ollama:
```
ollama --version
```

&nbsp;

6) Pull model once
```
ollama pull qwen2.5:7b-instruct
```

&nbsp;

7) (Optional) If artifacts/ is empty, train model using
```
python -m service.train_baseline --all-thresholds
```

&nbsp;

8) Verify that you have ran hedonic.ipynb and rdd.ipynb, and outputs/ folder contains:
```text
outputs/
├── rdd/            
├── rdd_improved/
```

&nbsp;

## Running the application
1) Start the application with
```
python -m uvicorn --env-file .env service.main:app --reload
```

&nbsp;

2) Open the dashboard in a browser of your choice at
```
http://127.0.0.1:8000/
```
