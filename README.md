## Link to deployed streamlit app for predictions: https://cse482finalproject.streamlit.app/

# Clutch Under Pressure: NFL Team Performance in High-Leverage Situations
This project analyzes how NFL teams perform during the highest-pressure moments of a game, known as clutch situations. Clutch plays are defined as offensive snaps occurring in the 4th quarter, within the final five minutes, and when the score margin is eight points or fewer. Using multiple seasons of NFL play-by-play data from the nflverse project, this analysis evaluates team performance using advanced metrics such as Expected Points Added (EPA) and Win Probability Added (WPA). The goal is to understand which teams rise under pressure, how clutch performance compares to overall offensive efficiency, and how machine learning can be used to predict clutch strength. The entire workflow is implemented in Python using Visual Studio Code with Jupyter Notebook.
## Features
- Multi-season NFL dataset (2019, 2020, 2021, 2023)
- Automatic clutch play identification
- Team-level clutch metrics: EPA, WPA, success rate, clutch improvement score
- Clean exploratory analysis with inline visualizations
- Machine learning prediction of clutch EPA using Random Forests
- JSON export of team-season clutch results
- Fully reproducible and organized analysis pipeline
## Clutch Definition
A play is considered clutch if it occurs in the 4th quarter, there are 300 seconds or fewer remaining, and the score differential is 8 points or fewer. These conditions reflect true high-leverage situations where game outcomes can shift dramatically and align with standards commonly used in football analytics.
## Technologies Used
- Python
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn
- Visual Studio Code with Jupyter Notebook
## Project Structure
project-root/  
│  
├── data/  
│   ├── play_by_play_2019.csv  
│   ├── play_by_play_2020.csv  
│   ├── play_by_play_2021.csv  
│   └── play_by_play_2023.csv  
│  
├── final_clutch_analysis.ipynb  
├── team_clutch_stats.json  
├── README.md  
└── requirements.txt
## How to Run
To run this project, first install dependencies using:  
pip install pandas numpy matplotlib seaborn scikit-learn  
Clone the repository using:  
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git  
Open the notebook in Visual Studio Code:  
final_clutch_analysis.ipynb  
Run all cells to load data, compute clutch metrics, generate visualizations, and view results. If using a script version, run:  
main()
## Data Source
NFL play-by-play data from nflverse: https://github.com/nflverse/nflverse-data/releases
## Authors
Randal Burks  
Nyla Blagrove  
Madison Honore  
Master of Science in Data Science, Michigan State University
## License
MIT License
