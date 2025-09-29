# From Crowdsourced Data to Policy Design:<br> Monitoring and Forecasting Homeless Tents


This repository includes all the code used in our study on monitoring and forecasting homeless tents through crowdsourced data. The figure below illustrates the overall process of our work. By following the workflow and using the provided Python code or Jupyter Notebook, our results should be reproducible. The data we used can be obtained only upon request. To request the dataset, please contact me at wjung@psu.edu.

![/Users/wooyongjung/WJ_Projects/HOMELESS_STUDY_EPJDS_2025_v3/figure/fig_overall_framework.png](https://github.com/yilmajung/HOMELESS_STUDY_EPJDS_2025_v3/blob/master/figure/fig_overall_framework.png)

### Installation
To avoid any dependency issues, please install requirements.txt first.
```
git clone git@github.com:yilmajung/HOMELESS_STUDY_EPJDS_2025_v3.git
pip install -r requirements.txt
pip install -e .
```

### Run
1. **Data Preparation**
    - Gather homeless tent information from street-view images:  `1_extract_streetview_images.ipynb` → `2_detect_tents_GroundingDino.py` → YOLO classifier (use weights in the YOLO folder)

    - Gather amenity and structure information: `4_extract_amenity_OverPass.py`

    - Combine all crowdsourced data: `3_combine_311_data.ipynb` → `5_combine_amenity_w_main.py`

2. **Spatiotemporal Analysis**
    - Train and Predict ST-VGP Model: `6_train_STVGP.py` → `7_predict_STVGP.py`

    - Aggregate the bounding-box level results to city-level: `8_aggregate_MC.ipynb`

    - Feature analysis: `9_check_importance.py`


3. **Model Evaluation**
    - Compare model performance: `10_0_compare_w_baseline_thres.py` → `10_1_compare_w_baseline_visualize.ipynb`
    - Ablation study: run the same process above using codes in ablation_study folder

