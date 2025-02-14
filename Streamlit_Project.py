import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import geopandas as gpd
from folium import Choropleth
import folium
import webbrowser
import os
from math import pi
import plotly.express as px
# Load and preprocess data
@st.cache_data
def load_and_preprocess_data(file_path):
    crime_data = pd.read_csv(file_path)
    excluded_features = [
        'assaultPerPop', 'robbbPerPop', 'murdPerPop', 'rapes', 'rapesPerPop', 'murders', 'autoTheftPerPop', 
        'arsonsPerPop', 'nonViolPerPop', 'assaults', 'burglPerPop', 'robberies', 'larcPerPop', 'burglaries', 
        'autoTheft', 'larcenies', 'arsons', 'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 
        'PctPolicMinor', 'PolicPerPop', 'PolicCars', 'PolicOperBudg', 'PctYoungKids2Par', 'PctTeen2Par', 
        'PctKidsBornNeverMar', 'PctPersDenseHous', 'PctHousOccup', 'PctKids2Par', 'PctFam2Par', 'PctLargHouseFam', 
        'LandArea', 'NumKidsBornNeverMar', 'MalePctDivorce', 'FemalePctDiv', 'pctWPubAsst', 'PctVacantBoarded', 
        'MalePctNevMarr', 'pctWInvInc', 'communityName', 'state', 'countyCode', 'communityCode', 'NumStreet'
    ]
    crime_data_cleaned = crime_data.drop(columns=excluded_features)
    crime_data_cleaned = crime_data_cleaned.dropna(subset=['ViolentCrimesPerPop'])

    imputer = SimpleImputer(strategy='mean')
    crime_data_imputed = pd.DataFrame(
        imputer.fit_transform(crime_data_cleaned),
        columns=crime_data_cleaned.columns
    )
    return crime_data_imputed

@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(
        random_state=42, n_estimators=200, max_depth=25, max_features=None, min_samples_leaf=2, min_samples_split=2
    )
    rf_model.fit(X_train, y_train)

    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Precompute results
    predictions = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        'model': rf_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_importances': feature_importances,
        'mae': mae,
        'mse': mse,
        'r2': r2
    }
@st.cache_data
def compute_correlation_analysis(X_train, y, top_features):
    correlation_top = X_train[top_features].corrwith(y)
    correlation_top_df = pd.DataFrame({
        'Feature': correlation_top.index,
        'Correlation': correlation_top.values
    }).sort_values(by='Correlation', ascending=False)
    return correlation_top_df

@st.cache_data
def compute_perturbation_analysis(_rf_model, X_test, top_features):
    perturbation_results = []
    for feature in top_features:
        original_predictions = _rf_model.predict(X_test)
        
        # Increase feature by 10%
        test_sample_increase = X_test.copy()
        test_sample_increase[feature] *= 1.1
        perturbed_predictions_increase = _rf_model.predict(test_sample_increase)
        mean_change_increase = np.mean(perturbed_predictions_increase - original_predictions)
        
        # Decrease feature by 10%
        test_sample_decrease = X_test.copy()
        test_sample_decrease[feature] *= 0.9
        perturbed_predictions_decrease = _rf_model.predict(test_sample_decrease)
        mean_change_decrease = np.mean(perturbed_predictions_decrease - original_predictions)
        
        perturbation_results.append({
            'Feature': feature,
            'Mean Change (Increase)': mean_change_increase,
            'Mean Change (Decrease)': mean_change_decrease
        })
    return pd.DataFrame(perturbation_results)

@st.cache_data
def load_raw_data(file_path):
    raw_data = pd.read_csv(file_path)
    return raw_data

@st.cache_data
def load_shapefile(shapefile_path):
    us_states = gpd.read_file(shapefile_path)
    us_states = us_states[us_states['admin'] == 'United States of America']
    return us_states

@st.cache_data
def merge_crime_with_shapefile(_us_states, state_crime_data):
    state_mapping = {
        'DC': 'District of Columbia', 'LA': 'Louisiana', 'SC': 'South Carolina',
        'MD': 'Maryland', 'FL': 'Florida', 'AL': 'Alabama', 'GA': 'Georgia',
        'NC': 'North Carolina', 'DE': 'Delaware', 'KS': 'Kansas', 'NM': 'New Mexico',
        'CA': 'California', 'TN': 'Tennessee', 'AR': 'Arkansas', 'KY': 'Kentucky',
        'MS': 'Mississippi', 'TX': 'Texas', 'NY': 'New York', 'AK': 'Alaska',
        'NV': 'Nevada', 'AZ': 'Arizona', 'WA': 'Washington', 'CO': 'Colorado',
        'VA': 'Virginia', 'MN': 'Minnesota', 'IN': 'Indiana', 'OK': 'Oklahoma',
        'MO': 'Missouri', 'MA': 'Massachusetts', 'WV': 'West Virginia',
        'NJ': 'New Jersey', 'IA': 'Iowa', 'OH': 'Ohio', 'OR': 'Oregon',
        'RI': 'Rhode Island', 'PA': 'Pennsylvania', 'WY': 'Wyoming', 'ID': 'Idaho',
        'CT': 'Connecticut', 'UT': 'Utah', 'SD': 'South Dakota', 'NH': 'New Hampshire',
        'WI': 'Wisconsin', 'ME': 'Maine', 'VT': 'Vermont', 'ND': 'North Dakota',
        'MI': 'Michigan', 'IL': 'Illinois'
    }
    state_crime_data['StateName'] = state_crime_data['state'].map(state_mapping)
    geo_data = _us_states.merge(state_crime_data, left_on='name', right_on='StateName', how='left')
    print(geo_data.head())
    return geo_data

@st.cache_data
def prepare_raw_data_with_imputation(file_path):
    """Prepare raw data with imputation but without dropping features"""
    raw_data = pd.read_csv(file_path)
    
    target = raw_data['ViolentCrimesPerPop']
    features = raw_data.drop('ViolentCrimesPerPop', axis=1)
    
    non_numeric_cols = ['communityName', 'state', 'countyCode', 'communityCode']
    numeric_features = features.drop(columns=non_numeric_cols)
    
    imputer = SimpleImputer(strategy='mean')
    imputed_data = pd.DataFrame(
        imputer.fit_transform(numeric_features),
        columns=numeric_features.columns
    )
    
    for col in non_numeric_cols:
        imputed_data[col] = features[col]
    imputed_data['ViolentCrimesPerPop'] = target
    
    return imputed_data

@st.cache_data
def get_state_data(imputed_data, state_code):
    """Get data for a specific state"""
    return imputed_data[imputed_data['state'] == state_code]

@st.cache_data
def compare_states(imputed_data, state1, state2, feature_importances):
    """Compare two states based on top features and violent crimes"""
    state1_data = get_state_data(imputed_data, state1)
    state2_data = get_state_data(imputed_data, state2)
    
    top_features = feature_importances['Feature'].head(10).tolist()
    
    comparison_data = []
    for feature in top_features:
        state1_mean = state1_data[feature].mean()
        state2_mean = state2_data[feature].mean()
        
        global_min = imputed_data[feature].min()
        global_max = imputed_data[feature].max()
        
        if global_max - global_min != 0:
            state1_normalized = (state1_mean - global_min) / (global_max - global_min)
            state2_normalized = (state2_mean - global_min) / (global_max - global_min)
        else:
            state1_normalized = 0.5  
            state2_normalized = 0.5
            
        comparison_data.append({
            'Feature': feature,
            f'{state1}_mean': state1_mean,
            f'{state2}_mean': state2_mean,
            f'{state1}_normalized': state1_normalized,
            f'{state2}_normalized': state2_normalized
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Get violent crimes per population and normalize using global values
    global_min_crime = imputed_data['ViolentCrimesPerPop'].min()
    global_max_crime = imputed_data['ViolentCrimesPerPop'].max()
    
    violent_crimes = {
        state1: state1_data['ViolentCrimesPerPop'].mean(),
        state2: state2_data['ViolentCrimesPerPop'].mean()
    }
    
    if global_max_crime - global_min_crime != 0:
        violent_crimes_normalized = {
            state: (value - global_min_crime) / (global_max_crime - global_min_crime)
            for state, value in violent_crimes.items()
        }
    else:
        violent_crimes_normalized = {state: 0.5 for state in [state1, state2]}
    
    return comparison_df, violent_crimes, violent_crimes_normalized

@st.cache_data
def compute_hotspot_analysis(data):
    # if 'state' not in data.columns or 'ViolentCrimesPerPop' not in data.columns:
    #     raise ValueError("Columns 'state' or 'ViolentCrimesPerPop' are missing from the dataset.")
    
    state_crime_data = data.groupby('state', as_index=False)['ViolentCrimesPerPop'].mean()

    state_crime_data.rename(columns={'ViolentCrimesPerPop': 'AvgViolentCrimesPerPop'}, inplace=True)

    state_crime_data = state_crime_data.sort_values(by='AvgViolentCrimesPerPop', ascending=False)

    national_avg = state_crime_data['AvgViolentCrimesPerPop'].mean()

    return state_crime_data, national_avg

def open_html_file():
    html_file_path = os.path.abspath("us_violent_crimes_map.html")
    webbrowser.open(f"file:///{html_file_path}")
# Load Data and Precompute Everything
file_path = 'crimedata.csv'
# shapefile_path = 'C:\Users\Abdullah\Desktop\BTU\Sem 4\Data Exploration\Project\ne_110m_admin_1_states_provinces.shp'
data = load_and_preprocess_data(file_path)
raw_data = load_raw_data(file_path)
X = data.drop(columns=['ViolentCrimesPerPop'])
y = data['ViolentCrimesPerPop']

results = train_model(X, y)
rf_model = results['model']
X_test = results['X_test']
feature_importances = results['feature_importances']
top_features = feature_importances.head(10)['Feature'].tolist()
perturbation_df = compute_perturbation_analysis(rf_model, X_test, top_features)
correlation_top_df = compute_correlation_analysis(results['X_train'], y, top_features)
state_crime_data, national_avg = compute_hotspot_analysis(raw_data)
# us_states = load_shapefile(shapefile_path)
# geo_data = merge_crime_with_shapefile(us_states, state_crime_data)
# Streamlit Interface
st.title("Crime Data Analysis")

menu = ["Data Overview", "Feature Importance", "Correlation Analysis", "Perturbation Analysis", "Hotspot Analysis"]
choice = st.sidebar.selectbox("Select a page", menu)

if choice == "Data Overview":
    st.subheader("Data Overview")
    st.dataframe(pd.read_csv("crimedata.csv"))

elif choice == "Feature Importance":
    st.subheader("Top 10 Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_importances['Importance'].head(10), 
            y=feature_importances['Feature'].head(10), 
            palette='viridis', ax=ax)
    for i, value in enumerate(feature_importances['Importance'].head(10)):
        ax.text(value + 0.005, i, f'{value:.2f}', va='center', ha='left')
    ax.set_title("Top 10 Important Features", fontsize=16)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    st.pyplot(fig)

    st.write(f"Model Evaluation: MAE={results['mae']}, MSE={results['mse']}, R²={results['r2']}")

elif choice == "Correlation Analysis":
    st.subheader("Correlation Analysis of Top Features")
    st.write("This analysis shows the correlation of the top 10 features with the target variable `ViolentCrimesPerPop`.")
    # Plot the correlation bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=correlation_top_df['Correlation'], y=correlation_top_df['Feature'], ax=ax)
    ax.set_title("Correlation of Top 10 Features with ViolentCrimesPerPop")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Feature")
    st.pyplot(fig)
    st.dataframe(correlation_top_df)

elif choice == "Perturbation Analysis":
    st.subheader("Perturbation Analysis")
    st.write("Effect of perturbing top features on predictions.")
    
    # fig, ax = plt.subplots(figsize=(10, 6))
    # perturbation_df.set_index('Feature')[['Mean Change (Increase)', 'Mean Change (Decrease)']].plot(kind='bar', ax=ax)
    # ax.set_ylabel("Mean Change in Predictions")
    # st.pyplot(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x='Mean Change (Increase)', y='Feature', data=perturbation_df, palette='Blues', ax=axes[0])
    axes[0].set_title("Increase in Features")
    sns.barplot(x='Mean Change (Decrease)', y='Feature', data=perturbation_df, palette='Reds', ax=axes[1])
    fig.tight_layout(pad=4.0)
    axes[1].set_title("Decrease in Features")
    st.pyplot(fig)

    st.dataframe(perturbation_df)


elif choice == "Hotspot Analysis":
    st.write("State-level analysis of average violent crimes per population:")
    
    # Enhanced Visualization with National Average Line
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(
        x='AvgViolentCrimesPerPop',
        y='state',
        data=state_crime_data,
        palette='Reds_r',
        ax=ax
    )
    ax.axvline(x=national_avg, color='blue', linestyle='--', label=f'National Average: {national_avg:.2f}')
    ax.set_title('Hotspot Analysis: Average Violent Crimes Per Population by State', fontsize=16)
    ax.set_xlabel('Average Violent Crimes Per Pop', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    

    st.dataframe(state_crime_data)
   

    # Create a markdown link
    # if st.link_button("Click here to open externally"):
    #     open_html_file()
    col1, col2 = st.columns([5, 4])
    with col1:
        st.write('Interactive map of state-level crime')
    with col2:
        if st.button("(open externally)", type="secondary"):
            file_path = os.path.abspath("us_violent_crimes_map.html")
            webbrowser.open(f"file:///{file_path}")
    path_to_html = "us_violent_crimes_map.html"
    with open(path_to_html, "r") as f:
        html_data = f.read()
    st.components.v1.html(html_data, height = 500 , width = 1000)
    st.markdown("---")
    st.subheader("State by State Comparison")
    imputed_data = prepare_raw_data_with_imputation(file_path)
    states = sorted(imputed_data["state"].unique())
    col1, col2 = st.columns(2)
    with col1:
        state1 = st.selectbox("Select first state", states, index=0)
    with col2:
        state2_options = [s for s in states if s != state1]
        state2 = st.selectbox("Select second state", state2_options, index=0)
    if st.button("Compare States"):
        comparison_df, violent_crimes, violent_crimes_normalized = compare_states(imputed_data, state1, state2, feature_importances)
    
    # Plot violent crimes comparison
        fig, ax = plt.subplots(figsize=(14, 8))
    
    # Original values
        states_list = [state1, state2]
        crimes_list = [violent_crimes[state1], violent_crimes[state2]]
        bars1 = sns.barplot(x=states_list, y=crimes_list, ax=ax, palette='viridis')
        ax.set_title("Violent Crimes Per Population")
        ax.set_ylabel("Violent Crimes Per Population")
    
    # # Add value labels on the bars
    #     for i, bar in enumerate(bars1.patches):
    #         ax1.text(
    #             bar.get_x() + bar.get_width()/2.,
    #             bar.get_height(),
    #             f'{crimes_list[i]:.3f}',
    #             ha='center', va='bottom'
    #     )
    
    # Normalized values
        # crimes_list_norm = [violent_crimes_normalized[state1], violent_crimes_normalized[state2]]
        # bars2 = sns.barplot(x=states_list, y=crimes_list_norm, ax=ax2, palette='viridis')
        # ax2.set_title("Violent Crimes Per Population (Normalized)")
        # ax2.set_ylabel("Normalized Value (0-1)")
    
    # Add value labels on the bars
        # for i, bar in enumerate(bars2.patches):
        #     ax2.text(
        #         bar.get_x() + bar.get_width()/2.,
        #         bar.get_height(),
        #         f'{crimes_list_norm[i]:.3f}',
        #         ha='center', va='bottom'
        # )
    
        plt.tight_layout()
        st.pyplot(fig)
    
    # Feature comparison plot
        fig2, ax4 = plt.subplots(figsize=(14, 8))
        x = np.arange(len(comparison_df['Feature']))
        width = 0.35
    
    
    # # Original values
    #     bars3_1 = ax3.bar(x - width/2, comparison_df[f'{state1}_mean'], width, label=state1)
    #     bars3_2 = ax3.bar(x + width/2, comparison_df[f'{state2}_mean'], width, label=state2)
    #     ax3.set_ylabel('Feature Values (Original)')
    #     ax3.set_title('Top Features Comparison (Original)')
    #     ax3.set_xticks(x)
    #     ax3.set_xticklabels(comparison_df['Feature'], rotation=45, ha='right')
    #     ax3.legend()
    
    # Normalized values
        bars4_1 = ax4.bar(x - width/2, comparison_df[f'{state1}_normalized'], width, label=state1)
        bars4_2 = ax4.bar(x + width/2, comparison_df[f'{state2}_normalized'], width, label=state2)
        ax4.set_ylabel('Feature Values (Normalized)')
        ax4.set_title('Top Features Comparison (Normalized)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(comparison_df['Feature'], rotation=45, ha='right')
        ax4.legend()
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Display numerical comparison
        st.write("Detailed Feature Comparison:")
        display_cols = ['Feature', f'{state1}_mean', f'{state2}_mean', 
                   f'{state1}_normalized', f'{state2}_normalized']
        st.dataframe(comparison_df[display_cols])   
    