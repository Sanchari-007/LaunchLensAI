import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request

app = Flask(__name__)

# Get base directory of the app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to your .pkl files
MODEL_FILE = os.path.join(BASE_DIR, 'xgb_product_launch_predictor.pkl')
THRESHOLD_FILE = os.path.join(BASE_DIR, 'xgb_product_launch_threshold.pkl')
ALL_YEARS_FILE = os.path.join(BASE_DIR, 'all_years.pkl')
ALL_MONTHS_FILE = os.path.join(BASE_DIR, 'all_months.pkl')
TRAIN_COLS_FILE = os.path.join(BASE_DIR, 'train_columns.pkl')

# Global variables for ML assets
loaded_model = None
loaded_threshold = None
loaded_all_years = None
loaded_all_months = None
loaded_train_columns = None

# Load ML assets with XGBoost 2.1.4 compatibility
def load_ml_assets():
    global loaded_model, loaded_threshold, loaded_all_years, loaded_all_months, loaded_train_columns
    
    try:
        print("Loading ML assets...")
        required_files = [MODEL_FILE, THRESHOLD_FILE, ALL_YEARS_FILE, ALL_MONTHS_FILE, TRAIN_COLS_FILE]
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"Error: {file_path} not found!")
                return False
        
        with open(MODEL_FILE, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Compatibility with XGBoost 2.1.4
        if hasattr(loaded_model, 'use_label_encoder'):
            loaded_model.use_label_encoder = False
        if hasattr(loaded_model, 'eval_metric') and loaded_model.eval_metric is None:
            loaded_model.eval_metric = 'logloss'
        
        with open(THRESHOLD_FILE, 'rb') as f:
            loaded_threshold = pickle.load(f)
        with open(ALL_YEARS_FILE, 'rb') as f:
            loaded_all_years = pickle.load(f)
        with open(ALL_MONTHS_FILE, 'rb') as f:
            loaded_all_months = pickle.load(f)
        with open(TRAIN_COLS_FILE, 'rb') as f:
            loaded_train_columns = pickle.load(f)
        
        print("ML assets loaded successfully!")
        return True
    
    except Exception as e:
        print(f"Error loading ML assets: {e}")
        return False


# Define dropdown options
MAIN_CATEGORIES = [
    'Film & Video', 'Music', 'Games', 'Technology', 'Design', 'Art',
    'Food', 'Fashion', 'Publishing', 'Theater', 'Photography', 'Crafts',
    'Dance', 'Comics', 'Journalism'
]

CATEGORIES = [
    'Film & Video', 'Music', 'Games', 'Technology', 'Design', 'Art',
    'Food', 'Fashion', 'Publishing', 'Theater', 'Photography', 'Crafts',
    'Dance', 'Comics', 'Journalism', 'Product Design', 'Graphic Design',
    'Video Games', 'Hardware', 'Software', 'Apps', 'Web'
]

COUNTRIES = [
    'United States (US)', 'United Kingdom (GB)', 'Canada (CA)', 'Australia (AU)', 
    'Germany (DE)', 'France (FR)', 'Italy (IT)', 'Spain (ES)', 
    'Netherlands (NL)', 'Sweden (SE)', 'Denmark (DK)', 'Norway (NO)', 
    'Switzerland (CH)', 'Belgium (BE)', 'Austria (AT)', 'Ireland (IE)', 
    'New Zealand (NZ)', 'Japan (JP)', 'Singapore (SG)', 'Hong Kong (HK)'
]

MONTHS = list(range(1, 13))
MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

YEARS = list(range(2025, 2050))

def extract_country_code(country_display):
    """Extract country code from display format 'Country Name (CODE)'"""
    if '(' in country_display and ')' in country_display:
        return country_display.split('(')[1].split(')')[0]
    return country_display

def create_probability_chart(probability, prediction_text):
    """Create animated probability visualization with multiple frames"""
    try:
        # Create enhanced styling
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.patch.set_facecolor('#f8fafc')
        
        # Chart 1: Enhanced Bar Chart with gradient
        categories = ['Failed', 'Successful']
        probabilities = [1 - probability, probability]
        
        # Create gradient colors
        colors = ['#ef4444', '#3b82f6']
        
        # Create bars with enhanced styling
        bars = ax1.bar(categories, probabilities, color=colors, alpha=0.9,
                      edgecolor='white', linewidth=3, capsize=10)
        
        # Add gradient effect and shadows
        for i, (bar, prob, color) in enumerate(zip(bars, probabilities, colors)):
            height = bar.get_height()
            width = bar.get_width()
            x = bar.get_x()
            
            # Add shadow effect
            shadow = plt.Rectangle((x + 0.02, -0.02), width, height, 
                                 facecolor='black', alpha=0.1, zorder=0)
            ax1.add_patch(shadow)
            
            # Add highlight effect
            highlight = plt.Rectangle((x, height * 0.7), width, height * 0.3, 
                                    facecolor='white', alpha=0.2, zorder=3)
            ax1.add_patch(highlight)
        
        # Enhanced axis styling
        ax1.set_ylabel('Probability', fontsize=14, fontweight='bold', color='#1e293b')
        ax1.set_title('Prediction Probabilities', fontsize=16, fontweight='bold', 
                     pad=25, color='#1e293b')
        ax1.set_ylim(0, 1.2)
        ax1.grid(True, alpha=0.3, axis='y', color='#3b82f6')
        ax1.set_facecolor('#fafbff')
        
        # Enhanced value labels
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{prob:.1%}', ha='center', va='bottom', 
                    fontsize=14, fontweight='bold', color='#1e293b',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Chart 2: Enhanced Donut Chart
        sizes = [probability, 1 - probability]
        colors_pie = ['#3b82f6', '#ef4444']
        labels = ['Success', 'Failure']
        
        # Create donut with enhanced styling
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie,
                                           autopct='%1.1f%%', startangle=90,
                                           wedgeprops=dict(width=0.6, edgecolor='white', 
                                                         linewidth=3, alpha=0.9),
                                           textprops={'fontsize': 12, 'fontweight': 'bold'})
        
        # Enhanced center circle
        centre_circle = plt.Circle((0, 0), 0.70, fc='white', linewidth=3, 
                                 edgecolor='#e2e8f0', alpha=0.95)
        ax2.add_patch(centre_circle)
        
        # Enhanced center text
        ax2.text(0, 0.1, f'{probability:.1%}', ha='center', va='center',
                fontsize=20, fontweight='bold', color='#1e293b')
        ax2.text(0, -0.15, 'Confidence', ha='center', va='center',
                fontsize=12, fontweight='600', color='#64748b')
        
        ax2.set_title(f'Prediction: {prediction_text}', fontsize=16, fontweight='bold', 
                     pad=25, color='#1e293b')
        
        # Remove spines for cleaner look
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        plt.tight_layout(pad=4.0)
        
        # Save with higher quality
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300,
                   facecolor='#f8fafc', edgecolor='none', pad_inches=0.3)
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_str
    except Exception as e:
        print(f"Error creating animated chart: {e}")
        return None

def preprocess_input(input_data):
    """Preprocess input data to match training format"""
    try:
        # Convert to DataFrame
        df_new_raw = pd.DataFrame([input_data])
        
        # Convert categorical columns to 'category' dtype
        categorical_cols_for_dtype = ['main_category', 'category', 'country']
        for col in categorical_cols_for_dtype:
            if col in df_new_raw.columns:
                df_new_raw[col] = df_new_raw[col].astype('category')
        
        # Ensure launch_year and launch_month categories are consistent
        if loaded_all_years is not None:
            df_new_raw['launch_year'] = pd.Categorical(df_new_raw['launch_year'], 
                                                       categories=loaded_all_years)
        if loaded_all_months is not None:
            df_new_raw['launch_month'] = pd.Categorical(df_new_raw['launch_month'], 
                                                        categories=loaded_all_months)
        
        # One-hot encoding
        categorical_cols_for_ohe = ['main_category', 'category', 'country', 
                                    'launch_month', 'launch_year']
        X_new_encoded_temp = pd.get_dummies(df_new_raw, columns=categorical_cols_for_ohe, 
                                            dtype=bool)
        
        # Reindex to align with training columns
        X_new_encoded = X_new_encoded_temp.reindex(columns=loaded_train_columns, 
                                                   fill_value=False)
        
        return X_new_encoded
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    # Check if ML assets are loaded
    if loaded_model is None:
        error_message = "ML model not loaded. Please ensure all .pkl files are in the correct directory."
        return render_template('index.html',
                             main_categories=MAIN_CATEGORIES,
                             categories=CATEGORIES,
                             countries=COUNTRIES,
                             months=MONTHS,
                             month_names=MONTH_NAMES,
                             years=YEARS,
                             error=error_message)
    
    if request.method == 'POST':
        try:
            # Get form data
            main_category = request.form.get('main_category')
            category = request.form.get('category')
            country_display = request.form.get('country')
            country = extract_country_code(country_display)  # Extract code for model
            usd_goal = float(request.form.get('usd_goal'))
            campaign_duration = int(request.form.get('campaign_duration'))
            launch_month = int(request.form.get('launch_month'))
            launch_year = int(request.form.get('launch_year'))
            
            # Validate inputs
            if not all([main_category, category, country_display]):
                raise ValueError("Please fill in all dropdown fields")
            
            # Calculate log of USD goal
            usd_goal_log = np.log10(usd_goal)
            
            # Prepare input data (use country code for model)
            input_data = {
                'main_category': main_category,
                'category': category,
                'country': country,  # Use extracted code
                'usd_goal_log': usd_goal_log,
                'campaign_duration': campaign_duration,
                'launch_month': launch_month,
                'launch_year': launch_year
            }
            
            # Preprocess input
            X_processed = preprocess_input(input_data)
            
            if X_processed is None:
                raise ValueError("Error processing input data")
            
            # Make prediction with XGBoost 2.1.4 compatibility
            try:
                prediction_probability = loaded_model.predict_proba(X_processed)[:, 1][0]
            except Exception as pred_error:
                # Alternative prediction method for XGBoost 2.1.4
                prediction_probability = loaded_model.predict_proba(X_processed.values)[:, 1][0]
            
            final_prediction = (prediction_probability >= loaded_threshold).astype(int)
            prediction_text = "Successful" if final_prediction == 1 else "Failed"
            
            # Create visualization
            chart_img = create_probability_chart(prediction_probability, prediction_text)
            
            # Prepare input data for display (use display format for country)
            display_input_data = {
                'main_category': main_category,
                'category': category,
                'country': country_display,  # Use full display name
                'usd_goal_log': usd_goal_log,
                'campaign_duration': campaign_duration,
                'launch_month': launch_month,
                'launch_year': launch_year
            }
            
            return render_template('result.html',
                                 prediction=prediction_text,
                                 probability=prediction_probability,
                                 chart_img=chart_img,
                                 input_data=display_input_data,
                                 usd_goal=usd_goal,
                                 month_name=MONTH_NAMES[launch_month])
            
        except Exception as e:
            error_message = f"Error processing your request: {str(e)}"
            return render_template('index.html',
                                 main_categories=MAIN_CATEGORIES,
                                 categories=CATEGORIES,
                                 countries=COUNTRIES,
                                 months=MONTHS,
                                 month_names=MONTH_NAMES,
                                 years=YEARS,
                                 error=error_message)
    
    return render_template('index.html',
                         main_categories=MAIN_CATEGORIES,
                         categories=CATEGORIES,
                         countries=COUNTRIES,
                         months=MONTHS,
                         month_names=MONTH_NAMES,
                         years=YEARS)

if __name__ == '__main__':
    # Load ML assets at startup
    success = load_ml_assets()
    if not success:
        print("Failed to load ML assets. Please check that all .pkl files are present.")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # debug=False for production


