import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import warnings
import base64

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('loan_approval_dataset.csv')
data=df

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Check if the 'residential_assets_value' column exists
if 'residential_assets_value' in df.columns:
    df = df[df['residential_assets_value'] >= 10000]
else:
    print("Column 'residential_assets_value' does not exist in the dataframe.")
    exit()

# Mapping values and stripping whitespace from values before mapping
df['education'] = df['education'].str.strip().map({'Graduate': 1, 'Not Graduate': 0})
df['self_employed'] = df['self_employed'].str.strip().map({'Yes': 1, 'No': 0})
df['loan_status'] = df['loan_status'].str.strip().map({'Approved': 1, 'Rejected': 0})

# Dropping unnecessary columns
df.drop(['bank_asset_value', 'loan_amount', 'luxury_assets_value'], axis=1, inplace=True)

# Check for and handle NaN values in 'loan_status'
df.dropna(subset=['loan_status'], inplace=True)

# Fill or drop remaining NaN values in features if any
df.fillna(0, inplace=True)  # Here we're filling NaN with 0; you might want to use a more suitable strategy

# Ensure there are still rows remaining after filtering and handling NaN values
if df.shape[0] == 0:
    print("No data available after filtering and handling NaN values.")
    exit()

# Scaling numerical features
mms = MinMaxScaler()
scaled_columns = ['income_annum', 'loan_term', 'cibil_score', 'residential_assets_value', 'commercial_assets_value']
df[scaled_columns] = mms.fit_transform(df[scaled_columns])

# Features and target variable
X = df.drop(columns=['loan_id', 'loan_status'])
y = df['loan_status']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56)

# Train the model
rf_clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=7)
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred_forest = rf_clf.predict(X_test)

# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred_forest))
print(classification_report(y_test, y_pred_forest))
print(X.columns)


st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon='<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="white" class="bi bi-tools" viewBox="0 0 16 16"><path d="M1 0 0 1l2.2 3.081a1 1 0 0 0 .815.419h.07a1 1 0 0 1 .708.293l2.675 2.675-2.617 2.654A3.003 3.003 0 0 0 0 13a3 3 0 1 0 5.878-.851l2.654-2.617.968.968-.305.914a1 1 0 0 0 .242 1.023l3.27 3.27a.997.997 0 0 0 1.414 0l1.586-1.586a.997.997 0 0 0 0-1.414l-3.27-3.27a1 1 0 0 0-1.023-.242L10.5 9.5l-.96-.96 2.68-2.643A3.005 3.005 0 0 0 16 3q0-.405-.102-.777l-2.14 2.141L12 4l-.364-1.757L13.777.102a3 3 0 0 0-3.675 3.68L7.462 6.46 4.793 3.793a1 1 0 0 1-.293-.707v-.071a1 1 0 0 0-.419-.814zm9.646 10.646a.5.5 0 0 1 .708 0l2.914 2.915a.5.5 0 0 1-.707.707l-2.915-2.914a.5.5 0 0 1 0-.708M3 11l.471.242.529.026.287.445.445.287.026.529L5 13l-.242.471-.026.529-.445.287-.287.445-.529.026L3 15l-.471-.242L2 14.732l-.287-.445L1.268 14l-.026-.529L1 13l.242-.471.026-.529.445-.287.287-.445.529-.026z"/></svg>',
    layout="centered",
    initial_sidebar_state="auto",
)


st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    .title {
        color: green;
        text-align: center;
        margin-bottom: 2rem;
    }
    .title1 {
        color: red;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to set background image
def set_background(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        opacity: 0.8;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('loan.jpeg')

st.title('Loan Approval Prediction')


col1, col2, col3, col4 = st.columns(4)
with col1:
    nof=st.number_input('no_of_dependents')
with col2:
    edu=st.selectbox('education',['Graduate','Not Graduate'])
with col3:
    se=st.selectbox('self_employed',['Yes','No'])
with col4:
    ia=st.number_input('income_annum')
col5, col6, col7,col8 = st.columns(4)
with col5:
    lt=st.number_input('loan_term')
with col6:
    cs=st.number_input('cibil_score')
with col7:
    rav=st.number_input('residential_assets_value')
with col8:
    cav=st.number_input('commercial_assets_value')

col9, col10, col11 = st.columns(3)
with col10:
    btn = st.button('Predict the Approval')

standard=pd.DataFrame({
    'no_of_dependents':[nof],
    'income_annum':[ia],
    'loan_term':[lt],
    'cibil_score':[cs],
    'residential_assets_value':[rav],
    'commercial_assets_value':[cav]
})

X=data[['no_of_dependents', 'education', 'self_employed', 'income_annum',
       'loan_term', 'cibil_score', 'residential_assets_value',
       'commercial_assets_value']]
X = pd.concat([X, standard], ignore_index=True)

X[['no_of_dependents', 'income_annum',
       'loan_term', 'cibil_score', 'residential_assets_value',
       'commercial_assets_value']]=mms.fit_transform(X[['no_of_dependents','income_annum',
       'loan_term', 'cibil_score', 'residential_assets_value',
       'commercial_assets_value']])
value = X.iloc[-1, :].values.tolist()

ed={'Graduate': 1, 'Not Graduate': 0}
s={'Yes': 1, 'No': 0}

ed_map=ed.get(edu,'Unknown')
s_map=s.get(se,'Unknown')

if btn:
    input_data=pd.DataFrame({
    'no_of_dependents':[value[0]],
    'education':[ed_map],
    'self_employed':[s_map],
    'income_annum':[value[1]],
    'loan_term':[value[2]],
    'cibil_score':[value[3]],
    'residential_assets_value':[value[4]],
    'commercial_assets_value':[value[5]]
     
    })
try:
    prediction=rf_clf.predict(input_data)
    if prediction==1:
        st.markdown(f"<h2 class='title'>Yes!! The Loan is Approved</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 class='title1'>No!! The Loan is Not Approved</h2>", unsafe_allow_html=True)
except Exception as e:
    print("Error during prediction:", str(e))
