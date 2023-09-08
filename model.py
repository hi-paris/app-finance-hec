import pandas as pd
from pandas import to_datetime
from pandas.plotting import register_matplotlib_converters
import numpy as np
from pathlib import Path
import base64
import io
import os
from datetime import date, datetime
import yfinance as yf
from PIL import Image # display an image
from io import StringIO # upload file


from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import altair as alt
from PIL import Image
from vega_datasets import data
import pandas_datareader as pdr
import streamlit as st



from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px
import seaborn as sns
import matplotlib.pyplot as plt
register_matplotlib_converters()


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import pmdarima as pm

from fpdf import FPDF



sns.set(style="whitegrid")
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
st.set_option('deprecation.showPyplotGlobalUse', False)






# Configuration de l'app (html, java script like venv\)

# Deploy the app localy in terminal: streamlit run model.py

st.set_page_config(
    page_title="Finance", layout="wide", page_icon="./images/flask.png"
)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

@st.cache_data # compression data
def get_data():
    source = data.stocks()
    source = source[source.date.gt("2004-01-01")]
    return source


@st.cache_data
def get_chart(data):
    hover = alt.selection_single(
        fields=["Date_2"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title="Evolution of stock prices")
        .mark_line()
        .encode(
            x="Date_2",
            y=kpi,
            #color="symbol",
            # strokeDash="symbol",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="yearmonthdate(date)",
            y=kpi,
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("date", title="Date"),
                alt.Tooltip(kpi, title="Price (USD)"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()


@st.cache_data
def convert_df(df):
# IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')



# from st_files_connection import FilesConnection

# # Create connection object and retrieve file contents.
# # Specify input format is a csv and to cache the result for 600 seconds.
# conn = st.experimental_connection('gcs', type=FilesConnection)
# df = conn.read("streamlit-hiparis/master.csv", input_format="csv", ttl=600)
# st.dataframe(df)







############################# LOAD STOCK DATA ####################################

# Load annual stock data
data = pd.read_csv(r"data/CourseData.csv",sep=",", decimal=".").iloc[:,1:]
data.columns = ["Year","Stock","Dividends","Price"]

# Dictionary for stock abbreviation
dictionary_assets = {
    "AAR CORP":"AIR",
    "AMERICAN AIRLINES GROUP INC": "AAL",
    "CECO ENVIRONMENTAL CORP" :"CECO", 
    "ASA GOLD AND PRECIOUS METALS":"ASA",
    "PINNACLE WEST CAPITAL CORP":"PNW",
    "PROG HOLDINGS INC":"PRG",
    "ABBOTT LABORATORIES":"ABT",
    "ACMAT CORP  -CL A":"ACMTA",
    "ACME UNITED CORP":"ACU",
    "BK TECHNOLOGIES CORP":"BKTI"
}

data["Stock"] = data["Stock"].map(dictionary_assets)
list_risky_assets = data["Stock"].unique()

data["Year"] = data["Year"].astype(str)
data["Dividends"] = data["Dividends"].fillna(0)


# Image HEC
image_hec = Image.open('images/hec.png')
st.image(image_hec, width=300)

# Image Hi Paris
image_hiparis = Image.open('images/hi-paris.png')



##################################################################################
############################# DASHBOARD PART #####################################
##################################################################################

# st.sidebar.image(image_hiparis, width=200)
# url = "https://www.hi-paris.fr/"
# st.sidebar.markdown("Made in collaboration with the [Hi! PARIS Engineering Team](%s)" % url)

#st.sidebar.markdown("  ")



st.sidebar.header("**Dashboard**") # .sidebar => add widget to sidebar
st.sidebar.markdown("  ")
#st.markdown("  ")
#st.sidebar.divider()
#st.sidebar.markdown("---")


############# SELECT TEACHER  ###############
list_teachers = ["François Derien","Irina Zviadadze","Mian Liu","Teodor Duevski","Quirin Fleckenstein"]
select_teacher = st.sidebar.selectbox('Select your teacher ➡️', list_teachers)


############# SELECT SECTION CODE ###############
list_section_code = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
select_code = st.sidebar.selectbox('Select your section code ➡️', list_section_code)


############# SELECT STUDENT IDS OF GROUP ###############
student_ids = np.arange(1000,2000,50)

session_state = {
    "selected_options": []
}

# Create a multiselect widget
selected_options = st.sidebar.multiselect(
    'Select the id of each group member ➡️',
    student_ids,
    max_selections=3,
    default=session_state["selected_options"]
)

# Update the session state list when the user changes the selection
session_state["selected_options"] = selected_options
select_group = "-".join([str(elem) for elem in session_state["selected_options"]])


# Select lab/exercice number
lab_numbers = st.sidebar.selectbox('Select the exercise ➡️', [
  '01 - One risky and one risk-free asset',
  '02 - Two risky assets',
#   '03 - Diversification',
#   '04 - Test of the CAPM',
  ])



########### TITLE #############

st.title("HEC Paris - Finance Labs🧪")
st.subheader("Portfolio theory 📈")
st.markdown("Course provided by: **François Derrien**, **Irina Zviadadze**, **Mian Liu**, **Teodor Duevski**, **Quirin Fleckenstein**")

st.markdown("  ")
st.markdown("---")

# default text for st.text_area()
default_text = ""






#####################################################################################
#                   EXERCICE 1 - One risky asset, one risk-free asset
#####################################################################################

startdate = datetime.now()

if lab_numbers == "01 - One risky and one risk-free asset": # premiere page

    #################################### SIDEBAR ##################################

    risky_asset = st.sidebar.selectbox("Select a risky asset ➡️", list_risky_assets, key="select_risky")
    #risk_free_asset = "^FVX"

    st.sidebar.markdown("  ")
    



    ################################### DATAFRAMES ###############################
    
    # data_risky = yf.Ticker(risky_asset)
    # df_risky = data_risky.history(period="16mo").reset_index()[["Date","Close","Dividends"]]
    # df_risky = df_risky.loc[(df_risky["Date"]<="2023-07-26") & (df_risky["Date"]>"2022-03-08")] 

    # df_risky["Date"] = pd.to_datetime(df_risky["Date"]).apply(lambda x: x.strftime("%d/%m/%Y"))
    # df_risky.columns = ["Date","Price","Dividends"]

    # Risky asset dataframe (df_risky)
    df_risky = data.loc[data["Stock"]==risky_asset]

    # Riskfree asset dataframe (df_Tbond)
    price_Tbond = [(1 + 0.02)**(1/365) - 1 for i in range(df_risky.shape[0])]
    df_Tbond = pd.DataFrame({"Year":df_risky["Year"].to_list(), "Tbond Price":price_Tbond})

    riskfree_returns = np.array([0.02 for i in range(df_risky.shape[0]-1)])
    riskfree_exp_returns = np.mean(riskfree_returns)
    riskfree_std = np.std(riskfree_returns, ddof=1)
    




    ##################################### TITLE ####################################
    st.markdown("## 01 - One risky and one risk-free asset")
    st.info("In this exercise, assume that there exists a risk-free asset (a T-bond) with an annual rate of return of 2%. You are given information on daily prices and dividends of individual (risky) stocks. You are asked to choose one risky stock and to compute its expected return and standard deviation of return. Then you have to find the (standard deviation of return, expected return) pairs you can obtain by combining this risky stock with the risk-free asset into portfolios.")
    st.markdown("    ")
    st.markdown("    ")





    #################################### QUESTION 1 ###################################

    st.subheader("Question 1 📝")

    #################### Part 1

    ## Title of PART 1
    st.markdown('''<p style="font-size: 22px;"> Please select one stock and <b>compute its realized (holding-period) returns.</b> 
                 Assume that holding, is one day. <br> Next, please <b>compute the expected return</b> and <b>standard deviation</b> of the holding-period returns</b></p>''',
                unsafe_allow_html=True)

    st.markdown("   ")

    # ## View risky dataset
    st.markdown(f"**View the {risky_asset} data** with Date, Dividend and Price.")
    st.dataframe(df_risky.drop(columns=["Stock"]))


    ## Download dataset as xlsx

    # Set the headers to force the browser to download the file
    headers = {
                'Content-Disposition': 'attachment; filename=dataset.xlsx',
                'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }

    # Create a Pandas Excel writer object
    excel_writer = pd.ExcelWriter(f"{risky_asset}.xlsx", engine='xlsxwriter')
    df_risky.drop(columns=["Stock"]).to_excel(excel_writer, index=False, sheet_name='Sheet1')
    excel_writer.close()

    # Download the file
    with open(f"{risky_asset}.xlsx", "rb") as f:
            st.download_button(
                    label=f"📥 **Download the {risky_asset} dataset**",
                    data=f,
                    file_name=f"{risky_asset}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )



    st.markdown("   ")
    st.markdown("   ")


    # Compute holding-period returns, expected returns, std 
    asset1_returns = (df_risky["Price"][1:].to_numpy() - df_risky["Price"][:-1].to_numpy() + df_risky["Dividends"].to_numpy()[1:])/df_risky["Price"][:-1].to_numpy()    
    asset_expected_return = np.mean(asset1_returns)
    asset_std_dev = np.std(asset1_returns, ddof=1)
    


    ####### Holding-period returns
    st.write(f"**Compute the holding-period returns**")

    upload_expected_return = st.file_uploader("Drop your results in an excel file (.xlsx)", key="Q1",type=['xlsx'])

    if upload_expected_return is not None:
        answer_1_Q1_1 = upload_expected_return.name
        returns_portfolios = pd.read_excel(upload_expected_return)

    else:
        answer_1_Q1_1 = np.nan



    solution = st.checkbox('**Solution** ✅',key="SQ1.1")
    if solution:
        returns_result = pd.DataFrame({"Year":df_risky["Year"].iloc[1:], "Return":asset1_returns})
        st.dataframe(returns_result)
        
    st.markdown("  ")
    st.markdown("  ")


    
    ###### Expected returns
    st.write(f"**Compute the expected returns**")
    answer_1_Q1_2 = st.text_input("Enter your results","", key="AQ1.2a")

    solution = st.checkbox('**Solution** ✅',key="SQ1.2a")
    if solution:
        answer_text = f'The expected return of {risky_asset} is **{np.round(asset_expected_return,4)}**.'
        st.success(answer_text)

    st.markdown("  ")
    st.markdown("  ")


    # Standard deviation
    st.write(f"**Compute the standard deviation**")
    answer_1_Q1_3 = st.text_input("Enter your results ","", key="AUQ1.2b")

    solution = st.checkbox('**Solution** ✅',key="SQ1.2b")

    if solution:
        answer_text = f'The standard deviation of {risky_asset} is **{np.round(asset_std_dev,4)}**.'
        st.success(answer_text)


    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ") 





    ###################################### QUESTION 2 ##########################################
       

    st.subheader("Question 2 📝")
    
    ### Part 1
    st.markdown('''<p style="font-size: 22px;">Assume that you have a capital of 1000 EUR that you fully invest in a portfolio. <b>Combine two assets</b> (one risky and one risk-free asset) into a <b>portfolio</b>. Next, <b>compute the expected returns</b> and <b>standard deviation</b> of the portfolio.</p>''',
                unsafe_allow_html=True)
    
    st.info("In this question, assume that **short-sale constraints** are in place (that is, the weight of each asset in your portfolio must be between 0 and 1). ")


    st.markdown("   ")
    st.markdown("   ")

    #st.subheader(f"1. Create a portfolio with {risky_asset} and a risk-free asset")

    #st.markdown("   ")


    # Concatenate graphs for plot
    # df_master = df_risky.merge(df_Tbond, how="inner", on="Date")[["Date","Price","Tbond Price"]].rename(columns={"Price": risky_asset, "Tbond Price": "Risk free"})
    # df_master = df_master.melt(id_vars="Date").rename(columns={"variable":"Asset","value":"Price"})
    # #df_master.columns = ["Date","Stock","Price"]
    
    # chart = alt.Chart(df_master, title="Evolution of stock prices").mark_line().encode(x="Date",y="Price",
    #             color="Asset")
    
    # st.altair_chart((chart).interactive(),use_container_width=True)
    
    
    # df_master = pd.concat([df_risky,df_Tbond]).reset_index(drop=True)
    # df_master = df_master[pd.to_datetime(df_master['Date']) > pd.to_datetime(start_date)]
    # df_master["Date"] = pd.to_datetime(df_master["Date"])


    # chart = alt.Chart(df_master, title="Evolution of stock prices").mark_line().encode(x="Date",y=kpi,
    #             color="symbol",
    #             # strokeDash="symbol",
    #         )
    
    # st.altair_chart((chart).interactive(), use_container_width=True)
    # csv_df = convert_df(df_master)
    # st.markdown("  ")

    
    # Create a portfolio by selecting amount (EUR) in risky asset
    st.write(f"**Select the amount you want to put in {risky_asset}**")

    risky_amount = st.slider(f"**Select the amount you want to put in {risky_asset}**", min_value=0, max_value=1000, step=50, value=500, label_visibility="collapsed")
    riskfree_amount = 1000 - risky_amount
    
    st.write(f"You've invested **{risky_amount}** EUR in {risky_asset} and **{riskfree_amount}** EUR in the risky-free asset.")

    st.markdown("  ")
    st.markdown("  ")



    # Weight of assets in the portfolio
    st.write("**Compute the weight of each asset in your portfolio**")

    risky_weight = risky_amount/1000
    riskfree_weight = riskfree_amount/1000

    answer_1_Q2_1 = st.text_input(f'Enter the weight of the {risky_asset} asset',"", key="AUQ2.1w1")

    # weight1_input = st.number_input(f'Enter the weight of the {risky_asset} asset',value=np.nan)
    # if weight1_input != np.nan:
    #     answer_1_Q2_1 = weight1_input

    # else:
    #     answer_1_Q2_1 = None

    solution = st.checkbox('**Solution** ✅', key="SQ2.1w1")
    if solution:
        answer_text1 = f'The weight of the {risky_asset} asset is **{np.round(risky_weight,2)}**.'
        st.success(answer_text1)


    st.markdown("  ")
    
    answer_1_Q2_2 = st.text_input(f'Enter the weight of the risk-free asset',"", key="AUQ2.1w2")

    solution = st.checkbox('**Solution** ✅',key="SQ2.1w2")
    if solution:
        answer_text = f'The weight of the risk free asset is **{np.round(riskfree_weight,2)}**.'
        st.success(answer_text)

    st.markdown("   ")
    st.markdown("   ") 



    # Compute portfolio returns, expected ret, std
    portfolio_returns = (risky_weight*asset1_returns) + (riskfree_weight*riskfree_returns)
    portfolio_expected_returns = np.mean(portfolio_returns)
    portfolio_std = np.std(portfolio_returns,ddof=1)


    # Enter portfolio expected returns
    st.write("**Compute the expected return of the portfolio**")
    answer_1_Q2_3 = st.text_input("Enter your results","", key="AQ2.21")

    solution = st.checkbox('**Solution** ✅',key="SQ2.21")
    if solution:
        answer_text = f"The portfolio's expected return is **{np.round(portfolio_expected_returns,4)}**"
        st.success(answer_text)

    st.markdown("    ")
    st.markdown("    ")


    # Enter portfolio standard deviation
    st.write("**Compute the standard deviation of the portfolio**")
    answer_1_Q2_4 = st.text_input("Enter your results","", key="AQ2.22")


    solution = st.checkbox('**Solution** ✅',key="SQ2.22")
    if solution:
        answer_text = f"The portfolio's standard deviation is **{np.round(portfolio_std,4)}**"
        st.success(answer_text)


    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ") 





################## QUESTION 3

    st.subheader("Question 3 📝")
    
    #### PART 1
    st.markdown('''<p style="font-size: 22px;"> Using Excel, <b> construct portfolios </b> that contain x% of the risky asset and (1-x)% of the risk-free asset, with x varying between 0 and 100% with 1% increments.
                For each portfolio, calculate its <b>standard deviation</b> of return and its <b>expected return</b>. 
                Represent these combinations in a graph, that is <b>draw the set of feasible portfolios</b>.''',
                unsafe_allow_html=True)
    
    
    # Weights of risky/riskfree in portfolios 
    weight_risky_portfolios = np.arange(0,1.01,0.01)
    weight_riskfree_portfolios = 1 - weight_risky_portfolios
    
    # Expected returns/std of portfolios
    expected_returns_portfolios = np.array([w*asset_expected_return + (1-w)*riskfree_exp_returns for w in weight_risky_portfolios])
    std_portfolios = np.array([(w**asset_std_dev)**2 + ((1-w)*riskfree_std)**2 for w in weight_risky_portfolios])
    std_portfolios = np.sqrt(std_portfolios)

    # Portfolio dataframe to plot
    df_portfolios = pd.DataFrame({f"{risky_asset}":np.round(weight_risky_portfolios,2),
                                  "Risk-free":np.round(weight_riskfree_portfolios,2),
                                  "Expected return":np.round(expected_returns_portfolios,4), 
                                  "Standard deviation":np.round(std_portfolios,4)})
        
    
    st.markdown("   ")
    st.write("**Compute the expected return and standard deviation for each portfolio**")


    upload_file = st.file_uploader("Drop your results in an excel file (.xlsx)", key="Q3.21",type=['xlsx'])

    if upload_file is not None:
        answer_1_Q3_1 = upload_file.name
        returns_portfolios = pd.read_excel(upload_file)

    else:
        answer_1_Q3_1 = np.nan



    solution = st.checkbox('**Solution** ✅',key="SQ3.1")
    if solution:
        st.markdown("  ")
        st.markdown("**Expected return and Standard deviation for each portfolio**")
        st.dataframe(df_portfolios)
        
        # Create a Pandas Excel writer object
        excel_writer = pd.ExcelWriter(f"portfolios_q3.xlsx", engine='xlsxwriter')
        df_portfolios.to_excel(excel_writer, index=False, sheet_name='Sheet1')
        excel_writer.close()

        # Download the file
        # with open(f"portfolios_q3.xlsx", "rb") as f:
        #     st.download_button(
        #             label=f"📥 Download the solution as xlsx",
        #             data=f,
        #             file_name=f"portfolios_q3.xlsx",
        #             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        #         )
            


    st.markdown("   ")
    st.markdown("   ")

    

    
    st.write("**Draw the set of feasible portfolios**")

    upload_graph = st.file_uploader("Drop the graph as an image (jpg, jpeg, png)", key="Q3.23", type=['jpg','jpeg','png'])

    if upload_graph is not None:
        answer_1_Q3_2 = upload_graph.name
        image = Image.open(upload_graph)
        #st.image(image, caption='Graph of the set of feasible portfolios')

    else:
        answer_1_Q3_2 = np.nan


    solution = st.checkbox('**Solution** ✅',key="SQ3.23")
    if solution:
        # Plot set of feasible portfolios 
        st.markdown("  ")
        chart_portfolios = alt.Chart(df_portfolios, title="Set of feasible portolios").mark_circle(size=40).encode(y="Expected return",x="Standard deviation")
        st.altair_chart(chart_portfolios.interactive(), use_container_width=True)


    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ") 




################## QUESTION 4

    st.subheader("Question 4 📝")

    st.markdown('''<p style="font-size: 22px;"> Consider the feasible portfolios from Question 3 and <b> answer the following questions. </b> </p>''',
                unsafe_allow_html=True)
    
    st.info("Provide specific answers, that is, **characterize the portfolios in terms of the weights on both assets**")
    st.markdown("   ")
    
    # EXPECTED RETURN 
    max_exp_row = df_portfolios.iloc[df_portfolios["Expected return"].idxmax(),:]
    max_exp_risky = max_exp_row[f"{risky_asset}"]
    max_exp_riskfree = max_exp_row["Risk-free"]
    max_exp_value = max_exp_row["Expected return"]

    min_exp_row = df_portfolios.iloc[df_portfolios["Expected return"].idxmin(),:]
    min_exp_risky = min_exp_row[f"{risky_asset}"]
    min_exp_riskfree = min_exp_row["Risk-free"]
    min_exp_value = min_exp_row["Expected return"]
    
    
    # STANDARD DEVIATION
    max_std_row = df_portfolios.iloc[df_portfolios["Standard deviation"].idxmax(),:]
    max_std_risky = max_std_row[f"{risky_asset}"]
    max_std_riskfree = max_std_row["Risk-free"]
    max_std_value = max_std_row["Standard deviation"]

    min_std_row = df_portfolios.iloc[df_portfolios["Standard deviation"].idxmin(),:]
    min_std_risky = min_std_row[f"{risky_asset}"]
    min_std_riskfree = min_std_row["Risk-free"]
    min_std_value = min_std_row["Standard deviation"]




    ###### MAX expected return   
    user_input_1 = st.text_area("**Can you find which portfolio has the highest expected return ?**", default_text)
    answer_1_Q4_1 = user_input_1
    solution = st.checkbox('**Solution** ✅',key="SQ4.1")
    if solution:
        st.success(f"The portfolio with **{max_exp_risky}** in the risky asset ({risky_asset}) and **{max_exp_riskfree}** in the risk free asset.")
        st.success(f"The portfolio's expected return is **{np.round(max_exp_value,4)}**")

    st.markdown("   ")
    st.markdown("   ")


    ###### MIN expected return   
    user_input_2 = st.text_area("**Can you find which portfolio has the lowest expected return ?**", default_text)
    answer_1_Q4_2 = user_input_2
    solution = st.checkbox('**Solution** ✅',key="SQ4.2")
    if solution:
        st.success(f"The portfolio with **{min_exp_risky}** in the risky asset ({risky_asset}) and **{min_exp_riskfree}** in the risk free asset.")
        st.success(f"The portfolio's expected return is **{np.round(min_exp_value,4)}**")

    st.markdown("   ")
    st.markdown("   ")



    ###### MAX standard deviation 
    user_input_3 = st.text_area("**Can you find which portfolio has the highest standard deviation ?**", default_text)
    answer_1_Q4_3 = user_input_3
    solution = st.checkbox('**Solution** ✅',key="SQ4.3")
    if solution:
        st.success(f"The portfolio with **{max_std_risky}** in the risky asset ({risky_asset}) and **{max_std_riskfree}** in the risk free asset.")
        st.success(f"The portfolio's standard deviation is **{np.round(max_std_value,4)}**")


    st.markdown("   ")
    st.markdown("   ")
    
    
    ###### MIN standard deviation   
    user_input_4 = st.text_area("**Can you find which portfolio has the lowest standard deviation ?**", default_text)
    answer_1_Q4_4 = user_input_4
    solution = st.checkbox('**Solution** ✅',key="SQ4.4")
    if solution:
        st.success(f"The portfolio with **{min_std_risky}** in the risky asset ({risky_asset}) and **{min_std_riskfree}** in the risk free asset.")
        st.success(f"The portfolio's standard deviation is **{np.round(min_std_value,4)}**")



    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ")





    ######################################################################################
    ##################################### QUESTION 5 #####################################
    ######################################################################################


    st.subheader("Question 5 📝")
    
    st.markdown('''<p style="font-size: 22px;"> <b>Repeat the exercise of Question 3</b>, but with the possibility of selling short one of the two assets. That is, vary x, for example, from -100% to 100%.''',
                unsafe_allow_html=True)
    

    # Weights of risky/riskfree in portfolios 
    weight_risky_portfolios = np.arange(-1,2.01,0.01)
    weight_riskfree_portfolios = 1 - weight_risky_portfolios
    
    # Expected returns/std of portfolios
    expected_returns_portfolios = np.array([w*asset_expected_return + (1-w)*0.02 for w in weight_risky_portfolios])   
    std_portfolios = np.array([(w*asset_std_dev)**2 + ((1-w)*riskfree_std)**2 for w in weight_risky_portfolios])
    std_portfolios = np.sqrt(std_portfolios)

    # Portfolio dataframe to plot
    df_portfolios = pd.DataFrame({f"{risky_asset}":np.round(weight_risky_portfolios,2),
                                  "Risk-free":np.round(weight_riskfree_portfolios,2),
                                  "Expected return":np.round(expected_returns_portfolios,4), 
                                  "Standard deviation":np.round(std_portfolios,4)})
    
    
    st.markdown("   ")

    st.write("**Compute the expected return and standard deviation for each portfolio**")
    upload_expected_return = st.file_uploader("Drop results in an excel file (.xlsx)", key="UQ5.1", type=['xlsx'])
    
    if upload_expected_return is not None:
        answer_1_Q5_1 = upload_expected_return.name
        expected_return_portfolios = pd.read_excel(upload_expected_return)

    else:
        answer_1_Q5_1 = np.nan

    solution = st.checkbox('**Solution** ✅', key="SQ5.1")
    if solution:
        st.markdown("  ")
        st.markdown("**Expected return and Standard deviation for each portfolio** (with short-selling)")
        st.dataframe(df_portfolios)

    
    st.markdown("   ")
    st.markdown("  ")


    st.write("**Draw the set of feasible portfolios**")

    upload_graph = st.file_uploader("Drop graph as an image (jpg, jpeg, png)", key="UQ5.2", type=['jpg','jpeg','png'])
    
    if upload_graph is not None:
        answer_1_Q5_2 = upload_graph.name
        image = Image.open(upload_graph)
        #st.image(image, caption='Graph of the set of feasible portfolios')

    else:
        answer_1_Q5_2 = np.nan
        

    solution = st.checkbox('**Solution** ✅',key="SQ5.2")
    if solution:
        st.markdown("  ")
        chart_portfolios = alt.Chart(df_portfolios, title="Set of feasible portolios with short-selling").mark_circle(size=20).encode(y="Expected return",x="Standard deviation")
        st.altair_chart(chart_portfolios.interactive(), use_container_width=True)
    

    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ")
    



################## QUESTION 6

    st.subheader("Question 6 📝")
    
    st.markdown('''<p style="font-size: 22px;"> <b>Repeat the exercise of Question 4</b>, but with the possibility of <b>selling short</b> one of the two assets. That is, analyze feasible portfolios from Question 5.''',
                unsafe_allow_html=True)
    
    st.markdown("  ")
    
    # EXPECTED RETURN 
    max_exp_row = df_portfolios.iloc[df_portfolios["Expected return"].idxmax(),:]
    max_exp_risky = max_exp_row[f"{risky_asset}"]
    max_exp_riskfree = max_exp_row["Risk-free"]
    max_exp_value = max_exp_row["Expected return"]

    min_exp_row = df_portfolios.iloc[df_portfolios["Expected return"].idxmin(),:]
    min_exp_risky = min_exp_row[f"{risky_asset}"]
    min_exp_riskfree = min_exp_row["Risk-free"]
    min_exp_value = min_exp_row["Expected return"]
    
    
    # STANDARD DEVIATION
    max_std_row = df_portfolios.iloc[df_portfolios["Standard deviation"].idxmax(),:]
    max_std_risky = max_std_row[f"{risky_asset}"]
    max_std_riskfree = max_std_row["Risk-free"]
    max_std_value = max_std_row["Standard deviation"]

    min_std_row = df_portfolios.iloc[df_portfolios["Standard deviation"].idxmin(),:]
    min_std_risky = min_std_row[f"{risky_asset}"]
    min_std_riskfree = min_std_row["Risk-free"]
    min_std_value = min_std_row["Standard deviation"]
       


    ###### PART 1 
    #answer_1_Q4_2 = user_input_2
    answer_1_Q6_1 = st.text_area("**Can you find which portfolio has the highest expected return?**", default_text, key="Q6.1")
    solution = st.checkbox('**Solution** ✅',key="SQ6.1")
    if solution:
        st.success(f"The portfolio with **{max_exp_risky}** in the risky asset ({risky_asset}) and **{max_exp_riskfree}** in the risk free asset.")
        st.success(f"The portfolio's expected return is **{np.round(max_exp_value,4)}**")

    st.markdown("   ")


    ###### PART 2
    answer_1_Q6_2 = st.text_area("**Can you find which portfolio has the lowest expected return?**", default_text, key="Q6.2")
    
    solution = st.checkbox('**Solution** ✅',key="SQ6.2")
    if solution:
        st.success(f"The portfolio with **{min_exp_risky}** in the risky asset ({risky_asset}) and **{min_exp_riskfree}** in the risk free asset.")
        st.success(f"The portfolio's expected return is **{np.round(min_exp_value,4)}**")

    st.markdown("   ")


    ###### PART 3
    answer_1_Q6_3 = st.text_area("**Can you find which portfolio has the highest standard deviation?**", default_text, key="Q6.3")
    solution = st.checkbox('**Solution** ✅',key="SQ6.3")
    if solution:
        st.success(f"The portfolio with **{max_std_risky}** in the risky asset ({risky_asset}) and **{max_std_riskfree}** in the risk-free asset")
        st.success(f"The portfolio's standard deviation is **{np.round(max_std_value,4)}**")

    st.markdown("   ")
    
    
    ###### PART 4
    answer_1_Q6_4 = st.text_area("**Can you find which portfolio has the lowest standard deviation?**", default_text, key="Q6.4")
    solution = st.checkbox('**Solution** ✅',key="SQ6.4")
    if solution:
        st.success(f"The portfolio where you invest **{min_std_riskfree}** in the risky asset ({risky_asset}) and **{min_std_risky}** in the risky-free asset")
        st.success(f"The portfolio's standard deviation is **{np.round(min_std_value,4)}**")

    st.markdown(" ")
    st.markdown(" ")
    st.markdown("#### Congratulations you finished Exercise 1 🎉")
    


    list_answer = [answer_1_Q1_1,
answer_1_Q1_2,
answer_1_Q1_3,
answer_1_Q2_1,
answer_1_Q2_2,
answer_1_Q2_3,
answer_1_Q2_4,
answer_1_Q3_1,
answer_1_Q3_2,
answer_1_Q4_1,
answer_1_Q4_2,
answer_1_Q4_3,
answer_1_Q4_4,
answer_1_Q5_1,
answer_1_Q5_2,
answer_1_Q6_1,
answer_1_Q6_2,
answer_1_Q6_3,
answer_1_Q6_4,]

#     count = len([x for x in list_answer if x not in [0, 0.0, None, "Write the answer in this box","0"]])
#     df_1 = pd.DataFrame({
#     'Professor': select_teacher,
#     'Section': select_code,
#     'Group': select_group,
#     'Part1': 1,
#     'Start time':startdate,
#     'End time': '05/09/2023 10:40',
#     'Completed':count,
#     'Completed %':round(count/19*100,2),
#     'Q1_1':answer_1_Q1_1,
#     'Q1_2':answer_1_Q1_2,
#     'Q1_3':answer_1_Q1_3,
#     'Q2_1':answer_1_Q2_1,
#     'Q2_2':answer_1_Q2_2,
#     'Q2_3':answer_1_Q2_3,
#     'Q2_4':answer_1_Q2_4,
#     'Q3_1':answer_1_Q3_1,
#     'Q3_2':answer_1_Q3_2,
#     'Q4_1':answer_1_Q4_1,
#     'Q4_2':answer_1_Q4_2,
#     'Q4_3':answer_1_Q4_3,
#     'Q4_4':answer_1_Q4_4,
#     'Q5_1':answer_1_Q5_1,
#     'Q5_2':answer_1_Q5_2,
#     'Q6_1':answer_1_Q6_1,
#     'Q6_2':answer_1_Q6_2,
#     'Q6_3':answer_1_Q6_3,
#     'Q6_4':answer_1_Q6_4
#     })

    #st.dataframe(df_1)
    if st.sidebar.button('**Submit answers Ex1**'):

        if len(session_state["selected_options"]) != 0:
            select_group = "-".join([str(elem) for elem in session_state["selected_options"]])
            
            answers = [x for x in list_answer if x not in [np.nan,"", None]]
            count = len(answers)

            df_1 = pd.DataFrame({
            'Professor': select_teacher,
            'Section': select_code,
            'Group': select_group,
            'Lab': 1,
            # "Stock": risky_asset,
            # 'Time': startdate - datetime.now(),
            'Start time':startdate,
            'End time': datetime.now(),
            'Completed':count,
            'Completed %':round((count/len(list_answer))*100,1),
            'Q1_1':answer_1_Q1_1,
            'Q1_2':answer_1_Q1_2,
            'Q1_3':answer_1_Q1_3,
            'Q2_1':answer_1_Q2_1,
            'Q2_2':answer_1_Q2_2,
            'Q2_3':answer_1_Q2_3,
            'Q2_4':answer_1_Q2_4,
            'Q3_1':answer_1_Q3_1,
            'Q3_2':answer_1_Q3_2,
            'Q4_1':answer_1_Q4_1,
            'Q4_2':answer_1_Q4_2,
            'Q4_3':answer_1_Q4_3,
            'Q4_4':answer_1_Q4_4,
            'Q5_1':answer_1_Q5_1,
            'Q5_2':answer_1_Q5_2,
            'Q6_1':answer_1_Q6_1,
            'Q6_2':answer_1_Q6_2,
            'Q6_3':answer_1_Q6_3,
            'Q6_4':answer_1_Q6_4, 
            },index=[0])

            path_results = r"results/"

            if "App_results_Ex1.csv" not in os.listdir(path_results):
                df_1.to_csv(os.path.join(path_results,"App_results_Ex1.csv"),index=False)

            else:
                df_old = pd.read_csv(os.path.join(path_results,"App_results_Ex1.csv"))
                result = pd.concat([df_old, df_1], ignore_index=True)
                result.to_csv(os.path.join(path_results,"App_results_Ex1.csv"),index=False)
            
            st.sidebar.info('**Your answers have been submitted !**')
            #st.dataframe(df_1)

        else:
            st.sidebar.info("**Please select at least one student id before submitting your answers !**")

    # st.sidebar.divider()
    # st.sidebar.header("****Acknowledgement****")
    st.sidebar.divider()
    st.sidebar.image(image_hiparis, width=150)
    url = "https://www.hi-paris.fr/"
    st.sidebar.markdown("**Made in collaboration with: [Hi! PARIS Engineering Team](%s)**" % url)
    
    







#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





#################################################################################################################
#                                        EXERCICE 2 - Two risky assets
#################################################################################################################

if lab_numbers == "02 - Two risky assets":

    ##################################### SIDEBAR ##########################################
    
    output_multiselect = st.sidebar.multiselect("Select two risky stocks", list_risky_assets, ["AIR","CECO"])

     ##################################### TITLE ##########################################
    st.markdown("## 02 - Two risky assets")
    
    if len(output_multiselect) != 2:
        st.info("**You must select two risky stocks to start the second exercice !**")
    
    else:
        risky_asset1_ex2, risky_asset2_ex2  = output_multiselect
        st.info("The purpose of this exercise is to understand how to **construct efficient portfolios** if you can invest in two risky assets or in two risky and one risk-free asset.")
    
        st.sidebar.markdown("  ")
        
        ##################################### QUESTION 1 #####################################
        st.markdown("   ")
        st.markdown("   ")

        st.subheader("Question 1 📝")
        
        ########### Q1 PART 1
        st.markdown('''<p style="font-size: 22px;"> Download prices for two risky stocks. <b>Compute their realized returns</b>.  
                    Next, estimate the <b>expected returns</b> and <b>standard deviations of returns</b> on these two stocks. 
                    Finally, compute the <b>correlation of the returns</b> on these two stocks.''',
                    unsafe_allow_html=True)

        st.markdown("  ")
        st.markdown("  ")

        # st.markdown('''<p style="font-size: 20px;"> <b>First asset</b></p>''',
        #             unsafe_allow_html=True)

        #st.divider()
        st.subheader(f"First risky stock ({risky_asset1_ex2}) 📋")
        st.markdown("  ")

  
        ######################## RISKY ASSET 1 ############################

        ## Dataframe 
        df_asset1_ex2 = data.loc[data["Stock"]==risky_asset1_ex2].drop(columns=["Stock"])

        st.markdown(f"View the **Date**, **Price** and **Dividends** of {risky_asset1_ex2}")
        st.dataframe(df_asset1_ex2)


        ## Download merge dataframe as xlsx
        headers = {
                    'Content-Disposition': 'attachment; filename=dataset.xlsx',
                    'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                }

        excel_writer = pd.ExcelWriter(f"{risky_asset1_ex2}_Ex2.xlsx", engine='xlsxwriter')
        df_asset1_ex2.to_excel(excel_writer, index=False, sheet_name='Sheet1')
        excel_writer.close()

        ## Download the file
        with open(f"{risky_asset1_ex2}_Ex2.xlsx", "rb") as f:
                st.download_button(
                        label=f"📥 **Download the {risky_asset1_ex2} dataset**",
                        data=f,
                        file_name=f"{risky_asset1_ex2}_Ex2.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )


        st.markdown(" ")
        #st.markdown(" ")



        ## Compute holding-period returns, expected returns, std 
        asset1_returns = (df_asset1_ex2[f"Price"][1:].to_numpy() - df_asset1_ex2[f"Price"][:-1].to_numpy() + df_asset1_ex2[f"Dividends"].to_numpy()[1:])/df_asset1_ex2[f"Price"][:-1].to_numpy()    
        asset1_expected_return = np.mean(asset1_returns)
        asset1_std_dev = np.std(asset1_returns, ddof=1) 


        ## Holding-period returns
        st.write(f"**Compute the holding-period returns**")

        # Upload answer
        upload_file = st.file_uploader("Drop your results in an excel file (.xlsx)", key="Ex2.Q1.11",type=['xlsx'])
        
        if upload_file is not None:
            answer_2_Q1_1a = upload_file.name
        
        else:
            answer_2_Q1_1a = np.nan

        # Solution
        solution = st.checkbox('**Solution** ✅',key="SQ2.11")
        if solution:
            st.markdown(" ")
            st.markdown(f"**Holding-period returns of {risky_asset1_ex2}**")

            df_returns1 = pd.DataFrame({"Year":df_asset1_ex2["Year"].iloc[1:],f"Returns ({risky_asset1_ex2})":asset1_returns})
            st.dataframe(df_returns1)

        st.markdown("  ")
        st.markdown("  ")


        ## Expected returns
        st.write(f"**Compute the expected return**")
        answer_2_Q1_1b = st.text_input("Enter your results","", key="Ex2.Q1.12")

        solution = st.checkbox('**Solution** ✅',key="SQ2.12")
        if solution:
            st.success(f"The expected return of {risky_asset1_ex2} is **{np.round(asset1_expected_return,3)}.**")

        st.markdown("  ")
        st.markdown("  ")


        ## Standard deviation of returns
        st.write(f"**Compute the standard deviation**")
        answer_2_Q1_1c = st.text_input(f"Enter your results","", key="Ex2.Q2.13")
        
        solution = st.checkbox('**Solution** ✅',key="SQ2.13")
        if solution:
            st.success(f"The standard deviation of {risky_asset1_ex2} is **{np.round(asset1_std_dev,3)}.**")

        st.markdown("  ")
        st.markdown("  ")






        ######################## RISKY ASSET 2 ############################

        st.subheader(f"Second risky stock ({risky_asset2_ex2}) 📋")
        st.markdown("  ")

        ## Dataset
        df_asset2_ex2 = data.loc[data["Stock"]==risky_asset2_ex2].drop(columns=["Stock"])

        ## View the dataset
        st.markdown(f"View the **Date**, **Price** and **Dividends** of {risky_asset2_ex2}")
        st.dataframe(df_asset2_ex2)


        ## Download dataframe as xlsx
        headers = {
                    'Content-Disposition': 'attachment; filename=dataset.xlsx',
                    'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                }

        ## Create excel object from dataframe
        excel_writer = pd.ExcelWriter(f"{risky_asset2_ex2}_Ex2.xlsx", engine='xlsxwriter')
        df_asset2_ex2.to_excel(excel_writer, index=False, sheet_name='Sheet1')
        excel_writer.close()

        ## Download the file
        with open(f"{risky_asset2_ex2}_Ex2.xlsx", "rb") as f:
                st.download_button(
                        label=f"📥 **Download the {risky_asset2_ex2} dataset**",
                        data=f,
                        file_name=f"{risky_asset2_ex2}_Ex2.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                
        
        st.markdown(" ")
        st.markdown(" ")



        ## Compute holding-period returns, expected returns, std 
        asset2_returns = (df_asset2_ex2[f"Price"][1:].to_numpy() - df_asset2_ex2[f"Price"][:-1].to_numpy() + df_asset2_ex2[f"Dividends"].to_numpy()[1:])/df_asset2_ex2[f"Price"][:-1].to_numpy()    
        asset2_expected_return = np.mean(asset2_returns)
        asset2_std_dev = np.std(asset2_returns, ddof=1)



        ## Holding-period returns
        st.write(f"**Compute the holding-period returns**")
        
        upload_expected_return = st.file_uploader("Drop your results in an excel file (.xlsx)", key="Ex2.Q1.21",type=['xlsx'])

        if upload_expected_return is not None:
            answer_2_Q1_2a = upload_expected_return.name

        else:
            answer_2_Q1_2a = np.nan

        solution = st.checkbox('**Solution** ✅',key="SQ2.21")
        if solution:
            st.markdown(" ")
            st.markdown(f"**Holding-period returns of {risky_asset2_ex2}**")

            df_returns2 = pd.DataFrame({"Year":df_asset2_ex2["Year"].iloc[1:],f"Returns ({risky_asset2_ex2})":asset2_returns})
            st.dataframe(df_returns2)

        st.markdown("  ")
        st.markdown("  ")


        ## Expected returns
        st.write(f"**Compute the expected return**")
        answer_2_Q1_2b = st.text_input("Enter your results","", key="Ex2.Q1.22")

        solution = st.checkbox('**Solution** ✅',key="SQ2.22")
        if solution:
            st.success(f"The expected return of {risky_asset2_ex2} is **{np.round(asset2_expected_return,3)}.**")

        st.markdown("  ")
        st.markdown("  ")


        ## Standard deviation
        st.write(f"**Compute the standard deviation**")
        answer_2_Q1_2c = st.text_input(f"Enter your results","", key="Ex2.Q2.23")

        solution = st.checkbox('**Solution** ✅',key="SQ2.23")
        if solution:
            st.success(f"The standard deviation of {risky_asset2_ex2} is **{np.round(asset2_std_dev,3)}**.")

        st.markdown("  ")
        st.markdown("  ")



        
        ############## CORRELATION ASSET 1 AND ASSET 2 ##############
        
        st.subheader(f"Correlation between {risky_asset1_ex2} and {risky_asset2_ex2} 📈")
        st.markdown("  ")

        # Compute correlation between assets 
        asset12_corr = np.corrcoef(asset1_returns,asset2_returns)[0,1]


        # Input answers
        st.write(f"**Compute the correlation between both assets**")
        answer_2_Q1_3 = st.text_input(f"Enter your results","", key="Ex2.Q2.3")

        solution = st.checkbox('**Solution** ✅',key="SQ2.3")
        if solution:
            st.success(f"The correlation between both assets is **{np.round(asset12_corr,3)}**.")


        # ## Merge dataframes with asset 1 and 2 to plot
        # df_asset1_ex2.columns = ["Year",f"Price {risky_asset1_ex2}",f"Dividends {risky_asset1_ex2}"]
        # df_asset2_ex2.columns = ["Year",f"Price {risky_asset2_ex2}",f"Dividends {risky_asset2_ex2}"]
        # df_merge_ex2 = df_asset1_ex2.merge(df_asset2_ex2, how="inner", on="Year")

        
        st.markdown("   ")
        st.markdown("   ") 
        st.markdown("   ")     
        st.markdown("   ")
        st.markdown("   ")






        ######################################################################################
        ##################################### QUESTION 2 #####################################
        ######################################################################################

        st.subheader("Question 2 📝")

        st.markdown('''<p style="font-size: 22px;"> Compose different <b>portfolios of two risky assets</b> by investing in one risky asset x% of your wealth and in the other asset (1-x)%.
                    Vary x from -50% to 150% with an increment of 5%. Compute the <b>expected returns</b> and <b>standard deviations</b> of the resulting portfolios.''', 
                    unsafe_allow_html=True)
        
        st.info("**Hint**: Do not forget about the correlation between the returns on these two stocks.")

        st.markdown("  ")
        st.markdown("  ")

        ######## Plot evolution asset 1 and asset 2 prices
        # df_plot_ex2 = df_merge_ex2.copy().drop(columns=[f"Dividends {risky_asset1_ex2}",f"Dividends {risky_asset2_ex2}"])
        # df_plot_ex2.columns = ["Date", f"{risky_asset1_ex2}", f"{risky_asset2_ex2}"]
        # df_plot_ex2 = df_plot_ex2.melt(id_vars=["Date"])
        # df_plot_ex2.columns = ["Date","Stock","Price"]
        
        # chart = alt.Chart(df_plot_ex2, title="View the evolution of stock prices").mark_line().encode(x="Date",y="Price",color="Stock")
        # st.altair_chart(chart.interactive(), use_container_width=True)

        # st.markdown("  ")


        # Weights & realized returns
        weight_portfolios = np.round(np.arange(-0.5,1.55,0.05),2)
        returns_portfolios = np.array([w*asset1_returns + (1-w)*asset2_returns for w in weight_portfolios])


        # Compute expected return and std of each portfolio 
        expected_returns_portfolios = np.array([w*asset1_expected_return + (1-w)*asset2_expected_return for w in weight_portfolios])
        std_portfolios = np.array([(w*asset1_std_dev)**2 + ((1-w)*asset2_std_dev)**2 + 2*w*(1-w)*asset12_corr*asset1_std_dev*asset2_std_dev for w in weight_portfolios])
        std_portfolios = np.sqrt(std_portfolios)

        df_exp_std_return_portfolios = pd.DataFrame({risky_asset1_ex2:np.round(weight_portfolios,2),
                                                     risky_asset2_ex2:np.round(1-weight_portfolios,2),
                                                    "Expected return":np.round(expected_returns_portfolios,4), 
                                                    "Standard deviation":np.round(std_portfolios,4)})
        

        # Feasible portfolios graph
        chart_portfolios = alt.Chart(df_exp_std_return_portfolios).mark_circle(size=40).encode(y="Expected return",x="Standard deviation")



        st.write("**Compute the expected return and standard deviation for each portfolio**")

        upload_expected_return = st.file_uploader("Drop your results in an excel file (.xlsx)", key="Q3.21", type=['xlsx'])
        if upload_expected_return is not None:
            answer_2_Q2 = upload_expected_return.name

        else:
            answer_2_Q2 = np.nan


        solution = st.checkbox('**Solution** ✅', key="SQ3.1")
        if solution:
            st.dataframe(df_exp_std_return_portfolios)
            

        # st.markdown("   ")
        # st.markdown("   ")
        


        # st.write("**Draw the feasible portfolios** 📉")


        # upload_graph = st.file_uploader("Drop graph as an image (jpg, jpeg, png)", key="Ex2Q3.23", type=['jpg','jpeg','png'])
        # if upload_graph is not None:

        #     image = Image.open(upload_graph)
        #     st.image(image, caption='Graph of the set of feasible portfolios')
            

        # st.markdown("   ")

        # solution = st.checkbox('**Solution** ✅',key="Ex2SQ3.23")
        # if solution:
        #     st.altair_chart(chart_portfolios.interactive(), use_container_width=True)
        



        st.markdown("   ")
        st.markdown("   ") 
        st.markdown("   ")     
        st.markdown("   ")
        st.markdown("   ") 






        ##################################### QUESTION 3 #####################################

        st.subheader("Question 3 📝")

        st.markdown('''<p style="font-size: 22px;"> Indicate the set of <b>feasible portfolios</b> and the set of <b>efficient portfolios</b>. Next, <b>draw a graph in which you represent the portfolios</b>, that is, the sigma-expected return pairs, you obtain with different combinations of the two risky assets.''', 
                    unsafe_allow_html=True)
        
        st.markdown(" ")

        ## Set of feasible portfolio
        st.write("**What is the set of feasible portfolios ?**")
        
        answer_2_Q3_1 = st.text_area("Write your answer here", default_text, key="Q3.Ex2.11")

        solution = st.checkbox('**Solution** ✅',key="SQ3.Ex2.11")
        if solution:
            st.success(f"The set of all portfolios (standard deviation-expected return) that can be obtained by building portfolios with {risky_asset1_ex2} and {risky_asset2_ex2}.")


        st.markdown("  ")
        st.markdown("   ")


        ## Set of efficient portfolios
        st.write("**What is the set of efficient portfolios ?**")

        upload_efficient_portfolios = st.file_uploader("Drop your results in an excel file (.xlsx)", key="Q3.Ex2.U12",type=['xlsx'])
        if upload_efficient_portfolios is not None:
            answer_2_Q3_2 = upload_efficient_portfolios

        else:
            answer_2_Q3_2 = np.nan

        min_std = df_exp_std_return_portfolios["Standard deviation"].idxmin()

        solution = st.checkbox('**Solution** ✅',key="SQ3.Ex2.12")
        if solution:
            st.success("The portfolios that offer the greatest expected rate of return for each level of standard deviation (risk).")
            st.dataframe(df_exp_std_return_portfolios[:(min_std+1)])

        
        st.markdown("  ")
        st.markdown("   ")
        


        ## Draw the set of feasible portfolios 
        st.write("**Draw the set of feasible portfolios**")

        upload_graph = st.file_uploader("Drop the graph as an image (jpg, jpeg, png)", key="Q3.Ex2.13", type=['jpg','jpeg','png'])
        if upload_graph is not None:
            answer_2_Q3_3 = upload_graph.name
        
        else:
            answer_2_Q3_3 = np.nan
            

        solution = st.checkbox('**Solution** ✅',key="SQ3.23")
        if solution:
            st.altair_chart(chart_portfolios.interactive(), use_container_width=True)

        st.markdown("   ")
        st.markdown("   ") 
        st.markdown("   ")     
        st.markdown("   ")
        st.markdown("   ")





        ##################################### QUESTION 4 #####################################

        st.subheader("Question 4 📝")

        st.markdown('''<p style="font-size: 22px;"> Assume that you cannot short-sell any of the risky assets (only in this exercise). 
                    Indicate the new <b>set of feasible portfolios</b> and the new <b>set of efficient portfolios</b>.''', 
                    unsafe_allow_html=True)
        

        #df_exp_std_return_portfolios_v2 = df_exp_std_return_portfolios.copy()
        #df_exp_std_return_portfolios_v2[risky_asset1_ex2] = df_exp_std_return_portfolios_v2[risky_asset1_ex2].apply(lambda x: f"{np.abs(float(x.split('%')[0]))}%")
        
        df_exp_std_return_portfolios_q4 = df_exp_std_return_portfolios.loc[(df_exp_std_return_portfolios[risky_asset1_ex2]>=0) & (df_exp_std_return_portfolios[risky_asset2_ex2]>=0)]
        
        st.markdown("  ")

        ## Set of feasible portfolios
        st.write("**What is the set of feasible portfolios ?**")
        
        upload_portfolios = st.file_uploader("Drop your results in an excel file (.xlsx)", key="Q4.Ex2.U11",type=['xlsx'])
        if upload_portfolios is not None:
            answer_2_Q4_1 = upload_portfolios.name

        else:
            answer_2_Q4_1 = np.nan

        solution = st.checkbox('**Solution** ✅',key="Q4.Ex2.S11")
        if solution:
            st.success(f"The set of feasible portfolios are the portfolios with only positive or null weights in {risky_asset1_ex2} and {risky_asset2_ex2}.")
            st.dataframe(df_exp_std_return_portfolios_q4)
            
        st.markdown("  ")
        st.markdown("  ")


        ## Set of efficient portfolios
        st.write("**What is the set of efficient portfolios ?**")
        upload_efficient_portfolios = st.file_uploader("Drop your results in an excel file (.xlsx)", key="Q4.Ex2.U12",type=['xlsx'])
        
        if upload_efficient_portfolios is not None:
            answer_2_Q4_2 = upload_efficient_portfolios.name

        else:
            answer_2_Q4_2 = np.nan

        
        min_std = df_exp_std_return_portfolios_q4["Standard deviation"].idxmin()

        solution = st.checkbox('**Solution** ✅',key="SQ4.Ex2.S12")
        if solution:
            st.dataframe(df_exp_std_return_portfolios_q4.iloc[:(min_std+1),:])
            #st.success("All of the feasible portfolios are efficient when short-selling isn't possible.")
            
        st.markdown("   ")
        st.markdown("   ") 
        st.markdown("   ")     
        st.markdown("   ")
        st.markdown("   ")





        ##################################### QUESTION 5 #####################################

        st.subheader("Question 5 📝")

        st.markdown('''<p style="font-size: 22px;"> Assume that you also have a risk-free asset with a rate of return of 2% per annum. 
                    <b>Find the tangency portfolio</b>.''', 
                    unsafe_allow_html=True)
        
        st.info("**Hint**: Compute the Sharpe ratio (the reward-to-variability ratio) for all feasible portfolios in Question 2. Find the portfolio with the maximal one.")


        riskfree_rate = 0.02
        df_exp_std_return_portfolios["Sharpe Ratio"] = np.round(((df_exp_std_return_portfolios["Expected return"] - riskfree_rate)/df_exp_std_return_portfolios["Standard deviation"]),4)
        
        max_sharpe_ratio = df_exp_std_return_portfolios["Sharpe Ratio"].idxmax()
        max_sharpe_ratio_row = df_exp_std_return_portfolios.iloc[max_sharpe_ratio,:]
        
        # Portfolio with max sharpe ratio
        max_sharpe_weight1, max_sharpe_weight2 = max_sharpe_ratio_row[risky_asset1_ex2], max_sharpe_ratio_row[risky_asset2_ex2]
        max_sharpe, max_sharpe_expected, max_sharpe_std = max_sharpe_ratio_row["Sharpe Ratio"], max_sharpe_ratio_row["Expected return"], max_sharpe_ratio_row["Standard deviation"]

        st.markdown(" ")

        st.write("**What is the tangency portfolio ?**")
        answer_2_Q5 = st.text_area("Write your answer here", default_text, key="UQ5.Ex2")


        solution = st.checkbox('**Solution** ✅',key="SQ5.Ex2")
        if solution:
            st.success(f"The tangency portfolio is the portfolio where you invest **{max_sharpe_weight1}** in {risky_asset1_ex2} and **{max_sharpe_weight2}** in {risky_asset2_ex2}, with a sharpe ratio of **{max_sharpe}**.")
            #st.success(f"The tangency portfolio's expected return is **{np.round(max_sharpe_expected,5)}** and its standard deviation is **{np.round(max_sharpe_std,5)}**")
            st.dataframe(df_exp_std_return_portfolios.style.highlight_max(color="lightgreen", subset="Sharpe Ratio",axis=0))
    
            
        st.markdown("   ")
        st.markdown("   ") 
        st.markdown("   ")     
        st.markdown("   ")
        st.markdown("   ")



        ######################################### QUESTION 6 #########################################

        st.subheader("Question 6 📝")

        ### Q5 PART 1
        st.markdown('''<p style="font-size: 22px;">Indicate the <b>set of efficient portfolios</b> that you can achieve if you invest in two risky assets and one risk-free asset.''', 
                    unsafe_allow_html=True)
        
        # Weight in risky asset 1, risky asset 2, the risky portfolio R and the risk free asset
        weight_portfoliosR = np.round(np.arange(-0.5,1.55,0.05),2)
        weight_riskfree = 1 - weight_portfoliosR

        weight_risk1_full = []
        weight_risk2_full = []
        weight_riskportfolio = []
        weight_riskfree = []

        # Weights in risky portfolio R (weight=1 for risky portfolio R)
        weight_risk1_portfolioR = []
        weight_risk2_portfolioR = []

        for wp in weight_portfoliosR:
            for w1 in weight_portfoliosR:
                weight_risk1_full.append(w1)
                weight_risk2_full.append(wp-w1)
                weight_riskportfolio.append(wp)
                weight_riskfree.append(1-wp)

                weight_risk1_portfolioR.append(w1/wp)
                weight_risk2_portfolioR.append((wp-w1)/wp)



        df_full_portfolio = pd.DataFrame({risky_asset1_ex2:np.round(weight_risk1_full,2),
                                        f"{risky_asset1_ex2} (in risky portfolio)":np.round(weight_risk1_portfolioR,2),
                                        risky_asset2_ex2:np.round(weight_risk2_full,2),
                                        f"{risky_asset2_ex2} (in risky portfolio)":np.round(weight_risk2_portfolioR,2),
                                        "risky portfolio":np.round(weight_riskportfolio,2),
                                        "risk-free":np.round(weight_riskfree,2)})
        

        #st.dataframe(df_full_portfolio)

        ## Note: A portfolio is efficient if and only if it is a combination of the riskless asset and the tangency portfolio T. ##

        # Compute returns of portfolio
        riskfree_returns = np.repeat(0.02,len(asset1_returns))
        returns_portfolios_risky = np.array([w1*asset1_returns + w2*asset2_returns + w3*riskfree_returns for w1, w2, w3 in zip(weight_risk1_full,weight_risk2_full,weight_riskfree)])



        # Compute expected return and std of each portfolio 
        expected_returns_portfolios = np.array([w1*asset1_expected_return + w2*asset2_expected_return + w3*0.02 for w1,w2,w3 in zip(weight_risk1_full,weight_risk2_full,weight_riskfree)])
        std_portfolios = np.array([(w1*asset1_std_dev)**2 + (w2*asset2_std_dev)**2 + 2*w1*w2*asset12_corr*asset1_std_dev*asset2_std_dev for w1,w2 in zip(weight_risk1_full,weight_risk2_full)])
        std_portfolios = np.sqrt(std_portfolios)

        df_full_portfolio["Expected return"] = expected_returns_portfolios
        df_full_portfolio["Standard deviation"] = std_portfolios

        # Find efficient portfolios 
        df_efficient_portfolios = df_full_portfolio.loc[(df_full_portfolio[f"{risky_asset1_ex2} (in risky portfolio)"]==max_sharpe_weight1) & (df_full_portfolio[f"{risky_asset2_ex2} (in risky portfolio)"]==max_sharpe_weight2)].drop(columns=[f"{risky_asset1_ex2} (in risky portfolio)",f"{risky_asset2_ex2} (in risky portfolio)"])
        st.markdown(" ")


        ## Efficient portfolios ?
        st.write("**What is the set of efficient portfolios ?**")
        upload_efficient_portfolios = st.file_uploader("Drop your results in an excel file (.xlsx)", key="UQ6.Ex6",type=['xlsx'])
        if upload_efficient_portfolios is not None:
            answer_2_Q6 = upload_efficient_portfolios
        else:
            answer_2_Q6 = np.nan

        solution = st.checkbox('**Solution** ✅',key="SQ6.Ex2")
        
        if solution:
            st.success("The efficient portfolios are the portfolios with a **combination of the risk-free asset and the tangency portfolio of Question 5.**")
            st.dataframe(df_efficient_portfolios)
            st.markdown(f"**Important**: The weights in {risky_asset1_ex2} and {risky_asset2_ex2} where computed based on their weights in the overall portfolio, not on their weight in a portfolio with only risky assets (risky portfolio).")

        st.markdown(" ")
        st.markdown(" ")
        st.markdown("#### Congratulations you finished Exercise 2 🎉")




        ######### SUBMIT ANSWERS #########
        list_answer = [answer_2_Q1_1a,
        answer_2_Q1_1b,
        answer_2_Q1_1c,
        answer_2_Q1_2a,
        answer_2_Q1_2b,
        answer_2_Q1_2c,
        answer_2_Q1_3,
        answer_2_Q2,
        answer_2_Q3_1,
        answer_2_Q3_2,
        answer_2_Q3_3,
        answer_2_Q4_1,
        answer_2_Q4_2,
        answer_2_Q5,
        answer_2_Q6]

        if st.sidebar.button('**Submit answers Ex1**'):

            if len(session_state["selected_options"]) != 0:
                select_group = "-".join([str(elem) for elem in session_state["selected_options"]])
                
                answers = [x for x in list_answer if x not in [np.nan,"", None]]
                count = len(answers)

                df_2 = pd.DataFrame({
                'Professor': select_teacher,
                'Section': select_code,
                'Group': select_group,
                'Lab': 2,
                # "Stock": risky_asset,
                # 'Time': startdate - datetime.now(),
                'Start time':startdate,
                'End time': datetime.now(),
                'Completed':count,
                'Completed %':round((count/len(list_answer))*100,1),
                'Q1_1a':answer_2_Q1_1a,
                'Q1_1b':answer_2_Q1_1b,
                'Q1_1c':answer_2_Q1_1c,
                'Q1_2a':answer_2_Q1_2a,
                'Q1_2b':answer_2_Q1_2b,
                'Q1_2c':answer_2_Q1_2c,
                'Q1_3':answer_2_Q1_3,
                'Q2':answer_2_Q2,
                'Q3_1':answer_2_Q3_1,
                'Q3_2':answer_2_Q3_2,
                'Q3_3':answer_2_Q3_3,
                'Q4_1':answer_2_Q4_1,
                'Q4_2':answer_2_Q4_2,
                'Q5': answer_2_Q5,
                'Q6':answer_2_Q6,
                },index=[0])

                path_results = r"results/"

                if "App_results_Ex2.csv" not in os.listdir(path_results):
                    pd.DataFrame(columns=df_2.columns).to_csv("App_results_Ex2.csv",index=False)

                else:
                    df_old = pd.read_csv(os.path.join(path_results,"App_results_Ex2.csv"))
                    result = pd.concat([df_old, df_2], ignore_index=True)
                    result.to_csv(os.path.join(path_results,"App_results_Ex1.csv"),index=False)
                
                st.sidebar.info('**Your answers have been submitted !**')
                st.dataframe(df_2)
            
            else:
                st.sidebar.info("**Please select at least one student id before submitting your answers.**")






# if lab_numbers == "03 - Diversification":

#     st.info("This page is a work in progress. Please check back later.")
#     st.markdown(" ")
#     st.markdown(" ")
#     st.markdown("#### Congratulations you finished Exercise 3 🎉")


# if lab_numbers == "04 - Test of the CAPM":

    
#     st.info("This page is a work in progress. Please check back later.")
#     st.markdown(" ")
#     st.markdown(" ")
#     st.markdown("#### Congratulations you finished Exercise 4 🎉")
























if __name__=='__main__':
    main()

#st.markdown(" ")
#st.markdown("### 👨🏼‍💻 **App Contributors:** ")
#st.image(['images/gaetan.png'], width=100,caption=["Gaëtan Brison"])

#st.markdown(f"####  Link to Project Website [here]({'https://github.com/gaetanbrison/app-predictive-analytics'}) 🚀 ")
#st.markdown(f"####  Feel free to contribute to the app and give a ⭐️")


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;background - color: white}
     .stApp { bottom: 80px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,

    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer2():
    myargs = [
        " Made in collaboration with: ",
        link("https://www.hi-paris.fr/", "Hi! PARIS Engineering Team"),
        "👨🏼‍💻"
    ]
    layout(*myargs)


# if __name__ == "__main__":
#     footer2()

