import pandas as pd
from pandas import to_datetime
from pandas.plotting import register_matplotlib_converters
import numpy as np
from pathlib import Path
import base64
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

image_hec = Image.open('images/hec.png')
st.image(image_hec, width=300)




########### DASHBOARD PART ###############

st.sidebar.header("Dashboard") # .sidebar => add widget to sidebar
st.sidebar.markdown("---")

#st.sidebar.number_input("**🪪 Input your groups student numbers:**",67609)


# Add multiple student numbers

import numpy as np

student_ids = np.arange(1000,2000,50)

st.sidebar.multiselect(
    'Input the student id of each group member',
    student_ids
)


lab_numbers = st.sidebar.selectbox('Select the exercise ➡️', [
  '01 - One risky and one risk-free asset',
  '02 - Two risky assets',
  '03 - Diversification',
  '04 - Test of the CAPM',
  ])

#st.sidebar.header("Select Stock Symbol")
# list_risky_assets = ['AAPL', 'AMZN', 'IBM','MSFT','TSLA','NVDA',
#                     'PG','JPM','WMT','CVX','BAC','PFE','GOOG',
#                 'ADBE','AXP','BBY','BA','CSCO','C','DIS','EBAY','ETSY','GE','INTC','JPM']

list_risky_assets = ['AAPL', 'AMZN', 'IBM','MSFT','TSLA','NVDA',
                     'PG','JPM','WMT','CVX','BAC','PFE','GOOG',
                    'ADBE','AXP','BBY','BA','ETSY','GE','INTC','JPM']


# Example of riskfree assets
list_riskfree_assets = ['CSCO','C','DIS','EBAY']


dictionary_symbols = {
    'AAPL':'Apple',
    'AMZN':'Amazon',
    'IBM':'IBM',
    'MSFT':'Microsoft',
    'TSLA':'Tesla',
    'NVDA':'Nvidia',
    'PG':'Procter & Gamble',
    'JPM':'J&P Morgan',
    'WMT':'Wallmart',
    'CVX':'Chevron Corporation',
    'BAC':'Bank of America',
    'PFE':'Pfizer',
    'GOOG':'Alphabet',
    'ADBE':'Adobe',
    'AXP':'American Express',
    'BBY':'Best Buy',
    'BA':'Bpeing',
    'CSCO': 'Cisco',
    'C': 'Citigroup',
    'DIS': 'Disney',
    'EBAY': 'eBay',
    'ETSY': 'Etsy',
    'GE': 'General Electric',
    'INTC': 'Intel',
    'JPM': 'JP Morgan Chase',
}

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



########### TITLE #############

st.title("HEC Paris - Finance Labs🧪")
st.subheader("Portfolio theory 📈")
st.markdown("Course provided by: **François Derrien**, **Irina Zviadadze**")

st.markdown("  ")

# with open("Lecture_notes_2021.pdf", "rb") as file:
#     st.download_button("Download Course PDF 💡", file.read(), file_name="Lecture_notes_2021.pdf")

st.markdown("---")

default_text = """Write the answer in this box"""




#####################################################################################
#                   EXERCICE 1 - One risky asset, one risk-free asset
#####################################################################################


if lab_numbers == "01 - One risky and one risk-free asset": # premiere page

    #################################### SIDEBAR ##################################

    risky_asset = st.sidebar.selectbox("Select a risky asset", list_risky_assets, key="select_risky")
    risk_free_asset = "^FVX"

    st.sidebar.markdown("  ")
    
    if st.sidebar.button('**Submit answers**'):
        st.sidebar.write('Your answers have been submitted !')




    ################################### DATAFRAMES ###############################

    # Risky asset 
    data_risky = yf.Ticker(risky_asset)
    df_risky = data_risky.history(period="16mo").reset_index()[["Date","Close"]]
    df_risky.columns = ["Date",risky_asset]
    df_risky = df_risky.loc[(df_risky["Date"]<="2023-07-26") & (df_risky["Date"]>"2022-03-08")] # filter dates 
    df_risky["Date"] = df_risky["Date"].dt.date 


    # Risk-free asset
    data_Tbond = yf.Ticker(risk_free_asset)
    df_Tbond = data_Tbond.history(period="16mo").reset_index()[["Date","Close"]]
    df_Tbond.columns = ["Date","T-bond"]
    df_Tbond = df_Tbond.loc[(df_Tbond["Date"]<="2023-07-26") & (df_Tbond["Date"]>"2022-03-08")] # filter dates 
    df_Tbond["Date"] = df_Tbond["Date"].dt.date




    ##################################### TITLE ####################################
    st.markdown("## 01 - One risky and one risk-free asset")
    st.info("In this exercise, assume that there exists a risk-free asset (a T-bond) with an annual rate of return of 2%. You are given information on daily prices and dividends of individual (risky) stocks. You are asked to choose one risky stock and to compute its expected return and standard deviation of return. Then you have to find the (standard deviation of return, expected return) pairs you can obtain by combining this risky stock with the risk-free asset into portfolios.")
    st.markdown("    ")





    #################################### QUESTION 1 ###################################

    st.subheader("Question 1")

    #################### Part 1

    ## Title of PART 1
    st.markdown('''<p style="font-size: 22px;"> 1. <b> Please select one stock and calculate its realized (holding period) returns. 
                </b> Assume that holding, is one day.</p>''',
                unsafe_allow_html=True)

    st.markdown("   ")

    # ## View risky dataset
    # st.markdown("View the dataset")
    # st.dataframe(df_risky.reset_index(drop=True).head(5))


    ## Download dataset as csv/xlsx

    # CSV
    csv_df = convert_df(df_risky)

    st.download_button(
    label=f"📥 Download **{risky_asset}** stocks as csv",
    data=csv_df,
    file_name=f'{risky_asset}.csv',
    mime='text/csv',
    )

    # # XLSX
    # import io
    #         # Set the headers to force the browser to download the file
    # headers = {
    #             'Content-Disposition': 'attachment; filename=dataset.xlsx',
    #             'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    #         }

    #         # Create a Pandas Excel writer object
    # excel_writer = pd.ExcelWriter(f"{risky_asset}.xlsx", engine='xlsxwriter')
    # df_risky.to_excel(excel_writer, index=False, sheet_name='Sheet1')
    # excel_writer.close()

    #         # Download the file
    # with open(f"{risky_asset}.xlsx", "rb") as f:
    #         st.download_button(
    #                 label=f"📥 Download **{risky_asset}** stocks as xlsx",
    #                 data=f,
    #                 file_name=f"{risky_asset}.xlsx",
    #                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    #             )



    #df_stock["Date"] = pd.to_datetime(df_stock["Date"])
    # chart = alt.Chart(df_risky, title="Evolution of stock prices").mark_line().encode(x="Date",y=risky_asset,
    #         color="symbol",
    #         # strokeDash="symbol",
    #     )

    st.markdown("   ")
    # st.write("**Variation of the stock 📈**")
 
    #st.header("Select Symbol to Forecast")
    #symbol_forecast = st.selectbox("", symbols)
    
    # df_data_2 = df_stock[df_stock["symbol"] == risky_asset].reset_index(drop=True)
    # df_inter_2 = df_data_2.copy()

    # df_inter_2 = df_risky.copy()
    # df_inter_3 = df_inter_2[['Date', kpi]]
    # df_inter_3.columns = ['Date', kpi]
    # df_inter_3 = df_inter_3.rename(columns={'Date': 'ds', kpi: 'y'})
    # df_inter_3['ds'] = to_datetime(df_inter_3['ds'])

    # df_final = df_inter_3.copy()
    # df_final['ds'] = pd.to_datetime(df_final['ds'],infer_datetime_format=True)
    # df_final = df_final.set_index(['ds'])


    # df_final2 = df_final.asfreq(pd.infer_freq(df_final.index))

    # start_date = datetime(2018,1,2)
    # end_date = today = date.today()
    # df_final3 = df_final2[start_date:end_date]

    # df_final4 = df_final3.interpolate(limit=2, limit_direction="forward")
    # df_final5 = df_final4.interpolate(limit=2, limit_direction="backward")


    # start_date = datetime(2018,1,2)
    # end_date = today = date.today()



    ###### Step 2: Calculate the asset's returns
    
    ## Compute past returns, expected return and standard deviation
    asset_returns = df_risky[risky_asset].pct_change().dropna().to_numpy()
    asset_expected_return = np.sum(asset_returns)
    asset_std_dev = np.std(asset_returns)


    st.write("Compute the **expected return** of the asset 📝")
    
    answer = st.text_input("Enter your results",0, key="AQ1.1")

    st.markdown("  ")

    solution = st.checkbox('**Solution** ✅',key="SQ1.1")
    if solution:
        answer_text = f'The expected return of the {dictionary_symbols[risky_asset]} stock is {np.round(asset_expected_return,4)}.'
        st.success(answer_text)
        
    st.markdown("  ")
    st.markdown("  ")



    ##### PART 2 
    st.markdown('<p style="font-size: 22px;"> <b> 2. Next, <b>please calculate the standard deviation of holding-period returns</b></p>',
                unsafe_allow_html=True)


    st.write("Compute the **standard deviation** of the asset 📝")
    
    st.text_input("Enter your results ",0, key="AUQ1.2")

    st.markdown("   ")

    solution = st.checkbox('**Solution** ✅',key="SQ1.2")

    if solution:
        answer_text = f'The standard deviation of the {dictionary_symbols[risky_asset]} stock is {np.round(asset_std_dev,4)}.'
        st.success(answer_text)


    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ") 





    ###################################### QUESTION 2 ##########################################
       

    st.subheader("Question 2")
    
    ### Part 1
    st.markdown('<p style="font-size: 22px;"> 1. Assume that you have a capital of 1000 EUR that you fully invest in a portfolio. <b>Combine two assets (one risky and one risk-free asset) into portfolios</b>, assuming that short-sale constraints are in place (that is, the weight of each asset in your portfolio must be between 0 and 1).</p>',
                unsafe_allow_html=True)

    st.markdown("   ")



    # Concatenate graphs for plot
    df_master = df_risky.merge(df_Tbond, how="inner", on="Date").melt(id_vars="Date", value_vars=[risky_asset,"T-bond"])
    df_master.columns = ["Date","Stock","Price"]
    
    chart = alt.Chart(df_master, title="Evolution of stock prices").mark_line().encode(x="Date",y="Price",
                color="Stock")
    
    st.altair_chart((chart).interactive(), use_container_width=True)
    
    
    
    # df_master = pd.concat([df_risky,df_Tbond]).reset_index(drop=True)
    # df_master = df_master[pd.to_datetime(df_master['Date']) > pd.to_datetime(start_date)]
    # df_master["Date"] = pd.to_datetime(df_master["Date"])


    # chart = alt.Chart(df_master, title="Evolution of stock prices").mark_line().encode(x="Date",y=kpi,
    #             color="symbol",
    #             # strokeDash="symbol",
    #         )
    
    # st.altair_chart((chart).interactive(), use_container_width=True)
    # csv_df = convert_df(df_master)

  

    import io


    st.markdown("  ")

    st.write("**Define the amount you want to put in the risky asset and in the risk-free asset ⚖️**")
    
    stock1_input = st.slider(f"Select the amount for the risky asset", min_value=0, max_value=1000, step=50, value=500)
    st.write("You've invested",stock1_input,"EUR in the",risky_asset,"stock")

    st.markdown("  ")


    stock2_input = st.slider(f"Select the amount in the risk-free asset", min_value=0, max_value=1000, step=50, value=500)
    st.write("You've invested",stock2_input,"EUR in T-bonds")

    st.markdown("  ")

    if stock1_input + stock2_input != 1000:
        st.warning("**Warning**: The sum of both investments should be equal to 1000 EUR")

    st.markdown("  ")
    st.markdown("  ")



    st.write("**Compute the weight of each asset in your portfolio 📝**")

    weight1_input = st.number_input(f'Enter the weight (in %) of {risky_asset}')

    st.markdown("  ")

    solution = st.checkbox('**Solution** ✅',key="SQ2.1w1")
    if solution:
        answer_text1 = f'The weight of the {risky_asset} stock is {np.round(100*stock1_input/1000,1)}%.'
        st.success(answer_text1)


    st.markdown("  ")
    
    weight2_input = st.number_input(f'Enter the weight (in %) of the risk-free asset')

    st.markdown("  ")

    solution = st.checkbox('**Solution** ✅',key="SQ2.1w2")
    if solution:
        answer_text = f'The weight of the risk free asset is {np.round(100*stock2_input/1000,1)}.'
        st.success(answer_text)

    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")



    ######### Part 2
    st.markdown('<p style="font-size: 22px;"> 2. <b>Compute the expected returns and standard deviation of the portfolio.</b> </p>',
                unsafe_allow_html=True)
    

    # Compute returns 
    asset1_returns_ex1 = df_risky[risky_asset].pct_change().dropna().to_numpy() # returns of risky asset
    asset2_returns_ex1 = df_Tbond['T-bond'].pct_change().dropna().to_numpy() # returns of risk free asset
    # st.write(asset1_returns_ex1.shape)
    # st.write(asset2_returns_ex1.shape)

    weight_asset1 = np.round(stock1_input/1000,1)
    weight_asset2 = np.round(stock2_input/1000,1)






    ####### Result: Portfolio expected return 
    st.write("Compute the **expected returns** of the portfolio 📝")

    st.text_input("Enter your results",0, key="AQ2.21")

    st.markdown("  ")

    # Compute expected returns
    portfolio_returns = (weight_asset1*asset1_returns_ex1) + (weight_asset2*asset2_returns_ex1)
    portfolio_expected_returns = np.sum(portfolio_returns)

    # Show solution
    solution = st.checkbox('**Solution** ✅',key="SQ2.21")
    if solution:
        answer_text = f"The portfolio's expected return is {np.round(portfolio_expected_returns,4)}"
        st.success(answer_text)


    st.markdown("    ")
    st.markdown("    ")



    # Portfolio standard deviation
    st.write("Compute the **standard deviation** of the portfolio 📝")

    # Compute standard deviation of the portfolio returns
    portfolio_std = np.std(portfolio_returns)

    st.text_input("Enter your results",0, key="AQ2.22")

    st.markdown("  ")

    solution = st.checkbox('**Solution** ✅',key="SQ2.22")
    if solution:
        answer_text = f"The portfolio's standard deviation is {np.round(portfolio_std,4)}"
        st.success(answer_text)


    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ") 




################## QUESTION 3

    st.subheader("Question 3")
    
    #### PART 1
    st.markdown('''<p style="font-size: 22px;"> Using Excel, <b> construct portfolios </b> that contain x% of the risky asset and (1-x)% of the risk-free asset, with x varying between 0 and 100% with 1% increments.
                For each portfolio, calculate its <b>standard deviation</b> of return and its <b>expected return</b> 
                Represent these combinations in a graph, that is draw the set of <b>feasible portfolios</b>.''',
                unsafe_allow_html=True)
    
    # st.markdown("   ")
    
    # #### PART 2
    # st.markdown('''<p style="font-size: 22px;"> 2.  For each portfolio, calculate its <b>standard deviation of return</b> and its <b>expected return</b> <br> 
    #             Represent these combinations in a <b>graph</b>, that is draw the set of feasible portfolios.''',
    #             unsafe_allow_html=True)
    
    # Weights & realized returns
    weight_portfolios = np.arange(0,1.1,0.01)
    returns_portfolios = np.array([w*asset1_returns_ex1 + (1-w)*asset2_returns_ex1 for w in weight_portfolios])

    weight_portfolios_perct1 = [str(np.round(100*weight))+"%" for weight in weight_portfolios] # add percentage
    weight_portfolios_perct2 = [str(np.round(100*(1-weight)))+"%" for weight in weight_portfolios] # add percentage


    # Compute expected return of each portfolio 
    expected_returns_portfolios = np.sum(returns_portfolios,axis=1)    
    df_exp_return_portfolios = pd.DataFrame({f"Risky asset ({risky_asset})":weight_portfolios_perct1,
                                             "T-bond":weight_portfolios_perct2,
                                             "Expected return":expected_returns_portfolios})
    

    # Compute standard deviation of each portfolio
    std_portfolios = np.std(returns_portfolios,axis=1)
    df_std_dev_portfolios = pd.DataFrame({f"Risky asset ({risky_asset})":weight_portfolios_perct1,
                                            "T-bond":weight_portfolios_perct2,
                                            "Standard deviation":std_portfolios})
    
    
    # Plot
    df_plot_portfolios = pd.DataFrame({"Expected return":df_exp_return_portfolios["Expected return"].to_numpy(),
                                       "Standard deviation":df_std_dev_portfolios["Standard deviation"].to_numpy()})

    chart_portfolios = alt.Chart(df_plot_portfolios, title="Set of feasible portfolios").mark_circle(size=40).encode(y="Expected return",x="Standard deviation")

    
    
    st.markdown("   ")

    st.write("Compute the **expected return** for each portfolio 📝")


    upload_expected_return = st.file_uploader("Drop your results in an excel file (csv or xlsx)", key="Q3.21",type=['csv','xlsx'])
    if upload_expected_return is not None:
        expected_return_portfolios = pd.read_csv(upload_expected_return)
        st.write(expected_return_portfolios.head())

    st.markdown("   ")


    solution = st.checkbox('**Solution** ✅',key="SQ3.1")
    if solution:
        st.dataframe(df_exp_return_portfolios.head())
        csv_df = convert_df(df_exp_return_portfolios)

        st.download_button(
            label=f"📥 Download solutions as csv",
            data=csv_df,
            file_name=f'expected_returns_q3.csv',
            mime='text/csv',
        )


    st.markdown("   ")
    st.markdown("   ")
    



    st.write("Compute the **standard deviation** for each portfolio 📝")

    upload_standard_deviation = st.file_uploader("Drop your results in an excel file (csv or xlsx)", 
                                                 key="SQ3.2", 
                                                 type=['csv','xlsx'])
    
    if upload_standard_deviation is not None:
        standard_deviation_portfolios = pd.read_csv(upload_standard_deviation)
        st.write(standard_deviation_portfolios.head())

    st.markdown("   ")

    solution = st.checkbox('**Solution** ✅',key="SQ3.22")
    if solution:
        st.dataframe(df_std_dev_portfolios.head())
        csv_df = convert_df(df_std_dev_portfolios)

        st.download_button(
            label=f"📥 Download solutions as csv",
            data=csv_df,
            file_name=f'std_dev.csv',
            mime='text/csv',
        )

    st.markdown("   ")
    st.markdown("   ")
    

    
    st.write("**Draw the set of feasible portfolios 📝**")

    upload_graph = st.file_uploader("Drop your graph as an image (jpg, jpeg, png)", key="Q3.23", type=['jpg','jpeg','png'])
    if upload_graph is not None:

        image = Image.open(upload_graph)
        st.image(image, caption='Graph of the set of feasible portfolios')
        

    st.markdown("   ")

    solution = st.checkbox('**Solution** ✅',key="SQ3.23")
    if solution:
        st.altair_chart(chart_portfolios.interactive(), use_container_width=True)

    

    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ") 




################## QUESTION 4

    st.subheader("Question 4")

    st.markdown('''<p style="font-size: 22px;"> Consider the feasible portfolios from Question 3 and <b> answer the following questions. </b> </p>''',
                unsafe_allow_html=True)
    
    st.info("Provide specific answers, that is, **characterize the portfolios in terms of the weights on both assets**")

    
    # Compute expected return and std dev for portfolios with 100% risky (0% risk-free) or 0% risky (100% risk-free)
    expected_return_risky = np.sum(asset1_returns_ex1)
    expected_return_riskfree = np.sum(asset2_returns_ex1)

    std_dev_risky = np.std(asset1_returns_ex1)
    std_dev_riskfree = np.std(asset2_returns_ex1)


    st.markdown("   ")

    ###### PART 1   
    user_input_1 = st.text_area("**Can you find which portfolio has the highest expected return?**", default_text)
    
    solution = st.checkbox('**Solution** ✅',key="SQ4.1")
    if solution:
        st.success(f"The portfolio with 100% in the risky asset ({risky_asset}) and 0% in the risk free asset.")
        st.success(f"The portfolio's expected return is {np.round(np.sum(expected_return_risky),3)}")

    st.markdown("   ")


    ###### PART 2
    user_input_2 = st.text_area("**Can you find which portfolio has the lowest expected return?**", default_text)
    
    solution = st.checkbox('**Solution** ✅',key="SQ4.2")
    if solution:
        st.success(f"The portfolio with 0% in the risky asset ({risky_asset}) and 100% in the risk free asset")
        st.success(f"The portfolio's expected return is {np.round(np.sum(expected_return_riskfree),3)}")

    st.markdown("   ")


    ###### PART 3
    user_input_3 = st.text_area("**Can you find which portfolio has the highest standard deviation?**", default_text)
    solution = st.checkbox('**Solution** ✅',key="SQ4.3")
    if solution:
        st.success(f"The portfolio with 100% in the risky asset ({risky_asset}) and 0% in the risk free asset")
        st.success(f"The portfolio's standard deviation is {np.round(std_dev_risky,3)}")


    st.markdown("   ")
    
    
    ###### PART 4
    user_input_4 = st.text_area("**Can you find which portfolio has the lowest standard deviation?**", default_text)
    solution = st.checkbox('**Solution** ✅',key="SQ4.4")
    if solution:
        st.success(f"The portfolio with 0% in the risky asset ({risky_asset}) and 100% in the risk free asset.")
        st.success(f"The portfolio's standard deviation is {np.round(std_dev_riskfree,3)}")




    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ")




    ##################################### QUESTION 5 #####################################


    st.subheader("Question 5")
    
    st.markdown('''<p style="font-size: 22px;"> Repeat the exercise of Q3, but with the possibility of selling short one of the two assets. That is, vary x, for example, from -100% to 100%.''',
                unsafe_allow_html=True)
    

    # Compute expected return for each portfolio
    weight_portfolios = np.arange(-1,1.1,0.01)
    returns_portfolios = np.array([w*asset1_returns_ex1 + (1-w)*asset2_returns_ex1 for w in weight_portfolios])
    expected_returns_portfolios = np.sum(returns_portfolios,axis=1)
    std_portfolios = np.std(returns_portfolios,axis=1)

    weight_portfolios_perct1 = [str(np.round(100*weight))+"%" for weight in weight_portfolios]
    weight_portfolios_perct2 = [str(np.round(100*(1-weight)))+"%" for weight in weight_portfolios]

    df_exp_return_portfolios = pd.DataFrame({f"Risky asset ({risky_asset})":weight_portfolios_perct1,
                                             "T-bond":weight_portfolios_perct2,
                                             "Expected return":expected_returns_portfolios})
    
    df_std_dev_portfolios = pd.DataFrame({f"Risky asset ({risky_asset})":weight_portfolios_perct1,
                                            "T-bond":weight_portfolios_perct2,
                                            "Standard deviation":std_portfolios})
    
    
    df_plot_portfolios = pd.DataFrame({"Expected return":df_exp_return_portfolios["Expected return"].to_numpy(),
                                       "Standard deviation":df_std_dev_portfolios["Standard deviation"].to_numpy()})

    
    chart_portfolios = alt.Chart(df_plot_portfolios, title="Set of feasible portfolios").mark_line().encode(x="Expected return",y="Standard deviation")

    
    
    st.markdown("   ")

    st.write("Compute the **expected return** for each portfolio 📝")

    upload_expected_return = st.file_uploader("Drop your results in an excel file (csv or xlsx)", key="UQ5.1",type=['csv','xlsx'])
    
    if upload_expected_return is not None:
        expected_return_portfolios = pd.read_csv(upload_expected_return)
        st.write(expected_return_portfolios.head())

    st.markdown("  ")

    solution = st.checkbox('**Solution** ✅',key="SQ5.21")
    if solution:
        st.dataframe(df_std_dev_portfolios.head())
        csv_df = convert_df(df_exp_return_portfolios)

        st.download_button(
            label=f"📥 Download solutions as csv",
            data=csv_df,
            file_name=f'expected_return_q5.csv',
            mime='text/csv',
        )

    st.markdown("   ")
    st.markdown("   ")


    
    st.write("Compute the **standard deviation** for each portfolio 📝")

    upload_standard_deviation = st.file_uploader("Drop your results in an excel file (csv or xlsx)", 
                                                 key="AQ5.1", 
                                                 type=['csv','xlsx'])
    
    if upload_standard_deviation is not None:
        standard_deviation_portfolios = pd.read_csv(upload_standard_deviation)
        st.write(standard_deviation_portfolios.head())

    st.markdown("   ")

    solution = st.checkbox('**Solution** ✅',key="SQ5.22")
    if solution:
        st.dataframe(df_std_dev_portfolios.head())
        csv_df = convert_df(df_std_dev_portfolios)

        st.download_button(
            label=f"📥 Download solutions as csv",
            data=csv_df,
            file_name=f'std_dev_q5.csv',
            mime='text/csv',
        )
    

    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ")
    



################## QUESTION 6

    st.subheader("Question 6")
    
    st.markdown('''<p style="font-size: 22px;"> Repeat the exercise of Q4, but with the possibility of selling short one of the two assets. That is, analyze feasible portfolios from Q5.''',
                unsafe_allow_html=True)
    
    st.markdown("  ")
    
    ###### PART 1   
    user_input_1 = st.text_area("**Can you find which portfolio has the highest expected return?**", default_text, key="Q6.1")
    
    solution = st.checkbox('**Solution** ✅',key="SQ6.1")
    if solution:
        st.success(f"The portfolio with 100% in the risky asset ({risky_asset}) and 0% in the risk free asset.")
        st.success(f"The portfolio's expected return is {np.round(np.sum(expected_return_risky),3)}")

    st.markdown("   ")


    ###### PART 2
    user_input_2 = st.text_area("**Can you find which portfolio has the lowest expected return?**", default_text, key="Q6.2")
    
    solution = st.checkbox('**Solution** ✅',key="SQ6.2")
    if solution:
        st.success(f"The portfolio with 0% in the risky asset ({risky_asset}) and 100% in the risk free asset")
        st.success(f"The portfolio's expected return is {np.round(np.sum(expected_return_riskfree),3)}")

    st.markdown("   ")


    ###### PART 3
    user_input_3 = st.text_area("**Can you find which portfolio has the highest standard deviation?**", default_text, key="Q6.3")
    solution = st.checkbox('**Solution** ✅',key="SQ6.3")
    if solution:
        st.success(f"The portfolio with 100% in the risky asset ({risky_asset}) and 0% in the risk free asset")
        st.success(f"The portfolio's standard deviation is {np.round(std_dev_risky,3)}")


    st.markdown("   ")
    
    
    ###### PART 4
    user_input_4 = st.text_area("**Can you find which portfolio has the lowest standard deviation?**", default_text, key="Q6.4")
    solution = st.checkbox('**Solution** ✅',key="SQ6.4")
    if solution:
        st.success(f"The portfolio with 0% in the risky asset ({risky_asset}) and 100% in the risk free asset.")
        st.success(f"The portfolio's standard deviation is {np.round(std_dev_riskfree,3)}")



    st.markdown(" ")
    st.markdown(" ")
    st.markdown("#### Congratulations you finished Exercise 1 🎉")












#################################################################################################################
#                                        EXERCICE 2 - Two risky assets
#################################################################################################################

if lab_numbers == "02 - Two risky assets":

    ##################################### SIDEBAR ##########################################
    risky_asset1_ex2, risky_asset2_ex2  = st.sidebar.multiselect("Select two risky stocks", list_risky_assets, ["AAPL","NVDA"])

    st.sidebar.markdown("  ")
    
    if st.sidebar.button('**Submit answers**'):
        st.sidebar.write('Your answers have been submitted !')
    
    
    
    ##################################### TITLE ##########################################
    st.markdown("## 02 - Two risky assets")
    st.info("The purpose of this exercise is to understand how to construct efficient portfolios if you can invest in two risky assets or in two risky and one risk-free asset")


    st.markdown("   ")
    st.markdown("   ")
    
    ##################################### QUESTION 1 #####################################
    
    st.subheader("Question 1")
    
    ########### Q1 PART 1
    st.markdown('''<p style="font-size: 22px;"> 1. Download prices for two risky stocks. Compute their realized returns.''',
                unsafe_allow_html=True)

    st.markdown("  ")
    
    # Create dataframe with both risky assets
    data_asset1_ex2 = yf.Ticker(risky_asset1_ex2)
    df_asset1_ex2 = data_asset1_ex2.history(period="16mo")

    data_asset2_ex2 = yf.Ticker(risky_asset2_ex2)
    df_asset2_ex2 = data_asset2_ex2.history(period="16mo")
    
    date_ex2 = pd.to_datetime(df_asset2_ex2.index).date.astype(str)

    df_ex2_1 = pd.DataFrame({'Date':date_ex2, str(risky_asset1_ex2):df_asset1_ex2["Close"].to_list()})
    df_ex2_2 = pd.DataFrame({'Date':date_ex2, str(risky_asset2_ex2):df_asset2_ex2["Close"].to_list()})

    df_ex2 = df_ex2_1.merge(df_ex2_2, how="inner", on="Date")

    # Create stock price evolution graph
    df_ex2_plot = df_ex2.melt(id_vars="Date",value_vars=[str(risky_asset1_ex2),str(risky_asset2_ex2)])
    df_ex2_plot.columns = ["date","asset","value"]

    chart = alt.Chart(df_ex2_plot, title="View the evolution of stock prices").mark_line().encode(x="date",y="value",
                color="asset"
            )
    
    st.altair_chart(chart.interactive(), use_container_width=True)

    

    # Download stock data 
    csv_df = convert_df(df_ex2_1)

    st.download_button(
    label=f"📥 Download **{risky_asset1_ex2}** stocks as csv",
    data=csv_df,
    file_name=f'{risky_asset1_ex2}.csv',
    mime='text/csv',
    )


    csv_df = convert_df(df_ex2_2)

    st.download_button(
    label=f"📥 Download **{risky_asset2_ex2}** stocks as csv",
    data=csv_df,
    file_name=f'{risky_asset2_ex2}.csv',
    mime='text/csv',
    )

    st.markdown("   ")
    st.markdown("  ")



    # Compute the realized returns for both stocks 
    realized_returns1 = np.round(df_ex2[str(risky_asset1_ex2)].pct_change().dropna().to_list(),3)
    realized_returns2 = np.round(df_ex2[str(risky_asset2_ex2)].pct_change().dropna().to_list(),3)



    st.write(f"Compute the **realized returns** of the first asset ({risky_asset1_ex2}) 📝")
    st.text_input("Enter your results",0, key="Ex2.Q1.11")

    st.markdown("  ")

    solution = st.checkbox('**Solution** ✅',key="SQ2.11")
    if solution:
        answer_text = f"The realized returns of {risky_asset1_ex2} are {realized_returns1[:5]} ..."
        st.success(answer_text)
        csv_df = convert_df(pd.DataFrame({risky_asset1_ex2:realized_returns1}))

        st.download_button(
            label=f"📥 Download a csv with the results",
            data=csv_df,
            file_name=f'returns_asset1.csv',
            mime='text/csv',
        )

    
    st.markdown("  ")
    st.markdown("  ")

    
    
    st.write(f"Compute the **realized returns** of the second asset ({risky_asset2_ex2}) 📝")
    st.text_input("Enter your results",0, key="Ex2.Q1.12")

    st.markdown("  ")

    solution = st.checkbox('**Solution** ✅',key="SQ2.12")
    if solution:
        answer_text = f"The realized returns of {risky_asset2_ex2} is {realized_returns2[:5]} ..."
        st.success(answer_text)
        csv_df = convert_df(pd.DataFrame({risky_asset2_ex2:realized_returns2}))

        st.download_button(
            label=f"📥 Download a csv with the results",
            data=csv_df,
            file_name=f'returns_asset2.csv',
            mime='text/csv',
        )


    st.markdown("  ")
    st.markdown("  ")




    
    #### Q1 PART 2
    st.markdown('''<p style="font-size: 22px;"> 2. Estimate the <b>expected returns</b> and <b>standard deviations</b> of returns on these two stocks. 
                Compute the <b>correlation</b> of the returns on these two stocks.''',
                unsafe_allow_html=True)
    
    st.markdown("  ")
    
    # Expected returns for both assets
    expected_returns1 = np.sum(realized_returns1)
    expected_returns2 = np.sum(realized_returns2)

    # Standard deviation for both assets
    std_returns1 = np.std(realized_returns1)
    std_returns2 = np.std(realized_returns2) 

    # Correlation between both assets
    corr_risky_assets = pd.DataFrame({"asset1":realized_returns1, "asset2":realized_returns2}).corr().iloc[0,1]    


    st.write(f"Compute the **expected returns** of the first asset ({risky_asset1_ex2}) 📝")
    st.text_input("Enter your results",0, key="Ex2.Q1.21")

    solution = st.checkbox('**Solution** ✅',key="SQ2.21")
    if solution:
        st.success(f"The expected return of {risky_asset1_ex2} is {expected_returns1} ...")

    st.markdown("  ")
    st.markdown("  ")
    
    st.write(f"Compute the **expected returns** of the second asset ({risky_asset2_ex2}) 📝")
    st.text_input("Enter your results",0, key="Ex2.Q1.22")


    solution = st.checkbox('**Solution** ✅',key="SQ2.22")
    if solution:
        st.success(f"The expected return of {risky_asset2_ex2} is {expected_returns2} ...")

    st.markdown("  ")
    st.markdown("  ")


    st.write(f"Compute the **standard deviation** of the first asset ({risky_asset1_ex2}) 📝")
    st.text_input(f"Enter your results",0, key="Ex2.Q2.23")
    
    
    solution = st.checkbox('**Solution** ✅',key="SQ2.23")
    if solution:
        st.success(f"The standard deviation of {risky_asset1_ex2} is {std_returns1} ...")

    st.markdown("  ")
    st.markdown("  ")


    st.write(f"Compute the **standard deviation** of the second asset ({risky_asset2_ex2}) 📝")
    st.text_input(f"Enter your results",0, key="Ex2.Q2.24")

    st.markdown("  ")

    solution = st.checkbox('**Solution** ✅',key="SQ2.24")
    if solution:
        st.success(f"The standard deviation of {risky_asset2_ex2} is {std_returns2} ...")

    st.markdown("  ")
    st.markdown("  ")

    # Compute the correlation of returns
    st.write(f"Compute the **correlation** between both assets 📝")
    st.text_input(f"Enter your results",0, key="Ex2.Q2.25")

    st.markdown("  ")

    solution = st.checkbox('**Solution** ✅',key="SQ2.25")
    if solution:
        st.success(f"The correlation of both assets is {np.round(corr_risky_assets,3)}")

    
    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ")



    ##################################### QUESTION 2 #####################################

    st.subheader("Question 2")

    #### Q2 PART 1
    st.markdown('''<p style="font-size: 22px;"> Compose different <b>portfolios of two risky assets</b> by investing in one risky asset x% of your wealth and in the other asset (1-x)%.
                Vary x from -50% to 150% with an increment of 5%. Compute the <b>expected returns</b> and <b>standard deviations</b> of the resulting portfolios.''', 
                unsafe_allow_html=True)
    
    st.info("**Hint**: Do not forget about the correlation between the returns on these two stocks.")

    
    st.markdown("  ")



    # Weights & realized returns
    weight_portfolios = np.round(np.arange(-0.5,1.55,0.05),2)
    returns_portfolios = np.array([w*realized_returns1 + (1-w)*realized_returns2 for w in weight_portfolios])

    weight_portfolios_perct1 = [str(np.round(100*weight))+"%" for weight in weight_portfolios] # add percentage
    weight_portfolios_perct2 = [str(np.round(100*(1-weight)))+"%" for weight in weight_portfolios] # add percentage


    # Compute expected return of each portfolio 
    expected_returns_portfolios = np.mean(returns_portfolios,axis=1)
    
    df_exp_return_portfolios = pd.DataFrame({risky_asset1_ex2:weight_portfolios_perct1,
                                             risky_asset2_ex2:weight_portfolios_perct2,
                                             "Expected return":expected_returns_portfolios})
    

    # Compute standard deviation of each portfolio
    cov_assets = corr_risky_assets*realized_returns1*realized_returns2
    std_portfolios = np.array([(w*np.std(realized_returns1))**2 + ((1-w)*np.std(realized_returns2)**2) + 2*w*(1-w)*cov_assets for w in weight_portfolios])
    std_portfolios = np.sqrt(np.sum(std_portfolios,axis=1))
    
    df_std_dev_portfolios = pd.DataFrame({risky_asset1_ex2:weight_portfolios_perct1,
                                          risky_asset1_ex2:weight_portfolios_perct2,
                                          "Standard deviation":std_portfolios})
    

    st.markdown("   ")

    st.write("Compute the **expected return** for each portfolio 📝")


    upload_expected_return = st.file_uploader("Drop your results in an excel file (csv or xlsx)", key="Q3.21",type=['csv','xlsx'])
    if upload_expected_return is not None:
        expected_return_portfolios = pd.read_csv(upload_expected_return)
        st.write(expected_return_portfolios.head())

    st.markdown("   ")


    solution = st.checkbox('**Solution** ✅',key="SQ3.1")
    if solution:
        st.dataframe(df_exp_return_portfolios.head())
        csv_df = convert_df(df_exp_return_portfolios)

        st.download_button(
            label=f"📥 Download solutions as csv",
            data=csv_df,
            file_name=f'expected_returns_ex2_q2.csv',
            mime='text/csv',
        )


    st.markdown("   ")
    st.markdown("   ")
    


    st.write("Compute the **standard deviation** for each portfolio 📝")

    upload_standard_deviation = st.file_uploader("Drop your results in an excel file (csv or xlsx)", 
                                                 key="SQ3.2", 
                                                 type=['csv','xlsx'])
    
    if upload_standard_deviation is not None:
        standard_deviation_portfolios = pd.read_csv(upload_standard_deviation)
        st.write(standard_deviation_portfolios.head())

    st.markdown("   ")

    solution = st.checkbox('**Solution** ✅',key="SQ3.22")
    if solution:
        st.dataframe(df_std_dev_portfolios.head())
        csv_df = convert_df(df_std_dev_portfolios)

        st.download_button(
            label=f"📥 Download solutions as csv",
            data=csv_df,
            file_name=f'std_dev_ex2_q2.csv',
            mime='text/csv',
        )

    

    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ") 




    ##################################### QUESTION 3 #####################################

    st.subheader("Question 3")

    st.markdown('''<p style="font-size: 22px;"> 1. Indicate the set of <b>feasible portfolios</b> and the set of <b>efficient portfolios</b>.''', 
                unsafe_allow_html=True)
    
    
    user_input_1 = st.text_area("**What is the set of feasible portfolios ?**", default_text, key="Q3.Ex2.11")

    solution = st.checkbox('**Solution** ✅',key="SQ3.Ex2.11")
    if solution:
        st.success("")

    st.markdown("  ")


    user_input_2 = st.text_area("**What is the set of efficient portfolios ?**", default_text, key="Q3.Ex2.12")

    solution = st.checkbox('**Solution** ✅',key="SQ3.Ex2.12")
    if solution:
        st.success("")

    
    st.markdown("  ")
    st.markdown("  ")
    
    st.markdown('''<p style="font-size: 22px;"> 2. Draw a graph in which you represent the portfolios, that is, the sigma-expected return pairs, you obtain with different combinations of the two risky assets..''', 
                unsafe_allow_html=True)
        
    # code for portfolio plot
    df_plot_portfolios = pd.DataFrame({"Expected return":df_exp_return_portfolios["Expected return"].to_numpy(),
                                       "Standard deviation":df_std_dev_portfolios["Standard deviation"].to_numpy()})

    chart_portfolios = alt.Chart(df_plot_portfolios, title="Set of feasible portfolios").mark_circle().encode(y="Expected return",x="Standard deviation")


    #st.write("**Draw the set of feasible portfolios 📝**")


    upload_graph = st.file_uploader("**Drop your graph as an image (jpg, jpeg, png)**", key="Q3.Ex2.13", type=['jpg','jpeg','png'])
    if upload_graph is not None:

        image = Image.open(upload_graph)
        st.image(image, caption='Graph of feasible portfolios')
        

    st.markdown("   ")

    solution = st.checkbox('**Solution** ✅',key="SQ3.23")
    if solution:
        st.altair_chart(chart_portfolios.interactive(), use_container_width=True)


    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ")



    ##################################### QUESTION 4 #####################################

    st.subheader("Question 4")

    st.markdown('''<p style="font-size: 22px;"> Assume that you cannot short-sell any of the risky assets (only in this exercise). 
                Indicate the new set of feasible portfolios and the net set of efficient portfolios.''', 
                unsafe_allow_html=True)
    
    user_input_1 = st.text_area("**What is the set of feasible portfolios ?**", default_text, key="Q4.Ex2.11")

    solution = st.checkbox('**Solution** ✅',key="SQ4.Ex2.11")
    if solution:
        st.success("")

    st.markdown("  ")
    st.markdown("  ")


    user_input_2 = st.text_area("**What is the set of efficient portfolios ?**", default_text, key="Q4.Ex2.12")

    solution = st.checkbox('**Solution** ✅',key="SQ4.Ex2.12")
    if solution:
        st.success("")
        
    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ")



    ##################################### QUESTION 5 #####################################

    st.subheader("Question 5")

    ### Q5 PART 1
    st.markdown('''<p style="font-size: 22px;">1. Assume that you also have a risk-free asset with a rate of return of 2% per annum. 
                Find the tangency portfolio. ''', 
                unsafe_allow_html=True)
    
    st.info("**Hint**: Compute the Sharpe ratio (the reward-to-variability ratio) for all feasible portfolios in Exercise 2.")
    
    st.markdown("  ")
    
    ### Q5 PART 2
    st.markdown('''<p style="font-size: 22px;">2. Find the portfolio with the maximal Sharpe ratio''', 
                unsafe_allow_html=True)
    
        
    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ")



    ######################################### QUESTION 6 #########################################

    st.subheader("Question 6")

    ### Q5 PART 1
    st.markdown('''<p style="font-size: 22px;">Indicate the set of efficient portfolios that you can achieve if you invest in two risky and one risk-free asset. ''', 
                unsafe_allow_html=True)

    st.markdown(" ")
    st.markdown(" ")
    st.markdown("#### Congratulations you finished Exercise 2 🎉")



if lab_numbers == "03 - Diversification":

    st.info("This page is a work in progress. Please check back later.")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown("#### Congratulations you masterise the Portfolio theory 🎉")


if lab_numbers == "04 - Test of the CAPM":

    st.info("This page is a work in progress. Please check back later.")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown("#### Congratulations you masterise the Portfolio theory 🎉")
























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


if __name__ == "__main__":
    footer2()

