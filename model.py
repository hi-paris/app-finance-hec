import pandas as pd
from pandas import to_datetime
from pandas.plotting import register_matplotlib_converters
import numpy as np
from pathlib import Path
import base64
from datetime import date, datetime
import yfinance as yf



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

st.sidebar.number_input("**🪪 Input your student number:**",67609)

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


########### TITLE #############

st.title("HEC Paris- Finance Labs🧪")
st.subheader("Portfolio theory 📈")
st.markdown("Course provided by: **François DERRIEN** & **Irina Zviadadze**")

st.markdown("  ")

with open("Lecture_notes_2021.pdf", "rb") as file:
    st.download_button("Download Course PDF 💡", file.read(), file_name="Lecture_notes_2021.pdf")

st.markdown("---")




#####################################################################################
#                   EXERCICE 1 - One risky asset, one risk-free asset
#####################################################################################


if lab_numbers == "01 - One risky and one risk-free asset": # premiere page

    # Ex1 sidebar
    risky_asset = st.sidebar.selectbox("Select a risky asset", list_risky_assets, key="select_risky")
    risk_free_asset = st.sidebar.selectbox("Select a risk-free asset", list_riskfree_assets, key="select_riskfree")
    list_kpi = ['High', 'Low','Open','Close','Volume']
    kpi = st.sidebar.selectbox("Select Stock KPI", list_kpi)

    # Title & Description
    st.markdown("## 01 - One risky and one risk-free asset")
    st.info("In this exercise, assume that there exists a risk-free asset (a T-bond) with an annual rate of return of 2%. You are given information on daily prices and dividends of individual (risky) stocks. You are asked to choose one risky stock and to compute its expected return and standard deviation of return. Then you have to find the (standard deviation of return, expected return) pairs you can obtain by combining this risky stock with the risk-free asset into portfolios.")
    st.markdown("    ")


######## QUESTION 1 

    st.subheader("Question 1")

    ### Part 1
    st.markdown('<p style="font-size: 22px;"> 1. <b> Please select one stock and calculate its realized (holding period) returns. </b> Assume that holding is one day.</p>',
                unsafe_allow_html=True)

    st.markdown("   ")

    start_date = st.date_input(
            "Select a start date",
            date(2022, 1, 1),
            min_value=datetime.strptime("2022-01-01", "%Y-%m-%d"),
            max_value=datetime.now(),
        )


    data = yf.Ticker(risky_asset)
    df_stock = data.history(period="16mo")
    df_stock['Date'] = pd.to_datetime(df_stock.index).date.astype(str)
    df_stock["symbol"] = [risky_asset]*len(list(df_stock.index))



    data = yf.Ticker(risky_asset)
    df_stock = data.history(period="16mo")
    df_stock['Date'] = pd.to_datetime(df_stock.index).date.astype(str)
    df_stock["symbol"] = [risky_asset]*len(list(df_stock.index))

    st.markdown("   ")

    head = st.radio('View data from top (head) or bottom (tail)', ('Head', 'Tail'))

    if head == 'Head':
        st.dataframe(df_stock.reset_index(drop=True).head(5))
    else:
        st.dataframe(df_stock.reset_index(drop=True).tail(5))

    @st.cache_data
    def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv_df = convert_df(df_stock)

    st.download_button(
    label=f"📥 Download **{risky_asset}** stocks as csv",
    data=csv_df,
    file_name=f'{risky_asset}.csv',
    mime='text/csv',
    )

    import io
            # Set the headers to force the browser to download the file
    headers = {
                'Content-Disposition': 'attachment; filename=dataset.xlsx',
                'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }

            # Create a Pandas Excel writer object
    excel_writer = pd.ExcelWriter(f"{risky_asset}.xlsx", engine='xlsxwriter')
    df_stock.to_excel(excel_writer, index=False, sheet_name='Sheet1')
    excel_writer.close()

            # Download the file
    with open(f"{risky_asset}.xlsx", "rb") as f:
            st.download_button(
                    label=f"📥 Download **{risky_asset}** stocks as xlsx",
                    data=f,
                    file_name=f"{risky_asset}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


    

    df_stock["Date"] = pd.to_datetime(df_stock["Date"])
    chart = alt.Chart(df_stock, title="Evolution of stock prices").mark_line().encode(x="Date",y=kpi,
            color="symbol",
            # strokeDash="symbol",
        )

    st.markdown("   ")
    # st.write("**Variation of the stock 📈**")
 
    #st.header("Select Symbol to Forecast")
    #symbol_forecast = st.selectbox("", symbols)
    df_data_2 = df_stock[df_stock["symbol"] == risky_asset].reset_index(drop=True)


    df_inter_2 = df_data_2.copy()
    #st.dataframe(df_data_2.head())
    df_inter_3 = df_inter_2[['Date', kpi]]
    df_inter_3.columns = ['Date', kpi]
    df_inter_3 = df_inter_3.rename(columns={'Date': 'ds', kpi: 'y'})
    df_inter_3['ds'] = to_datetime(df_inter_3['ds'])

    df_final = df_inter_3.copy()
    df_final['ds'] = pd.to_datetime(df_final['ds'],infer_datetime_format=True)
    df_final = df_final.set_index(['ds'])


    df_final2 = df_final.asfreq(pd.infer_freq(df_final.index))

    start_date = datetime(2018,1,2)
    end_date = today = date.today()
    df_final3 = df_final2[start_date:end_date]


    df_final4 = df_final3.interpolate(limit=2, limit_direction="forward")
    df_final5 = df_final4.interpolate(limit=2, limit_direction="backward")

    # Step 2: Calculate the asset returns
    asset_returns = df_inter_2['Close'].pct_change().dropna()
    print(df_inter_2)

    asset_std_dev = np.std(asset_returns)
    #st.write(asset_returns)


    plt.figure(figsize=(14,4))
    plt.plot(df_final5)
    plt.title(f'Variation of {dictionary_symbols[risky_asset]} Stock overtime', fontsize=20)
    plt.ylabel('Stock value in ($)', fontsize=16)
    st.pyplot()


    ######## CHECK RESULTS ########

    st.write("**Stock's returns 📝**")
    
    answer = st.text_input("Enter your results",0, key="AQ1.1")

    if answer == np.round(asset_returns,4).all():
        st.success("Congrats, you have the good answer!")
    else:
        st.warning("Try again")

    st.markdown("  ")

    solution = st.checkbox('**Solution** ✅',key="SQ1.1")


    if solution:
        answer_text = f'The returns of the {dictionary_symbols[risky_asset]} stock is {np.round(asset_returns.to_numpy(),4)} on the given period.'
        st.success(answer_text)
        
    st.markdown("  ")
    st.markdown("  ")



    ##### PART 2 
    st.markdown('<p style="font-size: 22px;"> <b> 2. Next, <b>please calculate the standard deviation of holding-period returns</b></p>',
                unsafe_allow_html=True)


    st.write("**Standard Deviation 📝**")
    
    answer = st.text_input("Enter your results ",0, key="AUQ1.2")
    if answer == np.round(asset_std_dev,4) :
        st.success("Congrats, you have the right answer!")
    else:
        st.warning("Try again")

    st.markdown("   ")

    solution = st.checkbox('**Solution** ✅',key="SQ1.2")

    if solution:
        answer_text = f'The standard Deviation of the {dictionary_symbols[risky_asset]} stock returns is {np.round(asset_std_dev,4)} on the given period.'
        st.success(answer_text)


    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ") 


######### QUESTION 2     

    st.subheader("Question 2")
    
    ### Part 1
    st.markdown('<p style="font-size: 22px;"> 1. Assume that you have a capital of 1000 EUR that you fully invest in a portfolio. <b>Combine two assets (one risky and one risk-free asset) into portfolios</b>, assuming that short-sale constraints are in place (that is, the weight of each asset in your portfolio must be between 0 and 1).</p>',
                unsafe_allow_html=True)

    st.markdown("   ")

    list_stocks = [risky_asset, risk_free_asset]
    list_dataframes = []
    for stock in list_stocks:
        data = yf.Ticker(stock)
        df_data = data.history(period="16mo")
        df_data['Date'] = pd.to_datetime(df_data.index).date.astype(str)
        df_data["symbol"] = [stock]*len(list(df_data.index))

        #df_data = pdr.get_data_yahoo(symbols[i])
        list_dataframes.append(df_data)
    
    df_master = pd.concat(list_dataframes).reset_index(drop=True)
    df_master = df_master[pd.to_datetime(df_master['Date']) > pd.to_datetime(start_date)]


    df_master["Date"] = pd.to_datetime(df_master["Date"])
    chart = alt.Chart(df_master, title="Evolution of stock prices").mark_line().encode(x="Date",y=kpi,
                color="symbol",
                # strokeDash="symbol",
            )
    #chart = get_chart(df_master)
    st.altair_chart((chart).interactive(), use_container_width=True)


    csv_df = convert_df(df_master)

    st.download_button(
    label=f"📥 Download **{risky_asset}_{risk_free_asset}** stocks as csv",
    data=csv_df,
    file_name=f'{risky_asset}_{risk_free_asset}.csv',
    mime='text/csv',
    )

    import io
    #if st.button("Download Dataset"):
            # Set the headers to force the browser to download the file
    headers = {
                'Content-Disposition': 'attachment; filename=dataset.xlsx',
                'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }

            # Create a Pandas Excel writer object
    excel_writer = pd.ExcelWriter(f"{risky_asset}_{risk_free_asset}.xlsx", engine='xlsxwriter')
    df_master.to_excel(excel_writer, index=False, sheet_name='Sheet1')
    excel_writer.close()

            # Download the file
    with open(f"{risky_asset}_{risk_free_asset}.xlsx", "rb") as f:
            st.download_button(
                    label=f"📥 Download **{risky_asset}_{risk_free_asset}** stocks as xlsx",
                    data=f,
                    file_name=f"{risky_asset}_{risk_free_asset}.xlsx",)


    st.markdown("  ")

    st.write("**Define the amount you want to put in the risky asset and in the risk-free asset**")
    stock1_input = st.number_input(f"Select the amount you want to invest in your stock",)
    stock2_input = st.number_input(f"Select the amount you want to invest in T-bonds",)

    st.markdown("  ")
    st.markdown("  ")

    st.write("**Define the weight of each asset in your portfolio ⚖️**")
    weight1_input = st.number_input(f"Select the % of your portfolio {risky_asset} will represent",)

    st.write(f"*The % of your portfolio invested in T-bonds is {100-weight1_input}*")
    #weight2_input = st.number_input(f"Select the % of your portfolio {symbols[1]} will represent ",)

    st.markdown("   ")
    st.markdown("   ") 



    ## Part 2
    st.markdown('<p style="font-size: 22px;"> 2. Now assume that you will use historical average return as an estimate for the expected return of your stock. <b>Compute the expected returns and standard deviation of the portfolio.</b> </p>',
                unsafe_allow_html=True)
    

    # Historical returns of assets in the portfolio
    asset1_returns = np.array([0.1, 0.05, 0.02, 0.08, -0.03])
    asset2_returns = np.array([0.05, 0.02, 0.07, -0.01, 0.04])
    weight_asset1 = 0.4
    weight_asset2 = 0.6

    # Calculate the portfolio return
    portfolio_returns = (weight_asset1 * asset1_returns) + (weight_asset2 * asset2_returns)
    portfolio_return = np.sum(portfolio_returns)

    # Calculate the standard deviation of the portfolio returns
    portfolio_std = np.std(portfolio_returns, ddof=1)



    # Portfolio expected return 
    st.write("**Expected return** 📝")

    answer = st.text_input("Enter your results",0, key="AQ2.21")
    if answer == np.round(portfolio_return,4):
        st.success("Congrats, you have the good answer!")
    else:
        st.warning("Try again")

    st.markdown("  ")

    solution = st.checkbox('**Solution** ✅',key="SQ2.21")
    if solution:
        answer_text = f"The portfolio's expected return is {np.round(portfolio_return,4)}"
        st.success(answer_text)


    st.markdown("    ")
    st.markdown("    ")



    # Portfolio standard deviation
    st.write("**Standard deviation 📝**")

    answer = st.text_input("Enter your results",0, key="AQ2.22")
    if answer == np.round(portfolio_std,4):
        st.success("Congrats, you have the good answer!")
    else:
        st.warning("Try again")

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
    st.markdown('''<p style="font-size: 22px;"> 1. Using Excel, <b> construct portfolios that contain x% of the risky asset and (1-x)% of the risk-free asset </b>, with x varying between 0 and 100% with 1% increments.''',
                unsafe_allow_html=True)
    
    # Upload csv file 
    from io import StringIO

    uploaded_file = st.file_uploader("Choose a csv file",key="Q3.1")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
    
    st.markdown("   ")
    st.markdown("   ")


    #### PART 2
    st.markdown('''<p style="font-size: 22px;"> 2.  For each portfolio, calculate its <b>standard deviation of return</b> and its <b>expected return</b> <br> 
                Represent these combinations in a <b>graph</b>, that is draw the set of feasible portfolios.''',
                unsafe_allow_html=True)
    


    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ") 



################## QUESTION 4

    st.subheader("Question 4")

    st.markdown('''<p style="font-size: 22px;"> <b>Consider the feasible portfolios from Question 3.</b> </p>
                <ul>
                    <li> Can you find which portfolio has the highest expected return?  </li>
                    <li> Can you find which portfolio has the lowest expected return? </li>
                    <li> Can you find which portfolio has the lowest standard deviation? </li>
                    <li> Can you find which portfolio has the highest standard deviation? </li>
                </ul>
                Provide specific answers, that is, <b>characterize the portfolios in terms of the weights on both assets</b>. 
                 </p>''',
                unsafe_allow_html=True)


    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ")




################## QUESTION 5

    st.subheader("Question 5")
    
    st.markdown('''<p style="font-size: 22px;"> Repeat the exercise of Q3, but with the possibility of selling short one of the two assets. That is, vary x, for example, from -100% to 100%.''',
                unsafe_allow_html=True)
    

    uploaded_file = st.file_uploader("Choose a csv file",key="Q5")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)
    
    

    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ")
    



################## QUESTION 6

    st.subheader("Question 6")
    
    st.markdown('''<p style="font-size: 22px;"> Repeat the exercise of Q4, but with the possibility of selling short one of the two assets. That is, analyze feasible portfolios from Q5.''',
                unsafe_allow_html=True)



    st.markdown(" ")
    st.markdown(" ")
    st.markdown("#### Congratulations you finished Exercice 1 🎉")



if lab_numbers == "02 - Two risky assets":

    two_risky_assets = st.sidebar.multiselect("Select two risky stocks", list_risky_assets, ["AAPL","NVDA"])

    st.info("This page is a work in progress. Please check back later.")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown("#### Congratulations you masterise the Portfolio theory 🎉")



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

