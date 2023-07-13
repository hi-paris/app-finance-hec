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



st.sidebar.header("Dashboard")
st.sidebar.markdown("---")

st.sidebar.number_input("**🪪 Input your student number:**",67609)

lab_numbers = st.sidebar.selectbox('Select the exercise ➡️', [
  '01 - One risky and one risk-free asset',
  '02 - Two risky assets',
  '03 - Diversification',
  '04 - Test of the CAPM',
  ])

#st.sidebar.header("Select Stock Symbol")
list_symbols = ['AAPL', 'AMZN', 'IBM','MSFT','TSLA','NVDA',
                    'PG','JPM','WMT','CVX','BAC','PFE','GOOG',
                'ADBE','AXP','BBY','BA','CSCO','C','DIS','EBAY','ETSY','GE','INTC','JPM']
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

@st.cache_data
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


st.title("HEC Paris- Finance Labs 🧪")
st.subheader("Portfolio theory 📈")
st.markdown("---")
st.markdown("Course provided by: **François DERRIEN** & **Irina Zviadadze**")
st.markdown("---")

#if st.button("Download PDF"):
with open("Lecture_notes_2021.pdf", "rb") as file:
    st.download_button("Download Course PDF 💡", file.read(), file_name="Lecture_notes_2021.pdf")

# st.markdown(
#     """
#     [<img src='data:image/png;base64,{}' class='img-fluid' width=25 height=25>](https://github.com/gaetanbrison/app-predictive-analytics) <small> app-predictive-analytics 1.0.0 | June 2022</small>""".format(
#         img_to_bytes("./images/github.png")
#     ),
#     unsafe_allow_html=True,
# )


if lab_numbers == "01 - One risky and one risk-free asset":

    st.markdown("## 01 - One risky and one risk-free asset")
    st.success("Investors want to earn the highest return possible for a level of risk that they are willing to take. So how does an investor allocate her capital to maximize her investment utility — the risk-return profile that yields the greatest satisfaction? The simplest way to examine this is to consider a portfolio consisting of 2 assets: a risk-free asset that has a low rate of return but no risk, and a risky asset that has a higher expected return for a higher risk.")



    # st.subheader(" ")
    # st.subheader("Stock Dataset")
    # st.subheader(" ")

    st.markdown("#### Q1 - Select One Stock and Calculate its returns and Standard deviation of Returns: ")
   
    start_date = st.date_input(
            "Select start date",
            date(2022, 1, 1),
            min_value=datetime.strptime("2022-01-01", "%Y-%m-%d"),
            max_value=datetime.now(),
        )
    symbols = st.sidebar.multiselect("Select stocks", list_symbols, ["AAPL","NVDA"])


   
    list_kpi = ['High', 'Low','Open','Close','Volume']
    kpi = st.sidebar.selectbox("Select Stock KPI", list_kpi)

    #symbols = ['AAPL', 'AMZN', 'IBM','MSFT','TSLA','NVDA',
    #                    'PG','JPM','WMT','CVX','BAC','PFE','GOOG',
    #                'ADBE','AXP','BBY','BA','CSCO','C','DIS','EBAY','ETSY','GE','INTC','JPM']
                    
    #kpi = ['High', 'Low','Open','Close','Volume']
    list_dataframes = []
    for i in range(0,len(symbols)):
        data = yf.Ticker(symbols[i])
        df_data = data.history(period="16mo")
        df_data['Date'] = pd.to_datetime(df_data.index).date.astype(str)
        df_data["symbol"] = [symbols[i]]*len(list(df_data.index))

        #df_data = pdr.get_data_yahoo(symbols[i])
        list_dataframes.append(df_data)
    df_master = pd.concat(list_dataframes).reset_index(drop=True)
    df_master = df_master[pd.to_datetime(df_master['Date']) > pd.to_datetime(start_date)]



    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))

    if head == 'Head':
        st.dataframe(df_master.reset_index(drop=True).head(5))
    else:
        st.dataframe(df_master.reset_index(drop=True).tail(5))

    @st.cache_data
    def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv_df = convert_df(df_master)

    st.download_button(
    label=f"📥 Download **{symbols[0]}** stocks as csv",
    data=csv_df,
    file_name=f'{symbols[0]}.csv',
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
    excel_writer = pd.ExcelWriter(f"{symbols[0]}.xlsx", engine='xlsxwriter')
    df_master.to_excel(excel_writer, index=False, sheet_name='Sheet1')
    excel_writer.close()

            # Download the file
    with open(f"{symbols[0]}.xlsx", "rb") as f:
            st.download_button(
                    label=f"📥 Download **{symbols[0]}** stocks as xlsx",
                    data=f,
                    file_name=f"{symbols[0]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )



    df_master["Date"] = pd.to_datetime(df_master["Date"])
    chart = alt.Chart(df_master, title="Evolution of stock prices").mark_line().encode(x="Date",y=kpi,
            color="symbol",
            # strokeDash="symbol",
        )
    #chart = get_chart(df_master)
    #st.altair_chart((chart).interactive(), use_container_width=True)




 
    #st.header("Select Symbol to Forecast")
    symbol_forecast = st.selectbox("", symbols)
 
    #data = yf.Ticker(symbol_forecast)
    #df_data = data.history(period="12mo")
    #    df_data['Date'] = pd.to_datetime(df_data.index).date.astype(str)
    #    df_data["symbol"] = [symbols[i]]*len(list(df_data.index))
    #st.dataframe(df_data.tail())
    df_data_2 = df_master[df_master["symbol"] == symbol_forecast].reset_index(drop=True)
    #df_data_2 = pdr.get_data_yahoo(symbol_forecast)
    #df_inter_2 = pd.DataFrame(
    #         {'symbol': [symbol_forecast]*len(list(df_data_2.index)),
    #         'date': list(df_data_2.index),
    #         kpi: list(df_data_2[kpi])
    #         })

    df_inter_2 = df_data_2.copy()
    #st.dataframe(df_data_2.head())
    df_inter_3 = df_inter_2[['Date', kpi]]
    df_inter_3.columns = ['Date', kpi]
    df_inter_3 = df_inter_3.rename(columns={'Date': 'ds', kpi: 'y'})
    df_inter_3['ds'] = to_datetime(df_inter_3['ds'])
    #st.dataframe(df_inter_3.head())


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
    asset_std_dev = np.std(asset_returns)
    #st.write(asset_returns)

    plt.figure(figsize=(14,4))
    plt.plot(df_final5)
    plt.title(f'Variation of {dictionary_symbols[symbol_forecast]} Stock overtime', fontsize=20)
    plt.ylabel('Stock value in ($)', fontsize=16)
    st.pyplot()

    st.warning(f"What is the return for the {symbols[0]} stock ❓")
    image_return = Image.open('images/return.png')
    st.image(image_return, width=600)
    st.warning(f"What is the Standard deviation of returns for the {symbols[0]} stock ❓")
    image_hec = Image.open('images/standarddeviation.png')
    st.image(image_hec, width=400)

    answer = st.checkbox('Answer 📝')
    if answer:
        user_input_return = st.number_input("Input answer Return ➡️",step=0.01)

        user_input_std = st.number_input("Input answer Standard Deviation ➡️",step=0.01)

   # if user_input.lower() == "portfolio2":
   #     st.success("You have the good answer!")
   # else:
   #     st.warning("Try again")    
        

    hint = st.checkbox('Hint 💡')

    if hint:
        image_hint = Image.open('images/hint.png')
        st.image(image_hint)


    solution = st.checkbox('Solution ✅')

    if solution:
        st.success(f'You have selected {dictionary_symbols[symbol_forecast]} stock.Standard Deviation of the stock returns is of {np.round(asset_returns,4)} on the given period.')
        st.success(f'You have selected {dictionary_symbols[symbol_forecast]} stock.Standard Deviation of the stock standard deviation is of {np.round(asset_std_dev,4)} on the given period.')


    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")     
    st.markdown("   ")
    st.markdown("   ") 
    st.markdown("   ")

    st.markdown("#### Q2 - Combine  two assets (one risky and one risk-free asset) into portfolios assuming that short-sell constraints are in place. Calculate the expected returns and standard deviation of the portfolio.")


    df_master["Date"] = pd.to_datetime(df_master["Date"])
    chart = alt.Chart(df_master, title="Evolution of stock prices").mark_line().encode(x="Date",y=kpi,
                color="symbol",
                # strokeDash="symbol",
            )
    #chart = get_chart(df_master)
    st.altair_chart((chart).interactive(), use_container_width=True)


    csv_df = convert_df(df_master)

    st.download_button(
    label=f"📥 Download **{symbols[0]}_{symbols[1]}** stocks as csv",
    data=csv_df,
    file_name=f'{symbols[0]}_{symbols[1]}.csv',
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
    excel_writer = pd.ExcelWriter(f"{symbols[0]}_{symbols[1]}.xlsx", engine='xlsxwriter')
    df_master.to_excel(excel_writer, index=False, sheet_name='Sheet1')
    excel_writer.close()

            # Download the file
    with open(f"{symbols[0]}_{symbols[1]}.xlsx", "rb") as f:
            st.download_button(
                    label=f"📥 Download **{symbols[0]}_{symbols[1]}** stocks as xlsx",
                    data=f,
                    file_name=f"{symbols[0]}_{symbols[1]}.xlsx",)
                   
    st.write("Define the amount you want to put in each stock in 💵")
    stock1_input = st.number_input(f"Select the amount you want to invest in {symbols[0]} stocks",)
    stock2_input = st.number_input(f"Select the amount you want to invest in {symbols[1]} stocks",)

    st.write("Define the weight (the sum of weights ⚖️")
    weight1_input = st.number_input(f"Select the % of your portfolio {symbols[0]} will represent",)
    weight2_input = st.number_input(f"Select the % of your portfolio {symbols[1]} will represent",)

    st.warning(f"What is the return for your Portfolio ❓")
    st.warning(f"What is the Standard deviation of returns for your Portfolio ❓")



    # Historical returns of assets in the portfolio
    asset1_returns = np.array([0.1, 0.05, 0.02, 0.08, -0.03])
    asset2_returns = np.array([0.05, 0.02, 0.07, -0.01, 0.04])
    #asset1_returns = df_inter_2[df_inter_2['symbol'] == symbols[0]]['Close'].pct_change().dropna()
    #asset2_returns = df_inter_2[df_inter_2['symbol'] == symbols[1]]['Close'].pct_change().dropna()
    # Weights of assets in the portfolio
    weight_asset1 = 0.4
    weight_asset2 = 0.6

    # Calculate the portfolio return
    portfolio_returns = (weight_asset1 * asset1_returns) + (weight_asset2 * asset2_returns)
    portfolio_return = np.sum(portfolio_returns)

    # Calculate the standard deviation of the portfolio returns
    portfolio_std = np.std(portfolio_returns, ddof=1)


    hint2 = st.checkbox('Hint 2 💡')

    if hint2:
        st.write("""
        To calculate the returns of a portfolio, you need to know the individual weights and returns of each asset within the portfolio. The returns can be calculated using the following formula:

    Portfolio Return = (Weight of Asset 1 * Return of Asset 1) + (Weight of Asset 2 * Return of Asset 2) + ... + (Weight of Asset n * Return of Asset n)

    For example, if you have a portfolio with two assets, Asset 1 and Asset 2, and their respective weights are 0.6 and 0.4, and their returns are 0.1 and 0.05, the portfolio return would be:

    Portfolio Return = (0.6 * 0.1) + (0.4 * 0.05) = 0.06 + 0.02 = 0.08 or 8%

    To calculate the standard deviation of the returns, you'll need historical data of the returns of the portfolio. Follow these steps:

    1. Calculate the average return (mean) of the portfolio returns over a specified period.

    2. Calculate the difference between each individual return and the average return, square the differences, and sum them up.

    3. Divide the sum by the number of returns minus one.

    4. Take the square root of the result obtained in step 3 to get the standard deviation.

    Here's the formula for calculating the standard deviation:

    Standard Deviation = sqrt((Σ(Ri - Rmean)^2) / (n - 1))

    Where:
    - Ri represents the individual returns of the portfolio.
    - Rmean is the average return of the portfolio.
    - Σ represents the sum of the values.
    - n is the number of returns in the data set.

    Once you have the portfolio return and standard deviation, you can use them to assess the risk and performance of the portfolio.

        """)

    solution2 = st.checkbox('Solution 2 ✅')

    if solution2:
        st.success(f"Portfolio Return: {np.round(portfolio_return,4)}")
        st.success(f"Portfolio Standard Deviation:{np.round(portfolio_std,4)}")



    st.markdown("#### Q3 - Same Question but - Include Short Sell Conditions - Combine  two assets (one risky and one risk-free asset) into portfolios assuming that short-sell constraints are in place. Calculate the expected returns and standard deviation of the portfolio.")

    st.success("If we want to include short sales in the portfolio, it means we can have negative weights for assets, indicating short positions. ")



    st.write("Define the amount you want to put in each stock in 💵")
    stock1_input = st.number_input(f"Select the amount you want to invest in {symbols[0]} stocks",2000)
    stock2_input = st.number_input(f"Select the amount you want to invest in {symbols[1]} stocks",30000)

    st.write("Define the weight (the sum of weights ⚖️")
    weight1_input = st.number_input(f"Select the % of your portfolio {symbols[0]} will represent",60)
    weight2_input = st.number_input(f"Select the % of your portfolio {symbols[1]} will represent",-40)

    # Historical returns of assets in the portfolio
    asset1_returns_new = np.array([0.1, 0.05, 0.02, 0.08, -0.03])
    asset2_returns_new = np.array([0.05, 0.02, 0.07, -0.01, 0.04])
    #asset1_returns = df_inter_2[df_inter_2['symbol'] == symbols[0]]['Close'].pct_change().dropna()
    #asset2_returns = df_inter_2[df_inter_2['symbol'] == symbols[1]]['Close'].pct_change().dropna()
    # Weights of assets in the portfolio
    weight_asset1_new = 0.4
    weight_asset2_new = 0.6

    # Calculate the portfolio return
    portfolio_returns_new = (weight_asset1_new * asset1_returns_new) + (weight_asset2_new * asset2_returns_new)
    portfolio_return_new = np.sum(portfolio_returns_new)

    # Calculate the standard deviation of the portfolio returns
    portfolio_std_new = np.std(portfolio_returns_new, ddof=1)

    st.warning(f"What is the return for your Portfolio ❓ take into account the short sell condition")
    st.warning(f"What is the Standard deviation of returns for your Portfolio ❓ take into account the short sell condition")



    hint3 = st.checkbox('Hint 3 💡')

    if hint3:
        st.write("""

    If we want to include short sales in the portfolio, it means we can have negative weights for assets, indicating short positions. Let's modify the example to include short sales:

    Let's say we have the same two assets, Asset A and Asset B, with the following annual returns:

    Asset A: 10%
    Asset B: 8%
    Now, let's assume we want to create a portfolio with the following allocations:

    Asset A: 60%
    Asset B: -40%
    To calculate the return of the portfolio, we'll follow a similar approach as before:

    Step 1: Calculate the weighted returns of each asset, considering short positions.
    Weighted Return of Asset A = Weight of Asset A * Return of Asset A
    Weighted Return of Asset B = Weight of Asset B * Return of Asset B

    In this case:
    Weighted Return of Asset A = 0.6 * 0.10 = 0.06 (or 6%)
    Weighted Return of Asset B = (-0.4) * 0.08 = -0.032 (or -3.2%)

    Step 2: Sum up the weighted returns of the two assets.
    Portfolio Return = Weighted Return of Asset A + Weighted Return of Asset B

    In this case:
    Portfolio Return = 0.06 + (-0.032) = 0.028 (or 2.8%)

    Therefore, considering the short position in Asset B, the return of the portfolio, with the given asset weights and individual returns, is 2.8%.

    Keep in mind that short sales introduce additional risks and complexities to the portfolio, such as borrowing costs and potential losses from negative returns


        """)

    solution3 = st.checkbox('Solution 3 ✅')

    if solution3:
        st.success(f"Portfolio Return: {np.round(portfolio_return_new,4)}")
        st.success(f"Portfolio Standard Deviation:{np.round(portfolio_std_new,4)}")



    st.markdown("#### Q4 - Find a portfolio with the highest and lowest expected return when short-sell constraints are in place and when they are relaxed")



    symbols_1 = ['AAPL', 'AMZN', 'IBM']
    symbols_2 = ['PG','JPM','WMT']
    symbols_3 = ['ADBE','AXP','BBY','BA','CSCO']


    #Portfolio 1

    list_dataframes_1 = []
    for i in range(0,len(symbols_1)):
        data_1 = yf.Ticker(symbols_1[i])
        df_data_1 = data_1.history(period="16mo")
        df_data_1['Date'] = pd.to_datetime(df_data_1.index).date.astype(str)
        df_data_1["symbol"] = [symbols_1[i]]*len(list(df_data_1.index))

        #df_data = pdr.get_data_yahoo(symbols[i])
        list_dataframes_1.append(df_data_1)
    df_master_1 = pd.concat(list_dataframes_1).reset_index(drop=True)
    df_master_1 = df_master_1[pd.to_datetime(df_master_1['Date']) > pd.to_datetime(start_date)]


    # Portfolio 2
    list_dataframes_2 = []
    for i in range(0,len(symbols_2)):
        data_2 = yf.Ticker(symbols_2[i])
        df_data_2 = data_2.history(period="16mo")
        df_data_2['Date'] = pd.to_datetime(df_data_2.index).date.astype(str)
        df_data_2["symbol"] = [symbols_2[i]]*len(list(df_data_2.index))

        #df_data = pdr.get_data_yahoo(symbols[i])
        list_dataframes_2.append(df_data_2)
    df_master_2 = pd.concat(list_dataframes_2).reset_index(drop=True)
    df_master_2 = df_master_2[pd.to_datetime(df_master_2['Date']) > pd.to_datetime(start_date)]





    # Portfolio 3
    list_dataframes_3 = []
    for i in range(0,len(symbols_3)):
        data_3 = yf.Ticker(symbols_3[i])
        df_data_3 = data_3.history(period="16mo")
        df_data_3['Date'] = pd.to_datetime(df_data_3.index).date.astype(str)
        df_data_3["symbol"] = [symbols_3[i]]*len(list(df_data_3.index))

        #df_data = pdr.get_data_yahoo(symbols[i])
        list_dataframes_3.append(df_data_3)
    df_master_3 = pd.concat(list_dataframes_3).reset_index(drop=True)
    df_master_3 = df_master_3[pd.to_datetime(df_master_3['Date']) > pd.to_datetime(start_date)]

    st.write("Portfolio 1 contains the following stocks:",symbols_1)



    csv_df = convert_df(df_master_1)

    st.download_button(
        label=f"📥 Download **Portfolio1** stocks as csv",
        data=csv_df,
        file_name=f'Portfolio1.csv',
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
    excel_writer = pd.ExcelWriter(f"Portfolio1.xlsx", engine='xlsxwriter')
    df_master_1.to_excel(excel_writer, index=False, sheet_name='Sheet1')
    excel_writer.close()

                # Download the file
    with open(f"Portfolio1.xlsx", "rb") as f:
                st.download_button(
                        label=f"📥 Download **Portfolio1** stocks as xlsx",
                        data=f,
                        file_name=f"Portfolio1.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

    st.write("Portfolio 2 contains the following stocks:",symbols_2)
    csv_df = convert_df(df_master_2)

    st.download_button(
        label=f"📥 Download **Portfolio2** stocks as csv",
        data=csv_df,
        file_name=f'Portfolio2.csv',
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
    excel_writer = pd.ExcelWriter(f"Portfolio2.xlsx", engine='xlsxwriter')
    df_master_2.to_excel(excel_writer, index=False, sheet_name='Sheet1')
    excel_writer.close()

                # Download the file
    with open(f"Portfolio2.xlsx", "rb") as f:
                st.download_button(
                        label=f"📥 Download **Portfolio2** stocks as xlsx",
                        data=f,
                        file_name=f"Portfolio2.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

    st.write("Portfolio 3 contains the following stocks:",symbols_3)
    csv_df = convert_df(df_master_3)

    st.download_button(
        label=f"📥 Download **Portfolio3** stocks as csv",
        data=csv_df,
        file_name=f'Portfolio3.csv',
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
    excel_writer = pd.ExcelWriter(f"Portfolio3.xlsx", engine='xlsxwriter')
    df_master_3.to_excel(excel_writer, index=False, sheet_name='Sheet1')
    excel_writer.close()

                # Download the file
    with open(f"Portfolio3.xlsx", "rb") as f:
                st.download_button(
                        label=f"📥 Download **Portfolio3** stocks as xlsx",
                        data=f,
                        file_name=f"Portfolio3.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )


    st.write(" ")
    st.write(" ")
    st.warning(f"**Find a portfolio with the highest and lowest expected return when short-sell constraints are in place**")

    st.write("Check Results ")
    user_input = st.text_input("Enter your results","portfolio1")

    if user_input.lower() == "portfolio2":
        st.success("You have the good answer!")
    else:
        st.warning("Try again")

    st.write(" ")
    st.write(" ")
    st.write(f"**Find a portfolio with the highest and lowest expected return when short-sell constraints are relaxed**")

    st.write("Check Results ✅")
    user_input = st.text_input("Enter your results","portfolio2")

    if user_input.lower() == "portfolio3":
        st.success("You have the good answer!")
    else:
        st.warning("Try again")








    st.markdown(" ")
    st.markdown(" ")
    st.markdown("#### Congratulations you masterise the Portfolio theory 🎉")







if lab_numbers == "02 - Two risky assets":

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

