import streamlit as st
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
import time

st.title("AI Investment Agent 📈🤖")
st.caption("This app allows you to compare the performance of two stocks and generate detailed reports.")

hf_token = st.text_input("Hugging Face Token", type="password")

# Initialize session state variables if they don't exist
if "response" not in st.session_state:
    st.session_state.response = None
if "tool_outputs" not in st.session_state:
    st.session_state.tool_outputs = []

# Model selection with fully open models only
default_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
with st.sidebar:
    st.subheader("Model Settings")
    model_option = st.selectbox(
        "Select Hugging Face model",
        [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "microsoft/phi-2",
            "facebook/opt-1.3b"
        ],
        index=0,
        help="Choose a model to use for analysis. All options are open-access models."
    )

def get_stock_price(ticker, period="1y"):
    """Get historical stock prices."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return f"No price data available for {ticker}"
        return hist
    except Exception as e:
        return f"Error fetching stock price data for {ticker}: {str(e)}"

def get_analyst_recommendations(ticker):
    """Get analyst recommendations for a stock."""
    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        if recommendations is None or recommendations.empty:
            return f"No analyst recommendations available for {ticker}"
        return recommendations
    except Exception as e:
        return f"Error fetching analyst recommendations for {ticker}: {str(e)}"

def get_stock_fundamentals(ticker):
    """Get fundamental data for a stock."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info:
            return f"No fundamental data available for {ticker}"
        
        # Extract the most relevant financial metrics
        fundamentals = {
            "Market Cap": info.get("marketCap"),
            "Forward P/E": info.get("forwardPE"),
            "Trailing P/E": info.get("trailingPE"),
            "Price to Book": info.get("priceToBook"),
            "Dividend Yield": info.get("dividendYield"),
            "EPS": info.get("trailingEps"),
            "Revenue Growth": info.get("revenueGrowth"),
            "Profit Margins": info.get("profitMargins"),
            "52 Week High": info.get("fiftyTwoWeekHigh"),
            "52 Week Low": info.get("fiftyTwoWeekLow"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry")
        }
        
        return fundamentals
    except Exception as e:
        return f"Error fetching stock fundamentals for {ticker}: {str(e)}"

def generate_report(stock1, stock2, tool_outputs, model_name):
    """Generate investment report using selected HF model."""
    try:
        # Configure model-specific settings
        model_prompt_formats = {
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
                "prompt_format": "<|system|>\nYou are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.\n<|user|>\nI have collected data on two stocks: {stock1} and {stock2}. Please analyze this data and provide a detailed investment comparison report.\n\nHere is the data I have collected:\n\n{tool_outputs}\n\nPlease format your response using markdown and include sections on:\n1. Price performance comparison\n2. Fundamental analysis comparison\n3. Analyst sentiment comparison (if available)\n4. Investment recommendation based on the data\n<|assistant|>",
                "response_extraction": lambda x: x.split("<|assistant|>")[-1].strip() if "<|assistant|>" in x else x
            },
            "microsoft/phi-2": {
                "prompt_format": "Instruct: You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.\n\nI have collected data on two stocks: {stock1} and {stock2}. Please analyze this data and provide a detailed investment comparison report.\n\nHere is the data I have collected:\n\n{tool_outputs}\n\nPlease format your response using markdown and include sections on:\n1. Price performance comparison\n2. Fundamental analysis comparison\n3. Analyst sentiment comparison (if available)\n4. Investment recommendation based on the data\n\nOutput:",
                "response_extraction": lambda x: x.split("Output:")[-1].strip() if "Output:" in x else x
            },
            "facebook/opt-1.3b": {
                "prompt_format": "You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.\n\nI have collected data on two stocks: {stock1} and {stock2}. Please analyze this data and provide a detailed investment comparison report.\n\nHere is the data I have collected:\n\n{tool_outputs}\n\nPlease format your response using markdown and include sections on:\n1. Price performance comparison\n2. Fundamental analysis comparison\n3. Analyst sentiment comparison (if available)\n4. Investment recommendation based on the data\n\n",
                "response_extraction": lambda x: x
            }
        }
        
        # Select the appropriate prompt format
        prompt_config = model_prompt_formats.get(model_name, model_prompt_formats[default_model])
        
        # Prepare the prompt
        prompt = prompt_config["prompt_format"].format(
            stock1=stock1,
            stock2=stock2,
            tool_outputs=tool_outputs
        )
        
        # Load model and tokenizer with error handling
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                device_map="auto",
                torch_dtype=torch.float16  # Use float16 to reduce memory usage
            )
        except Exception as e:
            return f"Error loading model {model_name}: {str(e)}\n\nPlease try a different model or check your token."
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate with appropriate parameters for the model size
        with torch.no_grad():
            try:
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=800,  # Smaller to ensure it completes faster
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            except Exception as e:
                return f"Error during text generation: {str(e)}\n\nThis might be due to memory constraints. Try using a smaller model like TinyLlama."
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Process the response according to model-specific requirements
        processed_response = prompt_config["response_extraction"](response)
        
        return processed_response
    except Exception as e:
        return f"Error generating report: {str(e)}"

def summarize_recommendations(recs_data, ticker):
    """Safely summarize recommendations dataframe."""
    if isinstance(recs_data, str):  # Error message
        return recs_data
    
    if not isinstance(recs_data, pd.DataFrame) or recs_data.empty:
        return f"No recommendations data available for {ticker}"
    
    # Try to get the number of available recommendations
    try:
        num_recs = len(recs_data)
        summary = f"- {num_recs} analyst recommendations available"
        
        # Try to extract the date range if possible
        if hasattr(recs_data.index, 'min') and hasattr(recs_data.index, 'max'):
            try:
                start_date = recs_data.index.min()
                end_date = recs_data.index.max()
                
                # Check if these are datetime objects and can be formatted
                if hasattr(start_date, 'strftime') and hasattr(end_date, 'strftime'):
                    date_range = f" (from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
                    summary += date_range
            except:
                pass
        
        # Try to summarize by recommendation type if available
        found_column = False
        for col_name in ['To Grade', 'grade', 'recommendation']:
            if col_name in recs_data.columns:
                try:
                    counts = recs_data[col_name].value_counts()
                    summary += f"\n{counts.to_markdown()}"
                    found_column = True
                    break
                except:
                    pass
                    
        if not found_column:
            # List available columns as a fallback
            summary += f"\nAvailable data columns: {', '.join(recs_data.columns.tolist())}"
                    
        return summary
    except Exception as e:
        return f"Error summarizing recommendations for {ticker}: {str(e)}"

def calculate_returns(price_data, ticker):
    """Safely calculate returns from price data."""
    if isinstance(price_data, str):  # Error message
        return price_data
    
    if not isinstance(price_data, pd.DataFrame) or price_data.empty:
        return f"No price data available for {ticker}"
    
    try:
        if 'Close' not in price_data.columns:
            return f"Price data for {ticker} does not contain 'Close' prices"
        
        if len(price_data['Close']) < 2:
            return f"Insufficient price history for {ticker} to calculate returns"
        
        current_price = price_data['Close'].iloc[-1]
        start_price = price_data['Close'].iloc[0]
        percent_return = ((current_price / start_price) - 1) * 100
        
        return {
            'current_price': current_price,
            'start_price': start_price,
            'percent_return': percent_return
        }
    except Exception as e:
        return f"Error calculating returns for {ticker}: {str(e)}"

if hf_token:
    col1, col2 = st.columns(2)
    with col1:
        stock1 = st.text_input("Enter first stock symbol (e.g. AAPL)")
    with col2:
        stock2 = st.text_input("Enter second stock symbol (e.g. MSFT)")

    if stock1 and stock2:
        if st.button("Analyze Stocks"):
            # Validate ticker symbols before proceeding
            if not stock1.strip() or not stock2.strip():
                st.error("Please enter valid stock symbols")
            else:
                with st.spinner(f"Analyzing {stock1} and {stock2}..."):
                    start_time = time.time()
                    
                    # Clear previous tool outputs
                    st.session_state.tool_outputs = []
                    progress_bar = st.progress(0)
                    
                    # Tool calls and collecting data
                    with st.expander("Data Collection", expanded=True):
                        # Get stock prices
                        st.write("📊 Getting historical stock prices...")
                        stock1_prices = get_stock_price(stock1)
                        stock2_prices = get_stock_price(stock2)
                        progress_bar.progress(0.25)
                        
                        # Process price data
                        price_data = "## Stock Price Data\n"
                        
                        # Calculate returns for stock1
                        stock1_return = calculate_returns(stock1_prices, stock1)
                        if isinstance(stock1_return, dict):
                            price_data += f"- {stock1} 1-year return: {stock1_return['percent_return']:.2f}%\n"
                            price_data += f"- {stock1} current price: ${stock1_return['current_price']:.2f}\n"
                        else:
                            price_data += f"- {stock1}: {stock1_return}\n"
                        
                        # Calculate returns for stock2
                        stock2_return = calculate_returns(stock2_prices, stock2)
                        if isinstance(stock2_return, dict):
                            price_data += f"- {stock2} 1-year return: {stock2_return['percent_return']:.2f}%\n"
                            price_data += f"- {stock2} current price: ${stock2_return['current_price']:.2f}\n"
                        else:
                            price_data += f"- {stock2}: {stock2_return}\n"
                        
                        st.session_state.tool_outputs.append(price_data)
                        st.write(price_data)
                        
                        # Get analyst recommendations
                        st.write("👩‍💼 Getting analyst recommendations...")
                        stock1_recs = get_analyst_recommendations(stock1)
                        stock2_recs = get_analyst_recommendations(stock2)
                        progress_bar.progress(0.5)
                        
                        recs_data = "## Analyst Recommendations\n"
                        
                        # Safely process recommendations
                        recs_data += f"\n### {stock1} Recommendations:\n"
                        recs_data += summarize_recommendations(stock1_recs, stock1)
                        
                        recs_data += f"\n\n### {stock2} Recommendations:\n"
                        recs_data += summarize_recommendations(stock2_recs, stock2)
                        
                        st.session_state.tool_outputs.append(recs_data)
                        st.write(recs_data)
                        
                        # Get fundamentals
                        st.write("📝 Getting stock fundamentals...")
                        stock1_fund = get_stock_fundamentals(stock1)
                        stock2_fund = get_stock_fundamentals(stock2)
                        progress_bar.progress(0.75)
                        
                        fund_data = "## Fundamental Data\n"
                        
                        # Process fundamental data
                        if isinstance(stock1_fund, dict) and isinstance(stock2_fund, dict):
                            # Create comparative table
                            fund_df_data = {
                                'Metric': [],
                                f'{stock1}': [],
                                f'{stock2}': []
                            }
                            
                            # Collect all available keys
                            all_keys = set(stock1_fund.keys()).union(set(stock2_fund.keys()))
                            
                            # Build the dataframe row by row
                            for key in all_keys:
                                fund_df_data['Metric'].append(key)
                                fund_df_data[f'{stock1}'].append(stock1_fund.get(key, "N/A"))
                                fund_df_data[f'{stock2}'].append(stock2_fund.get(key, "N/A"))
                                
                            fundamentals_df = pd.DataFrame(fund_df_data)
                            fund_data += fundamentals_df.to_markdown(index=False)
                        else:
                            if isinstance(stock1_fund, str):
                                fund_data += f"{stock1}: {stock1_fund}\n\n"
                            if isinstance(stock2_fund, str):
                                fund_data += f"{stock2}: {stock2_fund}\n"
                        
                        st.session_state.tool_outputs.append(fund_data)
                        st.write(fund_data)
                    
                    # Combine all tool outputs
                    combined_tool_outputs = "\n\n".join(st.session_state.tool_outputs)
                    
                    # Generate final report
                    st.write(f"🤖 Generating investment report using {model_option}...")
                    progress_bar.progress(0.9)
                    
                    with st.spinner("This may take a moment..."):
                        st.session_state.response = generate_report(stock1, stock2, combined_tool_outputs, model_option)
                    
                    progress_bar.progress(1.0)
                    elapsed_time = time.time() - start_time
                    st.success(f"Analysis completed in {elapsed_time:.1f} seconds!")
                
                # Display the response
                st.markdown("## Investment Analysis Report")
                st.markdown(st.session_state.response)
    
    # Add information about the app
    with st.sidebar:
        st.subheader("About")
        st.write("""
        This app uses:
        - Open-source Hugging Face models for analysis
        - yfinance for fetching stock data
        - Streamlit for the web interface
        
        No data is stored or shared beyond what's needed for analysis.
        """)

    # Add usage instructions
    with st.sidebar:
        st.subheader("How to Use")
        st.write("""
        1. Enter your Hugging Face API token
        2. Input two stock symbols to compare
        3. Click "Analyze Stocks"
        4. Review the AI-generated investment report
        
        Example symbols: AAPL, MSFT, GOOGL, AMZN, TSLA
        """)
        
    # Add memory optimization tips
    with st.sidebar:
        st.subheader("Troubleshooting")
        st.info("""
        If you encounter errors:
        - Verify stock symbols are correct
        - Try TinyLlama model for lower memory usage
        - Check your Hugging Face token has API access
        - Some tickers may have limited data available
        """)
else:
    st.info("Please enter your Hugging Face token to start using the app.")
