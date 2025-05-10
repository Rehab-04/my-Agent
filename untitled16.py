import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import streamlit as st
from datetime import datetime, timedelta

# Load dataset
df = pd.read_csv("jordan_transactions.csv")
df['transaction_date'] = pd.to_datetime(df['transaction_date'], dayfirst=True, errors='coerce')

# Enrich with new features
df['day_of_week'] = df['transaction_date'].dt.day_name()
df['time_of_day'] = df['transaction_date'].dt.strftime('%H:%M')
df['amount_level'] = df['transaction_amount'].apply(lambda x: 'high value' if x > 10 else 'low value')
df['city'] = df['branch_name'].str.extract(r'\b(Amman|Irbid|Zarqa|Aqaba|Madaba)\b', expand=False).fillna("Unknown")

# Create enriched semantic knowledge base
corpus = df.apply(lambda row:
    f"{row['amount_level']} {row['transaction_status']} {row['transaction_type']} transaction at {row['mall_name']} "
    f"({row['branch_name']}) in {row['city']} on {row['day_of_week']} at {row['time_of_day']} for "
    f"{row['transaction_amount']} JOD (tax: {row['tax_amount']} JOD) [ID: {row['transaction_id']}]",
    axis=1
).tolist()

# Embed using SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(corpus, show_progress_bar=True)

# Store in FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Load FLAN-T5 model for text generation
llm = pipeline("text2text-generation", model="google/flan-t5-large")

def get_yesterday():
    return (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

def smart_agent_with_kb(query):
    yesterday_date = get_yesterday()

    # Embed the query
    query_vec = model.encode([query])
    D, I = index.search(query_vec, k=10)

    # Get top rows from the knowledge base
    top_rows = [corpus[i] for i in I[0]]
    context = "\n- " + "\n- ".join(top_rows)

    # Refine the prompt to make the model's response more targeted to your query
    prompt = f"""
You're a helpful assistant analyzing financial transactions.

Here are some relevant transactions:
{context}

Answer the following question clearly, briefly, and without repeating yourself:

Question: {query}
Answer:
"""
    # Get response from FLAN-T5 model
    response = llm(prompt, max_length=256, do_sample=False)[0]['generated_text']
    return response

# Streamlit UI setup
st.title("Smart Transaction Analyst")
st.write("""
Enter a financial question about the transactions, like:

- What are the top refund branches?
- Any unusual failed transactions?
- Are there patterns in high-value transactions?
""")

# Get user input query
query = st.text_input("🔍 Your Query:")

# Only run the function if input is provided
if query:
    st.write("✅ Analysis Complete")
    result = smart_agent_with_kb(query)
    st.write("📊 Answer:", result)

# Workflow automation for additional checks (run independently)
def run_workflow_automation(df):
    actions = []

    # 1. Flag branches with more than 5 failed transactions
    failed_counts = df[df['transaction_status'] == 'Failed'].groupby('branch_name').size()
    for branch, count in failed_counts.items():
        if count > 5:
            actions.append(f"🚨 Branch '{branch}' has {count} failed transactions. Consider reviewing operations.")

    # 2. Alert on high refund rate per branch
    total_per_branch = df.groupby('branch_name').size()
    refunds_per_branch = df[df['transaction_status'] == 'Refunded'].groupby('branch_name').size()
    refund_rate = (refunds_per_branch / total_per_branch).fillna(0)

    for branch, rate in refund_rate.items():
        if rate > 0.2:
            actions.append(f"⚠ Branch '{branch}' has a high refund rate ({rate:.0%}). Investigate refund policies.")

    # 3. Detect high-value anomalies (e.g. > 500 JOD with failure or refund)
    anomalies = df[
        (df['transaction_amount'] > 500) & 
        (df['transaction_status'].isin(['Failed', 'Refunded']))
    ]

    for _, row in anomalies.iterrows():
        actions.append(
            f"🔍 High-value {row['transaction_status']} of {row['transaction_amount']} JOD at {row['mall_name']} - {row['branch_name']} on {row['transaction_date'].date()}"
        )

    # Only print actions related to workflow automation here (will not display with user query result)
    if actions:
        for action in actions:
            st.write(action)

# Only run workflow automation if necessary (e.g., upon initial load or via button click)
if st.button('Run Workflow Automation'):
    run_workflow_automation(df)

# Function to flag failed high-value transactions
def auto_flag_failed_high_value(df):
    flagged = df[(df['transaction_status'] == 'Failed') & (df['transaction_amount'] > 100)]
    if not flagged.empty:
        st.write("⚠ Alert: High-value failed transactions detected!")
        for _, row in flagged.iterrows():
            st.write(f"🔍 {row['transaction_status']} of {row['transaction_amount']} JOD at {row['mall_name']} - {row['branch_name']} on {row['transaction_date'].date()}")
