import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

st.title("Invoice Extractor")
st.write("Upload a PDF invoice and get structured JSON output.")

# Upload PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_invoice.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully!")

    # Step 1: Load PDF
    loader = PyPDFLoader("temp_invoice.pdf")
    docs = loader.load()
    invoice_text = "\n".join([doc.page_content for doc in docs])

    # Step 2: Define prompt
    parser = JsonOutputParser()
    prompt_template = """
    You are an expert at reading invoices.

    Extract the following information from the text below:

    - Invoice Number
    - Invoice Date
    - Vendor / Supplier Name
    - Vendor TRN / VAT
    - Bill To (Customer Name, Address, TRN)
    - Total Amount
    - Balance Due
    - Line Items (Item Name, Quantity, Rate, Taxable Amount, Tax Amount, Total)
    - Sub Total
    - Tax Summary (Taxable Amount, Tax Amount, Total, Tax Rate)
    - Bank Details (Account Holder, Account Number, IBAN, BIC)
    - Business Address

    Invoice Text:
    {invoice_text}

    Provide the output in **JSON format only** with keys exactly like:
    invoice_number, invoice_date, vendor, bill_to, total_amount, balance_due, line_items, sub_total, tax_summary, bank_details, business_address
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["invoice_text"],
        partial_variables={'format_instruction': parser.get_format_instructions()}
    )

    # Step 3: Initialize LLM
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Step 4: Create chain
    chain = prompt | model | parser

    # Step 5: Run chain
    with st.spinner("Extracting invoice data..."):
        result_json = chain.invoke({'invoice_text': invoice_text})

    st.success("Extraction complete!")

    # Step 6: Display JSON
    st.subheader("Extracted Invoice Data")
    st.json(result_json)

    # Optional: download JSON
    st.download_button(
        label="Download JSON",
        data=json.dumps(result_json, indent=4),
        file_name="invoice_data.json",
        mime="application/json"
    )
