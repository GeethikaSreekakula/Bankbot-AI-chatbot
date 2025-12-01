# milestone3_updated.py - BankBot AI (Updated for new dataset)
# Infosys Springboard Project - Milestone 3

import streamlit as st
import pandas as pd
import joblib
import spacy
import re
import random
from datetime import datetime
import time
import base64

# =============================================
# PAGE CONFIGURATION
# =============================================

st.set_page_config(
    page_title="Finexa Bank - BankBot AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================
# LOAD LOGO
# =============================================

def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

logo_base64 = get_image_base64("logo.png")

# =============================================
# CUSTOM CSS
# =============================================

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .block-container {
        padding-top: 2rem;
        max-width: 100%;
    }
    
    .navbar {
        background: linear-gradient(135deg, #0d4f5c 0%, #1a7a8a 100%);
        padding: 1.2rem 3rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin: -2rem -1rem 2rem -1rem;
    }
    
    .logo img {
        height: 45px;
    }
    
    .logo-text {
        color: white;
        font-size: 1.5rem;
        font-weight: 800;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        transition: all 0.3s;
        border: 2px solid transparent;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: #0d4f5c;
    }
    
    .feature-card h3 {
        color: #0d4f5c;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    .feature-card p {
        color: #374151;
        font-size: 1rem;
        line-height: 1.6;
        margin: 0;
    }
    
    .stChatMessage {
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stChatMessageContent"] {
        background: #f3f4f6 !important;
        color: #1f2937 !important;
        border-radius: 12px !important;
        padding: 1rem 1.5rem !important;
        border: 1px solid #e5e7eb !important;
    }
    
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #0d4f5c 0%, #1a7a8a 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    .intent-badge {
        display: inline-block;
        background: #8b6f47;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
        text-transform: uppercase;
    }
    
    .fallback-badge {
        background: #ef4444;
    }
    
    .stButton button {
        background: #8b6f47 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
    }
    
    .stButton button:hover {
        background: #6d5637 !important;
        transform: translateY(-2px) !important;
    }
    
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 5px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# LOAD RESOURCES
# =============================================

@st.cache_resource
def load_resources():
    try:
        clf = joblib.load("intent_pipeline.joblib")
        nlp = spacy.load("en_core_web_sm")
        data = pd.read_csv("bankbot_expanded_dataset.csv", on_bad_lines='skip', encoding='utf-8')
        
        # Handle missing values
        data = data.dropna(subset=['text', 'intent'])
        data['response'] = data['response'].fillna('')
        data['entities'] = data['entities'].fillna('[]')
        
        return clf, nlp, data, None
    except Exception as e:
        return None, None, None, str(e)

clf, nlp, data, error = load_resources()

@st.cache_data
def build_response_database(_data):
    intent_responses = {}
    intent_text_map = {}
    
    if _data is not None:
        for _, row in _data.iterrows():
            intent = row['intent']
            text = str(row['text']).strip().lower()
            response = str(row['response']).strip()
            
            if pd.isnull(row['response']) or not response or response == 'nan' or response == '':
                continue
            
            if intent not in intent_responses:
                intent_responses[intent] = []
                intent_text_map[intent] = {}
            
            if response and response not in intent_responses[intent]:
                intent_responses[intent].append(response)
            
            intent_text_map[intent][text] = response
    
    return intent_responses, intent_text_map

intent_responses, intent_text_map = build_response_database(data)

# =============================================
# CONTEXT & ENTITIES
# =============================================

REQUIRED_ENTITIES = {
    'check_balance': ['account_number'],
    'balance_enquiry': ['account_number'],
    'account_balance': ['account_number'],
    'mini_statement': ['account_number'],
    'statement_request': ['account_number'],
    'transaction_status': ['transaction_id'],
    'transaction_inquiry': ['transaction_id'],
    'transaction_enquiry': ['transaction_id'],
    'atm_locator': ['pincode'],
    'find_atm': ['pincode'],
    'branch_locator': ['pincode'],
    'find_branch': ['pincode'],
    'transfer_money': ['person', 'money'],
    'money_transfer': ['person', 'money'],
    'send_money': ['person', 'money'],
    'card_block': ['card_number'],
    'block_card': ['card_number'],
    'debit_card_block': ['card_number'],
    'credit_card_block': ['card_number'],
    'close_account': ['account_number'],
    'account_closure': ['account_number'],
    'fd_creation': ['amount', 'tenure'],
    'fixed_deposit_info': ['amount', 'tenure'],
    'rd_creation': ['amount', 'tenure'],
    'recurring_deposit_info': ['amount', 'tenure'],
}

FOLLOWUP_QUESTIONS = {
    'check_balance': {'account_number': "Please provide your account number."},
    'balance_enquiry': {'account_number': "Please provide your account number."},
    'account_balance': {'account_number': "Please provide your account number."},
    'mini_statement': {'account_number': "Please provide your account number."},
    'statement_request': {'account_number': "Please provide your account number."},
    'transaction_status': {'transaction_id': "Please provide the transaction ID."},
    'transaction_inquiry': {'transaction_id': "Please provide the transaction ID."},
    'atm_locator': {'pincode': "Please provide your pincode."},
    'branch_locator': {'pincode': "Please provide your pincode."},
    'transfer_money': {'person': "To whom would you like to send money?", 'money': "How much would you like to transfer?"},
    'money_transfer': {'person': "To whom would you like to send money?", 'money': "How much would you like to transfer?"},
    'card_block': {'card_number': "Please provide your card number."},
    'block_card': {'card_number': "Please provide your card number."},
    'debit_card_block': {'card_number': "Please provide your card number."},
    'credit_card_block': {'card_number': "Please provide your card number."},
    'close_account': {'account_number': "Please provide your account number."},
    'account_closure': {'account_number': "Please provide your account number."},
    'fd_creation': {'amount': "What amount would you like to deposit?", 'tenure': "For how long?"},
    'rd_creation': {'amount': "What monthly amount?", 'tenure': "For how long?"},
}

BANKING_KEYWORDS = [
    'account', 'balance', 'bank', 'banking', 'statement', 'mini',
    'savings', 'current', 'salary', 'loan', 'loans', 'personal', 'home', 
    'car', 'education', 'emi', 'interest', 'rate', 'card', 'cards', 
    'debit', 'credit', 'atm', 'pin', 'block', 'lost', 'stolen',
    'transfer', 'money', 'send', 'payment', 'pay', 'transaction',
    'withdraw', 'deposit', 'cash', 'cheque', 'upi', 'netbanking',
    'fd', 'rd', 'fixed', 'recurring', 'branch', 'near', 'location',
    'pincode', 'ifsc', 'pan', 'kyc', 'nominee'
]

def is_banking_related(text):
    text_lower = text.lower()
    words = text_lower.split()
    return any(kw in words or (len(kw) > 3 and kw in text_lower) for kw in BANKING_KEYWORDS)

def extract_entities(text):
    if nlp is None:
        return {}
    
    doc = nlp(text)
    entities = {}
    
    # SpaCy NER
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["person"] = ent.text
        elif ent.label_ == "MONEY":
            money_value = re.sub(r'[^\d.]', '', ent.text)
            if money_value:
                entities["money"] = money_value
        elif ent.label_ == "DATE":
            entities["date"] = ent.text
    
    # Account number (10-16 digits)
    account_matches = re.findall(r'\b\d{10,16}\b', text)
    if account_matches:
        for match in account_matches:
            if len(match) <= 12:
                entities["account_number"] = match
                break
    
    # Card number (16 digits)
    if re.search(r'\b\d{16}\b', text):
        entities["card_number"] = re.search(r'\b\d{16}\b', text).group()
    
    # Transaction ID
    txn_patterns = [
        r'\b(TXN|txn|TRANS|trans)\w+\b',
        r'\bTXID\d+\b',
        r'\b[A-Z]{3}\d{6,}\b'
    ]
    for pattern in txn_patterns:
        txn_match = re.search(pattern, text, re.IGNORECASE)
        if txn_match:
            entities["transaction_id"] = txn_match.group().upper()
            break
    
    # Pincode
    if any(w in text.lower() for w in ['pincode', 'pin code', 'near', 'location', 'atm', 'branch']):
        pincode_match = re.search(r'\b\d{6}\b', text)
        if pincode_match and pincode_match.group() not in entities.values():
            entities["pincode"] = pincode_match.group()
    
    # Mobile number
    if any(w in text.lower() for w in ['mobile', 'phone', 'contact']):
        mobile_match = re.search(r'\b[6-9]\d{9}\b', text)
        if mobile_match:
            entities["mobile_number"] = mobile_match.group()
    
    # Money amount
    if any(w in text.lower() for w in ['send', 'transfer', 'deposit', 'withdraw', 'pay', 'amount', 'rupees', 'rs', 'fd', 'rd']):
        money_patterns = [
            r'(?:rs\.?|₹)\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'\b(\d{3,}(?:,\d+)*(?:\.\d+)?)\s*(?:rupees|rs|inr)\b',
            r'\b(\d{3,})\b'
        ]
        for pattern in money_patterns:
            money_match = re.search(pattern, text.lower())
            if money_match:
                amount = re.sub(r'[,]', '', money_match.group(1) if len(money_match.groups()) > 0 else money_match.group())
                if amount and "money" not in entities:
                    entities["money"] = amount
                    entities["amount"] = amount
                break
    
    # Tenure
    tenure_match = re.search(r'\b(\d+)\s*(month|months|year|years|day|days)\b', text.lower())
    if tenure_match:
        entities["tenure"] = tenure_match.group()
    
    return entities

def detect_intent_manually(text):
    """Manually detect intent for common phrases."""
    text_lower = text.lower().strip()
    
    # Exact greetings
    greetings = ['hi', 'hello', 'hey', 'hii', 'helo', 'hola']
    if text_lower in greetings:
        return 'greet'
    
    # Greeting with time
    if any(g in text_lower for g in ['good morning', 'good afternoon', 'good evening', 'good night']):
        return 'greet'
    
    # Thanks
    if text_lower in ['thanks', 'thank you', 'thankyou', 'thnks', 'thnx', 'ty']:
        return 'thanks'
    if text_lower.startswith('thank'):
        return 'thanks'
    
    # Goodbye
    if text_lower in ['bye', 'goodbye', 'good bye', 'see you', 'exit', 'quit']:
        return 'goodbye'
    
    # Single word queries
    general_info_keywords = {
        'loan': 'I can help you with loans! We offer personal loans, home loans, car loans, and education loans. What type of loan are you interested in?',
        'loans': 'I can help you with loans! We offer personal loans, home loans, car loans, and education loans. What type of loan are you interested in?',
        'card': 'I can help you with cards! We offer debit cards and credit cards. Would you like information about card types, applying for a card, or managing an existing card?',
        'cards': 'I can help you with cards! We offer debit cards and credit cards. Would you like information about card types, applying for a card, or managing an existing card?',
        'account': 'I can help you with your account! Would you like to check your balance, open a new account, or update account details?',
        'balance': 'I can help you check your account balance. Please provide your account number.',
        'transfer': 'I can help you transfer money. Who would you like to send money to and how much?',
        'atm': 'I can help you find ATMs near you. Please share your pincode or location.',
        'branch': 'I can help you find our bank branches. Please share your pincode or location.'
    }
    
    if text_lower in general_info_keywords:
        return ('general_query', text_lower, general_info_keywords[text_lower])
    
    return None

def get_response(intent, user_text=None):
    """Get response from dataset with smart matching - matches Milestone 2 logic."""
    
    # Try exact text match first (same as Milestone 2)
    if user_text and intent in intent_text_map:
        user_text_lower = user_text.strip().lower()
        if user_text_lower in intent_text_map[intent]:
            return intent_text_map[intent][user_text_lower]
    
    # Get from intent responses - USE FIRST RESPONSE (consistent behavior)
    if intent in intent_responses and intent_responses[intent]:
        # Filter out bad responses
        valid_responses = [r for r in intent_responses[intent] 
                          if r and len(r) > 10 and 'provide further details' not in r.lower() 
                          and r != 'nan']
        if valid_responses:
            return valid_responses[0]  # Changed from random.choice() to [0]
    
    # Comprehensive fallback responses for common intents
    fallback_responses = {
        'greet': "Hello! Welcome to Finexa Bank. How can I assist you with your banking needs today?",
        'greeting': "Hello! Welcome to Finexa Bank. How can I assist you with your banking needs today?",
        'thanks': "You're welcome! Is there anything else I can help you with?",
        'thank_you': "You're welcome! Is there anything else I can help you with?",
        'goodbye': "Thank you for contacting Finexa Bank. Have a great day!",
        'bye': "Thank you for contacting Finexa Bank. Have a great day!",
        
        # Banking specific fallbacks
        'check_balance': "To check your balance, I'll need your account number. Please provide your 10-digit account number.",
        'balance_enquiry': "To check your balance, I'll need your account number. Please provide your 10-digit account number.",
        'transfer_money': "I can help you transfer money. Please tell me the recipient's name and the amount you'd like to send.",
        'loan_info': "I can provide information about our loan products including Home Loans, Personal Loans, Car Loans, and Education Loans. Which type interests you?",
        'loan_application': "I can help you apply for a loan. What type of loan are you interested in?",
        'card_block': "I can help you block your card immediately. Please provide your 16-digit card number.",
        'atm_locator': "I can help you find the nearest ATM. Please share your pincode or location.",
        'branch_locator': "I can help you find our nearest branch. Please share your pincode or area.",
        'general_banking_info': "I can provide information about our banking services including accounts, cards, loans, and digital banking. What would you like to know more about?",
        
        # Generic fallback
        'fallback': "I'm here to help with banking services. You can ask me about account balance, money transfers, loans, cards, ATM locations, and more."
    }
    
    return fallback_responses.get(intent, "I can help you with that. Could you please provide more specific details about what you need?")

def get_followup(intent, entity):
    if intent in FOLLOWUP_QUESTIONS and entity in FOLLOWUP_QUESTIONS[intent]:
        return FOLLOWUP_QUESTIONS[intent][entity]
    return f"Please provide {entity.replace('_', ' ')}."

def check_missing(intent, entities):
    if intent not in REQUIRED_ENTITIES:
        return None
    for req in REQUIRED_ENTITIES[intent]:
        if req not in entities:
            return req
    return None

def process_message(user_input, context):
    if clf is None:
        return "System error. Please make sure the model is trained.", 'error', 0.0, {}, context
    
    user_lower = user_input.lower().strip()
    
    # Manual detection (HIGHEST PRIORITY)
    manual_intent = detect_intent_manually(user_input)
    if manual_intent:
        # Check if it's a tuple (general query)
        if isinstance(manual_intent, tuple) and manual_intent[0] == 'general_query':
            keyword = manual_intent[1]
            response = manual_intent[2]
            return response, 'info_' + keyword, 0.95, {}, context
        
        # Regular manual intent
        if manual_intent == 'goodbye':
            context.clear()
        return get_response(manual_intent, user_input), manual_intent, 1.0, {}, context
    
    current_entities = extract_entities(user_input)
    
    # If waiting for entity
    if context.get('waiting_for') and context.get('current_intent'):
        if context['waiting_for'] in current_entities:
            context['entities'].update(current_entities)
            intent = context['current_intent']
            
            missing = check_missing(intent, context['entities'])
            if missing:
                context['waiting_for'] = missing
                return get_followup(intent, missing), intent, 0.85, context['entities'], context
            else:
                response = get_response(intent)
                entities = context['entities'].copy()
                context.clear()
                return response, intent, 0.95, entities, context
        else:
            if is_banking_related(user_input) and not any(w in user_lower for w in ['yes', 'no', 'cancel']):
                context.clear()
            else:
                return get_followup(context['current_intent'], context['waiting_for']), context['current_intent'], 0.5, context['entities'], context
    
    # Check if banking-related
    if not is_banking_related(user_input):
        msg = "I'm a banking assistant. I can help with accounts, loans, cards, transfers, and other banking services."
        return msg, 'fallback', 0.0, {}, context
    
    try:
        predicted = clf.predict([user_input])[0]
        proba = clf.predict_proba([user_input])[0]
        confidence = max(proba)
        
        missing = check_missing(predicted, current_entities)
        
        if missing:
            context['current_intent'] = predicted
            context['waiting_for'] = missing
            context['entities'] = current_entities.copy()
            return get_followup(predicted, missing), predicted, confidence, current_entities, context
        else:
            response = get_response(predicted, user_input)
            return response, predicted, confidence, current_entities, context
            
    except Exception as e:
        return f"Error processing: {str(e)}", 'error', 0.0, {}, context

# =============================================
# SESSION STATE
# =============================================

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'username' not in st.session_state:
    st.session_state.username = ""

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'context' not in st.session_state:
    st.session_state.context = {}

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# =============================================
# LOGIN PAGE
# =============================================

if not st.session_state.logged_in:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if logo_base64:
            st.image(f"data:image/png;base64,{logo_base64}", width=200)
        
        st.markdown("# 🏦 Finexa Bank")
        st.markdown("### Welcome! Please login to continue")
        st.markdown("---")
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            
            st.markdown("")
            col_a, col_b = st.columns(2)
            
            with col_a:
                submit = st.form_submit_button("🔐 Login", use_container_width=True)
            with col_b:
                demo = st.form_submit_button("⚡ Demo Login", use_container_width=True)
            
            if submit:
                if username and password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("✅ Login Successful!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Please enter both username and password")
            
            if demo:
                st.session_state.logged_in = True
                st.session_state.username = "Demo User"
                st.success("✅ Welcome! Redirecting...")
                time.sleep(0.5)
                st.rerun()
        
        st.markdown("---")
        st.info("💡 **Tip:** Click 'Demo Login' for quick access")
    
    st.stop()

# =============================================
# MAIN INTERFACE (After Login)
# =============================================

# Navbar
logo_html = f'<img src="data:image/png;base64,{logo_base64}" alt="Finexa Bank">' if logo_base64 else '<span class="logo-text">FINEXA BANK</span>'

st.markdown(f"""
<div class="navbar">
    <div class="logo">
        {logo_html}
    </div>
</div>
""", unsafe_allow_html=True)

# Welcome & Logout
col_welcome, col_logout = st.columns([4, 1])
with col_welcome:
    st.success(f"👤 Welcome, **{st.session_state.username}**!")
with col_logout:
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.messages = []
        st.session_state.context = {}
        st.rerun()

# Navigation
nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

with nav_col1:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = 'home'
        st.rerun()

with nav_col2:
    if st.button("⚡ Services", use_container_width=True):
        st.session_state.current_page = 'services'
        st.rerun()

with nav_col3:
    if st.button("💰 Loans", use_container_width=True):
        st.session_state.current_page = 'loans'
        st.rerun()

with nav_col4:
    if st.button("📞 Contact", use_container_width=True):
        st.session_state.current_page = 'contact'
        st.rerun()

st.markdown("---")

# =============================================
# PAGES
# =============================================

if st.session_state.current_page == 'home':
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("## 🏦 Banking Services")
        st.markdown("---")
        st.markdown("""
        <div class="feature-card">
            <h3>💳 Card Services</h3>
            <p>Get premium credit and debit cards with exclusive rewards, cashback offers, and global acceptance. Enjoy contactless payments, EMI options, and 24/7 security.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📱 Digital Banking</h3>
            <p>Experience seamless banking with our Net Banking portal, Mobile App, and UPI services. Transfer money, pay bills, and manage accounts anytime, anywhere.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>💰 Account Services</h3>
            <p>Open Savings, Current, or Fixed Deposit accounts with attractive interest rates. Enjoy zero-balance accounts and personalized banking solutions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🏠 Loan Products</h3>
            <p>Apply for Home, Personal, Car, or Education Loans with competitive rates starting from 8.5% p.a. Quick approval and flexible EMI options available.</p>
        </div>
        """, unsafe_allow_html=True)

        st.info("📞 **24/7 Support:** 1800-123-4567 | 📧 **Email:** support@finexabank.com")
    
    with col2:
        st.markdown("### 🤖 BankBot AI Assistant")
        st.markdown('<span class="status-dot"></span>**Online & Ready**', unsafe_allow_html=True)
        st.markdown("---")
        
        if error:
            st.error(f"⚠️ {error}")
            st.info("💡 Make sure you've run `python milestone1_updated.py` to train the model first!")
        
        chat_container = st.container(height=450)
        
        with chat_container:
            if len(st.session_state.messages) == 0:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(f"👋 Hello **{st.session_state.username}**! How can I assist you with your banking needs today?")
                    st.markdown('<span class="intent-badge">Greet</span>', unsafe_allow_html=True)
            
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                    st.write(msg["content"])
                    
                    if msg["role"] == "assistant":
                        if "intent" in msg:
                            badge_class = "fallback-badge" if msg["intent"] in ['fallback', 'error'] else "intent-badge"
                            st.markdown(f'<span class="{badge_class}">{msg["intent"]}</span>', unsafe_allow_html=True)
                        
                        if "entities" in msg and msg["entities"]:
                            entity_str = ", ".join([f"**{k}**: {v}" for k, v in msg["entities"].items()])
                            st.caption(f"📋 {entity_str}")
                    
                    if "timestamp" in msg:
                        st.caption(f"🕐 {msg['timestamp']}")
        
        if prompt := st.chat_input("Ask about banking services..."):
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
            
            with st.spinner("Processing..."):
                time.sleep(0.3)
                response, intent, confidence, entities, updated_context = process_message(prompt, st.session_state.context)
                st.session_state.context = updated_context
            
            st.session_state.messages.append({
                "role": "assistant", "content": response, "intent": intent,
                "confidence": confidence, "entities": entities, "timestamp": timestamp
            })
            st.rerun()

elif st.session_state.current_page == 'services':
    st.title("⚡ Our Services")
    st.markdown("---")
    
    st.header("💳 Card Services")
    st.write("- **Credit Cards:** Up to 5% cashback and reward points")
    st.write("- **Debit Cards:** Free ATM withdrawals nationwide")
    st.write("- **Contactless Payments:** Secure tap-and-pay technology")
    st.markdown("")
    
    st.header("📱 Digital Banking")
    st.write("- **Net Banking:** Manage accounts online 24/7")
    st.write("- **Mobile App:** Bank on-the-go securely")
    st.write("- **UPI Payments:** Instant money transfers")
    st.markdown("")
    
    st.header("💰 Account Services")
    st.write("- **Savings Account:** Up to 4% interest")
    st.write("- **Current Account:** Unlimited transactions")
    st.write("- **Fixed Deposits:** Up to 7.5% returns")

elif st.session_state.current_page == 'loans':
    st.title("💰 Loan Products")
    st.markdown("---")
    
    st.header("🏠 Home Loan")
    st.write("- Interest rates from 8.5% p.a.")
    st.write("- Loan amount up to ₹5 Crores")
    st.write("- Tenure up to 30 years")
    st.markdown("")
    
    st.header("👤 Personal Loan")
    st.write("- Up to ₹50 Lakhs")
    st.write("- Interest from 10.5% p.a.")
    st.write("- Quick approval")
    st.markdown("")
    
    st.header("🚗 Car Loan")
    st.write("- Finance up to 90% of car value")
    st.write("- Interest from 9% p.a.")
    st.write("- Tenure up to 7 years")

elif st.session_state.current_page == 'contact':
    st.title("📞 Contact Us")
    st.markdown("---")
    
    st.header("🏢 Head Office")
    st.write("Finexa Bank Corporate Office")
    st.write("123 Financial District")
    st.write("Mumbai - 400001, India")
    st.markdown("")
    
    st.header("📱 Customer Support")
    st.write("- **Toll-Free:** 1800-123-4567")
    st.write("- **Email:** support@finexabank.com")
    st.write("- **WhatsApp:** +91-98765-43210")

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #6b7280;">
    <p>🔒 Secure Banking | 🤖 AI-Powered | © 2024 Finexa Bank</p>
</div>
""", unsafe_allow_html=True)