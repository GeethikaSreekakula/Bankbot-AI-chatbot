# milestone2_updated.py - FIXED with greeting detection

import pandas as pd
import json
import re
import joblib
import spacy
import random

print("=" * 70)
print("  MILESTONE 2: DIALOGUE FLOW SYSTEM")
print("=" * 70)

# ======================
# Load Resources (Silent mode - only show errors)
# ======================

try:
    clf = joblib.load("intent_pipeline.joblib")
except:
    print("\n[ERROR] Model not found! Run milestone1_updated.py first")
    exit()

try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("\n[ERROR] Run: python -m spacy download en_core_web_sm")
    exit()

try:
    data = pd.read_csv("bankbot_expanded_dataset.csv", on_bad_lines='skip', encoding='utf-8')
except:
    print("\n[ERROR] Dataset 'bankbot_expanded_dataset.csv' not found!")
    exit()

print()  # Blank line for spacing

# Handle missing values
data = data.dropna(subset=['text', 'intent'])
data['response'] = data['response'].fillna('')
data['entities'] = data['entities'].fillna('[]')

# Build intent-response mapping
intent_responses = {}
intent_text_map = {}  # Map text to response for exact matching

for _, row in data.iterrows():
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
    
    # Map exact text to response for better matching
    intent_text_map[intent][text] = response

# Check what intents we have
print(f"[INFO] Loaded {len(intent_responses)} intents with responses")
if 'greet' in intent_responses:
    print(f"[INFO] ✓ 'greet' intent found with {len(intent_responses['greet'])} responses")
if 'thanks' in intent_responses:
    print(f"[INFO] ✓ 'thanks' intent found with {len(intent_responses['thanks'])} responses")
if 'goodbye' in intent_responses:
    print(f"[INFO] ✓ 'goodbye' intent found with {len(intent_responses['goodbye'])} responses")
print()

# ======================
# Entity Extraction (Enhanced for new dataset)
# ======================

def extract_entities(text):
    """Extract entities from text - enhanced for larger dataset."""
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
        elif ent.label_ == "ORG":
            entities["organization"] = ent.text
        elif ent.label_ == "GPE":
            entities["location"] = ent.text
    
    # Account number (10-16 digits)
    account_pattern = r'\b\d{10,16}\b'
    account_matches = re.findall(account_pattern, text)
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
    
    # Email
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        entities["email"] = email_match.group()
    
    # UPI ID
    if '@' in text and 'email' not in entities:
        upi_match = re.search(r'\b[\w.-]+@[a-z]+\b', text, re.IGNORECASE)
        if upi_match:
            upi_id = upi_match.group().lower()
            provider = upi_id.split('@')[1]
            if '.' not in provider:
                entities["upi_id"] = upi_id
    
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

# ======================
# Required Entities (Extended for larger dataset)
# ======================

REQUIRED_ENTITIES = {
    'check_balance': ['account_number'],
    'balance_enquiry': ['account_number'],
    'account_balance': ['account_number'],
    'mini_statement': ['account_number'],
    'transaction_status': ['transaction_id'],
    'transaction_enquiry': ['transaction_id'],
    'transaction_inquiry': ['transaction_id'],
    'atm_locator': ['pincode'],
    'find_atm': ['pincode'],
    'branch_locator': ['pincode'],
    'find_branch': ['pincode'],
    'transfer_money': ['person', 'money'],
    'money_transfer': ['person', 'money'],
    'send_money': ['person', 'money'],
    'update_mobile': ['account_number', 'mobile_number'],
    'change_mobile': ['account_number', 'mobile_number'],
    'update_email': ['account_number', 'email'],
    'change_email': ['account_number', 'email'],
    'card_block': ['card_number'],
    'block_card': ['card_number'],
    'debit_card_block': ['card_number'],
    'credit_card_block': ['card_number'],
    'card_lost': ['card_number'],
    'card_replace': ['card_number'],
    'replace_card': ['card_number'],
    'close_account': ['account_number'],
    'account_closure': ['account_number'],
    'fd_creation': ['amount', 'tenure'],
    'fixed_deposit': ['amount', 'tenure'],
    'open_fd': ['amount', 'tenure'],
    'rd_creation': ['amount', 'tenure'],
    'recurring_deposit': ['amount', 'tenure'],
    'open_rd': ['amount', 'tenure']
}

FOLLOWUP_QUESTIONS = {
    'check_balance': {'account_number': "Please provide your account number to check balance."},
    'transaction_inquiry': {'transaction_id': "Please provide the transaction ID."},
    'atm_locator': {'pincode': "Please share your pincode to find ATMs."},
    'branch_locator': {'pincode': "Please provide your pincode to find branches."},
    'transfer_money': {
        'person': "To whom would you like to send money?",
        'money': "How much would you like to transfer?"
    },
    'block_card': {'card_number': "Please provide your card number to block it."},
}

# ======================
# ChatBot Class
# ======================

class BankBot:
    def __init__(self, debug=False):
        self.clf = clf
        self.intent_responses = intent_responses
        self.intent_text_map = intent_text_map
        self.current_intent = None
        self.waiting_for = None
        self.entities = {}
        self.active = True
        self.debug = debug
        
        # Extended banking keywords
        self.banking_keywords = [
            'account', 'balance', 'bank', 'banking', 'statement', 'mini',
            'savings', 'current', 'salary', 'loan', 'loans', 'personal', 'home', 
            'car', 'education', 'emi', 'interest', 'rate', 'card', 'cards', 
            'debit', 'credit', 'atm', 'pin', 'block', 'lost', 'stolen',
            'transfer', 'money', 'send', 'payment', 'pay', 'transaction',
            'withdraw', 'deposit', 'cash', 'cheque', 'upi', 'netbanking',
            'fd', 'rd', 'fixed', 'recurring', 'branch', 'near', 'location',
            'pincode', 'ifsc', 'pan', 'kyc', 'nominee'
        ]
        
        # Fallback messages
        self.fallback_messages = [
            "I'm a banking assistant. I can only help with banking-related queries. How can I assist you with your banking needs?",
            "I'm here to help with banking services only. Please ask me about accounts, loans, cards, or other banking services.",
            "I specialize in banking assistance. What banking service do you need help with today?"
        ]
    
    def debug_print(self, message):
        """Print debug messages only if debug mode is on."""
        if self.debug:
            print(message)
    
    def is_banking_related(self, text):
        """Check if query is banking-related."""
        text_lower = text.lower()
        words = text_lower.split()
        
        for keyword in self.banking_keywords:
            if keyword in words or (len(keyword) > 3 and keyword in text_lower):
                return True
        return False
    
    def get_response(self, intent, user_text=None):
        """Get response from dataset with smart matching."""
        
        # Try exact match first
        if user_text and intent in self.intent_text_map:
            user_text_lower = user_text.strip().lower()
            if user_text_lower in self.intent_text_map[intent]:
                return self.intent_text_map[intent][user_text_lower]
        
        # Default: pick FIRST response (consistent behavior)
        if intent in self.intent_responses and self.intent_responses[intent]:
            return self.intent_responses[intent][0]  # Changed from random.choice()
        
        # Fallback for common intents
        fallback_responses = {
            'greet': "Hello! How can I help you with your banking needs today?",
            'thanks': "You're welcome! Is there anything else I can help you with?",
            'goodbye': "Thank you for contacting us. Have a great day!"
        }
        
        return fallback_responses.get(intent, "I can help you with that. Please provide more details.")
    
    def get_followup(self, intent, entity):
        """Get follow-up question."""
        if intent in FOLLOWUP_QUESTIONS and entity in FOLLOWUP_QUESTIONS[intent]:
            return FOLLOWUP_QUESTIONS[intent][entity]
        return f"Please provide {entity.replace('_', ' ')}."
    
    def check_missing(self, intent, entities):
        """Check missing entities."""
        if intent not in REQUIRED_ENTITIES:
            return None
        for req in REQUIRED_ENTITIES[intent]:
            if req not in entities:
                return req
        return None
    
    def detect_intent_manually(self, text):
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
        
        # Single word queries that need general info
        general_info_keywords = {
            'loan': 'loan_info',
            'loans': 'loan_info',
            'card': 'general_banking_info',
            'cards': 'general_banking_info',
            'account': 'general_banking_info',
            'balance': 'check_balance',
            'transfer': 'transfer_money',
            'atm': 'atm_locator',
            'branch': 'branch_locator'
        }
        
        if text_lower in general_info_keywords:
            detected_intent = general_info_keywords[text_lower]
            # Return a special marker to show general info
            return ('general_query', text_lower)
        
        return None
    
    def process(self, user_input):
        """Process user input."""
        
        user_lower = user_input.lower().strip()
        
        self.debug_print(f"  [DEBUG] Processing: '{user_input}'")
        
        # Manual detection for common phrases (HIGHEST PRIORITY)
        manual_intent = self.detect_intent_manually(user_input)
        if manual_intent:
            # Check if it's a tuple (general query)
            if isinstance(manual_intent, tuple) and manual_intent[0] == 'general_query':
                keyword = manual_intent[1]
                responses = {
                    'loan': "I can help you with loans! We offer personal loans, home loans, car loans, and education loans. What type of loan are you interested in?",
                    'loans': "I can help you with loans! We offer personal loans, home loans, car loans, and education loans. What type of loan are you interested in?",
                    'card': "I can help you with cards! We offer debit cards and credit cards. Would you like information about card types, applying for a card, or managing an existing card?",
                    'cards': "I can help you with cards! We offer debit cards and credit cards. Would you like information about card types, applying for a card, or managing an existing card?",
                    'account': "I can help you with your account! Would you like to check your balance, open a new account, or update account details?",
                    'balance': "I can help you check your account balance. Please provide your account number.",
                    'transfer': "I can help you transfer money. Who would you like to send money to and how much?",
                    'atm': "I can help you find ATMs near you. Please share your pincode or location.",
                    'branch': "I can help you find our bank branches. Please share your pincode or location."
                }
                return responses.get(keyword, "How can I assist you with that?"), 'info_' + keyword
            
            self.debug_print(f"  [DEBUG] Manual detection: {manual_intent}")
            if manual_intent == 'goodbye':
                self.active = False
            return self.get_response(manual_intent, user_input), manual_intent
        
        # Extract entities
        current_entities = extract_entities(user_input)
        
        # If waiting for entity
        if self.waiting_for:
            self.debug_print(f"  [DEBUG] Waiting for: {self.waiting_for}")
            
            if self.waiting_for in current_entities:
                self.entities.update(current_entities)
                self.waiting_for = None
                
                missing = self.check_missing(self.current_intent, self.entities)
                if missing:
                    self.waiting_for = missing
                    return self.get_followup(self.current_intent, missing), self.current_intent
                else:
                    response = self.get_response(self.current_intent, user_input)
                    intent = self.current_intent
                    self.current_intent = None
                    self.entities = {}
                    return response, intent
            else:
                if self.is_banking_related(user_input) and not any(w in user_lower for w in ['yes', 'no', 'cancel']):
                    self.debug_print(f"  [DEBUG] New query detected while waiting - resetting context")
                    self.waiting_for = None
                    self.current_intent = None
                    self.entities = {}
                else:
                    return f"I didn't get that. {self.get_followup(self.current_intent, self.waiting_for)}", self.current_intent
        
        # Check if banking-related
        is_banking = self.is_banking_related(user_input)
        if not is_banking:
            return random.choice(self.fallback_messages), 'fallback'
        
        # Predict intent
        predicted = self.clf.predict([user_input])[0]
        proba = self.clf.predict_proba([user_input])[0]
        confidence = max(proba) * 100
        
        self.debug_print(f"  [DEBUG] Predicted intent: {predicted} (confidence: {confidence:.1f}%)")
        
        self.current_intent = predicted
        self.entities = current_entities.copy()
        
        # Check missing entities
        missing = self.check_missing(predicted, current_entities)
        
        if missing:
            self.waiting_for = missing
            return self.get_followup(predicted, missing), predicted
        else:
            response = self.get_response(predicted, user_input)
            self.current_intent = None
            self.entities = {}
            return response, predicted

# ======================
# Main Chat
# ======================

def run_chatbot():
    """Run chatbot."""
    
    print("=" * 70)
    print("        BANKBOT - Your Banking Assistant")
    print("=" * 70)
    print("\nI can help with:")
    print("  * Balance & statements")
    print("  * Money transfers")
    print("  * Loans & cards")
    print("  * ATM & branch locations")
    print("  * Account updates")
    print("  * FD/RD creation")
    print("  * UPI & Netbanking issues")
    print("\nType 'bye' to exit")
    print("=" * 70)
    
    # Welcome message
    print("\nBankBot [greet]: Hello! How can I help you with your banking needs today?\n")
    
    # Set debug=True to see debug messages, debug=False to hide them
    bot = BankBot(debug=False)  # Change to True if you need debugging
    
    while bot.active:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            response, intent = bot.process(user_input)
            print(f"BankBot [{intent}]: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("\n" + "=" * 70)
    print("  Thank you for using BankBot!")
    print("=" * 70)

# ======================
# Run
# ======================

if __name__ == "__main__":
    run_chatbot()