# milestone1_updated.py

import pandas as pd
import json
import spacy 
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

print("=" * 70)
print("  MILESTONE 1: INTENT CLASSIFICATION & ENTITY EXTRACTION")
print("=" * 70)

# ======================
# Load & Prepare Data
# ======================
print("\n[INFO] Loading dataset...")

try:
    # Load the new expanded dataset
    data = pd.read_csv(
        "bankbot_expanded_dataset.csv",
        encoding='utf-8',
        on_bad_lines='skip',
        engine='python'
    )
    print(f"[SUCCESS] Dataset loaded: {len(data)} records")
except Exception as e:
    print(f"[ERROR] Failed with utf-8, trying with latin-1 encoding...")
    try:
        data = pd.read_csv(
            "bankbot_expanded_dataset.csv",
            encoding='latin-1',
            on_bad_lines='skip',
            engine='python'
        )
        print(f"[SUCCESS] Dataset loaded: {len(data)} records")
    except Exception as e2:
        print(f"[ERROR] Could not load CSV: {e2}")
        raise

# Display basic info about the dataset
print(f"\n[INFO] Dataset columns: {data.columns.tolist()}")
print(f"[INFO] Dataset shape: {data.shape}")

# Check for missing values
print("\n[INFO] Missing values per column:")
for col in data.columns:
    missing = data[col].isnull().sum()
    if missing > 0:
        print(f"  - {col}: {missing} ({missing/len(data)*100:.2f}%)")

# Handle missing values
print("\n[INFO] Handling missing values...")
data = data.dropna(subset=['text', 'intent'])  # Must have text and intent
data['response'] = data['response'].fillna('')  # Fill empty responses
data['entities'] = data['entities'].fillna('[]')  # Fill empty entities
print(f"[SUCCESS] Clean dataset: {len(data)} records")

# Safe conversion of entities from string to list/dict
def parse_entities(entity_str):
    """
    Parse entities from string format to Python object.
    Handles: JSON arrays, JSON objects, empty strings, etc.
    """
    try:
        if pd.notnull(entity_str) and isinstance(entity_str, str):
            entity_str = entity_str.strip()
            
            # Empty or "[]"
            if entity_str == "" or entity_str == "[]" or entity_str == "{}":
                return []
            
            # Try parsing as JSON
            parsed = json.loads(entity_str)
            
            # Convert to list format if it's a dict
            if isinstance(parsed, dict):
                return [{"entity": k, "value": v} for k, v in parsed.items()]
            elif isinstance(parsed, list):
                return parsed
            
        return []
    except json.JSONDecodeError:
        # If JSON parsing fails, try custom format parsing
        try:
            # Handle format like: PERSON:John|MONEY:500
            entities = []
            for pair in entity_str.split("|"):
                if ":" in pair:
                    key, val = pair.split(":", 1)
                    entities.append({"entity": key.strip(), "value": val.strip()})
            return entities if entities else []
        except:
            return []
    except Exception:
        return []

print("\n[INFO] Processing entities...")
data["parsed_entities"] = data["entities"].apply(parse_entities)
print("[SUCCESS] Entities processed")

# Display dataset info
print("\n[INFO] Dataset Overview:")
print(f"  - Total samples: {len(data)}")
print(f"  - Unique intents: {data['intent'].nunique()}")

# Show sample of data
print("\n[INFO] Sample records:")
print(data[['text', 'intent', 'response']].head(3))

print(f"\n[INFO] Complete Intent Distribution:")

# Check class distribution
intent_counts = data['intent'].value_counts()
print(f"\n  Top 20 most common intents:")
for intent, count in intent_counts.head(20).items():
    percentage = (count / len(data)) * 100
    print(f"    * {intent}: {count} ({percentage:.2f}%)")

# Show total number of intents
total_intents = len(intent_counts)
print(f"\n  Total unique intents: {total_intents}")

# Identify class imbalance issues
max_count = intent_counts.max()
min_count = intent_counts.min()
imbalance_ratio = max_count / min_count

print(f"\n[INFO] Class Balance Analysis:")
print(f"  - Most common intent: {intent_counts.index[0]} ({max_count} samples)")
print(f"  - Least common intent: {intent_counts.index[-1]} ({min_count} samples)")
print(f"  - Imbalance ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 5:
    print(f"  [WARNING] High class imbalance detected!")
    print(f"  [INFO] Using class_weight='balanced' to handle this")

# Check for classes with only 1 sample
single_sample_classes = intent_counts[intent_counts < 2].index.tolist()
if single_sample_classes:
    print(f"\n[WARNING] Found {len(single_sample_classes)} intent(s) with only 1 sample:")
    for intent in single_sample_classes[:5]:  # Show first 5
        print(f"    * {intent}")
    if len(single_sample_classes) > 5:
        print(f"    ... and {len(single_sample_classes) - 5} more")
    print("[INFO] These will be handled specially in train-test split")

# Split into features & labels
X = data["text"]
y = data["intent"]

# Use stratify only if all classes have at least 2 samples
if len(single_sample_classes) == 0:
    print("\n[INFO] Using stratified split (all classes have 2+ samples)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    print("\n[INFO] Using regular split (some classes have only 1 sample)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"\n[SUCCESS] Train-Test Split:")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Testing samples: {len(X_test)}")

# ======================
# Build Intent Classifier
# ======================
print("\n[INFO] Training Intent Classifier...")
print("[INFO] This may take a few minutes with 30K+ samples...")

# Use balanced class weights to handle class imbalance
clf = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,  # Increased for larger dataset
        ngram_range=(1, 3),
        min_df=2,  # Ignore very rare terms
        max_df=0.95,  # Ignore very common terms
        sublinear_tf=True,
        token_pattern=r'\b\w+\b'
    )),
    ("logreg", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
        C=1.0,
        solver='saga',  # Better for large datasets
        n_jobs=-1  # Use all CPU cores
    ))
])

clf.fit(X_train, y_train)
print("[SUCCESS] Model training completed!")

# Evaluate
print("\n[INFO] Evaluating model on test set...")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("  MODEL EVALUATION RESULTS")
print("=" * 70)
print(f"\n[RESULT] Overall Accuracy: {round(accuracy * 100, 2)}%\n")

# Show detailed report for top intents only (to avoid overwhelming output)
print("[INFO] Classification Report (Top 10 intents):")
top_10_intents = intent_counts.head(10).index.tolist()
print(classification_report(y_test, y_pred, labels=top_10_intents, zero_division=0))

# Overall metrics
print("\n[INFO] Overall Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save the model
print("\n[INFO] Saving model...")
joblib.dump(clf, "intent_pipeline.joblib")
print("[SUCCESS] Model saved as 'intent_pipeline.joblib'")

# Save intent mapping for reference
intent_mapping = {
    "intents": intent_counts.to_dict(),
    "total_samples": len(data),
    "unique_intents": len(intent_counts),
    "model_accuracy": float(accuracy)
}
with open("intent_mapping.json", "w") as f:
    json.dump(intent_mapping, f, indent=2)
print("[SUCCESS] Intent mapping saved as 'intent_mapping.json'")

# ======================
# Enhanced Entity Extraction
# ======================
print("\n[INFO] Loading spaCy model for entity extraction...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("[SUCCESS] spaCy model loaded")
except:
    print("[WARNING] spaCy model not found. Installing...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("[SUCCESS] spaCy model loaded")

def extract_entities(text):
    """
    Enhanced entity extraction using spaCy and custom regex rules.
    Extracts: account numbers, card numbers, transaction IDs, pincodes,
    mobile numbers, emails, UPI IDs, money amounts, names, dates, etc.
    """
    doc = nlp(text)
    entities = {}
    
    # SpaCy NER - Extract common entities
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
        elif ent.label_ == "GPE":  # Geo-political entity
            entities["location"] = ent.text
    
    # Account number (10-16 digits)
    account_pattern = r'\b\d{10,16}\b'
    account_matches = re.findall(account_pattern, text)
    if account_matches:
        # Filter out card numbers (16 digits)
        for match in account_matches:
            if len(match) <= 12:
                entities["account_number"] = match
                break
    
    # Card number (16 digits, may have spaces/hyphens)
    card_pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    card_match = re.search(card_pattern, text)
    if card_match:
        card_num = re.sub(r'[\s-]', '', card_match.group())
        entities["card_number"] = card_num
    
    # Transaction ID (various formats)
    txn_patterns = [
        r'\b(TXN|txn|TRANS|trans)\w+\b',
        r'\bTXID\d+\b',
        r'\b[A-Z]{3}\d{6,}\b'  # Like ABC123456
    ]
    for pattern in txn_patterns:
        txn_match = re.search(pattern, text, re.IGNORECASE)
        if txn_match:
            entities["transaction_id"] = txn_match.group().upper()
            break
    
    # Pincode (6 digits)
    pincode_pattern = r'\b\d{6}\b'
    if any(word in text.lower() for word in ['pincode', 'pin code', 'postal', 'near', 'location', 'atm', 'branch']):
        pincode_match = re.search(pincode_pattern, text)
        if pincode_match and pincode_match.group() not in entities.values():
            entities["pincode"] = pincode_match.group()
    
    # Mobile number (10 digits starting with 6-9)
    mobile_pattern = r'\b[6-9]\d{9}\b'
    if any(word in text.lower() for word in ['mobile', 'phone', 'number', 'contact']):
        mobile_match = re.search(mobile_pattern, text)
        if mobile_match:
            entities["mobile_number"] = mobile_match.group()
    
    # Email address
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, text)
    if email_match:
        entities["email"] = email_match.group()
    
    # UPI ID (format: username@provider)
    upi_pattern = r'\b[\w.-]+@[a-z]+\b'
    if '@' in text and "email" not in entities:
        upi_match = re.search(upi_pattern, text, re.IGNORECASE)
        if upi_match:
            upi_id = upi_match.group().lower()
            provider = upi_id.split('@')[1]
            if '.' not in provider:  # UPI doesn't have dots in provider
                entities["upi_id"] = upi_id
    
    # Money amount (with context)
    if any(word in text.lower() for word in ['send', 'transfer', 'deposit', 'withdraw', 'pay', 'amount', 'rupees', 'rs']):
        money_patterns = [
            r'(?:rs\.?|₹)\s*(\d+(?:,\d+)*(?:\.\d+)?)',  # Rs. 1000 or ₹1000
            r'\b(\d{3,}(?:,\d+)*(?:\.\d+)?)\s*(?:rupees|rs|inr)\b',  # 1000 rupees
            r'\b(\d{3,})\b'  # Standalone number (3+ digits)
        ]
        for pattern in money_patterns:
            money_match = re.search(pattern, text.lower())
            if money_match:
                amount = re.sub(r'[,]', '', money_match.group(1) if len(money_match.groups()) > 0 else money_match.group())
                if amount and "money" not in entities:
                    entities["money"] = amount
                    entities["amount"] = amount
                break
    
    # Account types
    account_types = ["savings", "current", "salary", "nri", "senior citizen", "basic"]
    for acc_type in account_types:
        if acc_type in text.lower():
            entities["account_type"] = acc_type
            break
    
    # Loan types
    loan_types = ["personal loan", "home loan", "car loan", "education loan", 
                  "business loan", "gold loan", "vehicle loan", "mortgage"]
    for loan_type in loan_types:
        if loan_type in text.lower():
            entities["loan_type"] = loan_type
            break
    
    # Card types
    card_types = ["credit card", "debit card", "prepaid card", "atm card"]
    for card_type in card_types:
        if card_type in text.lower():
            entities["card_type"] = card_type
            break
    
    # Deposit types
    if "fixed deposit" in text.lower() or "fd" in text.lower():
        entities["deposit_type"] = "fixed deposit"
    elif "recurring deposit" in text.lower() or "rd" in text.lower():
        entities["deposit_type"] = "recurring deposit"
    elif "savings deposit" in text.lower():
        entities["deposit_type"] = "savings deposit"
    
    # Tenure (for FD/RD)
    tenure_pattern = r'\b(\d+)\s*(month|months|year|years|day|days)\b'
    tenure_match = re.search(tenure_pattern, text.lower())
    if tenure_match:
        entities["tenure"] = tenure_match.group()
    
    # IFSC Code
    ifsc_pattern = r'\b[A-Z]{4}0[A-Z0-9]{6}\b'
    ifsc_match = re.search(ifsc_pattern, text)
    if ifsc_match:
        entities["ifsc_code"] = ifsc_match.group()
    
    # Pan Card
    pan_pattern = r'\b[A-Z]{5}\d{4}[A-Z]\b'
    if 'pan' in text.lower():
        pan_match = re.search(pan_pattern, text)
        if pan_match:
            entities["pan_number"] = pan_match.group()
    
    return entities


# ======================
# Test Entity Extraction
# ======================
print("\n" + "=" * 70)
print("  TESTING ENTITY EXTRACTION")
print("=" * 70)

test_cases = [
    "hello",
    "what is my account balance",
    "transfer 5000 to John",
    "check status of transaction TXN12345",
    "find ATM near 700120",
    "block my card 1234567890123456",
    "update mobile 9876543210",
    "create FD of Rs. 50000 for 12 months",
    "what is the interest rate for home loan",
    "I want to apply for a credit card",
    "thank you",
    "goodbye"
]

print("\n[INFO] Running test cases:\n")
for i, test in enumerate(test_cases, 1):
    intent = clf.predict([test])[0]
    entities = extract_entities(test)
    proba = clf.predict_proba([test])[0]
    confidence = max(proba) * 100
    
    print(f"Test {i}:")
    print(f"  Query: {test}")
    print(f"  Intent: {intent} ({confidence:.2f}%)")
    print(f"  Entities: {entities}")
    print()

# ======================
# Interactive Testing
# ======================
print("\n" + "=" * 70)
print("  INTERACTIVE TESTING MODE")
print("=" * 70)
print("\nType your query to test intent and entity extraction.")
print("Type 'exit' to quit.\n")

while True:
    try:
        user_input = input("Your Query: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\n✓ Testing completed. Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Predict intent
        predicted_intent = clf.predict([user_input])[0]
        
        # Extract entities
        extracted_entities = extract_entities(user_input)
        
        # Get confidence scores
        probabilities = clf.predict_proba([user_input])[0]
        max_prob = max(probabilities)
        
        # Get top 3 predictions
        top_indices = probabilities.argsort()[-3:][::-1]
        top_intents = clf.classes_[top_indices]
        top_probs = probabilities[top_indices]
        
        print(f"\n  >> Predicted Intent: {predicted_intent}")
        print(f"  >> Confidence: {round(max_prob * 100, 2)}%")
        print(f"\n  >> Top 3 predictions:")
        for intent, prob in zip(top_intents, top_probs):
            print(f"     - {intent}: {round(prob * 100, 2)}%")
        print(f"\n  >> Entities: {extracted_entities}")
        print()
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Testing interrupted. Goodbye!")
        break
    except Exception as e:
        print(f"\n  [ERROR] Error: {e}\n")

print("\n" + "=" * 70)
print("  MILESTONE 1 COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\n[SUCCESS] Model trained and saved")
print("[SUCCESS] Entity extraction implemented")
print("[SUCCESS] Ready for Milestone 2 - Dialogue Flow\n")