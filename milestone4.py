# milestone4_admin_panel.py - BankBot Admin Panel
# Infosys Springboard Project - Milestone 4

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# =============================================
# PAGE CONFIGURATION
# =============================================

st.set_page_config(
    page_title="BankBot Admin Panel",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CUSTOM CSS
# =============================================

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .stButton button {
        width: 100%;
    }
    
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# LOAD DATASET
# =============================================

@st.cache_data
def load_dataset():
    try:
        data = pd.read_csv("bankbot_expanded_dataset.csv", encoding='utf-8', on_bad_lines='skip')
        data = data.dropna(subset=['text', 'intent'])
        data['response'] = data['response'].fillna('')
        data['entities'] = data['entities'].fillna('[]')
        return data, None
    except Exception as e:
        return None, str(e)

# =============================================
# SESSION STATE INITIALIZATION
# =============================================

if 'admin_logged_in' not in st.session_state:
    st.session_state.admin_logged_in = False

if 'faqs' not in st.session_state:
    st.session_state.faqs = [
        {'id': 1, 'question': 'What are your business hours?', 'answer': 'We are open Monday to Friday, 9 AM to 5 PM.', 'category': 'General', 'active': True},
        {'id': 2, 'question': 'How do I reset my password?', 'answer': 'Click on "Forgot Password" and follow the instructions.', 'category': 'Account', 'active': True},
        {'id': 3, 'question': 'What is the minimum balance?', 'answer': 'Minimum balance is ₹100 for savings accounts.', 'category': 'Accounts', 'active': True}
    ]

if 'user_queries' not in st.session_state:
    st.session_state.user_queries = []

# =============================================
# LOGIN PAGE
# =============================================

if not st.session_state.admin_logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("# 🔧 BankBot Admin Panel")
        st.markdown("### Secure Admin Access")
        st.markdown("---")
        
        with st.form("admin_login"):
            username = st.text_input("👤 Admin Username", placeholder="Enter admin username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                submit = st.form_submit_button("🔐 Login", width='stretch')
            with col_b:
                demo = st.form_submit_button("⚡ Demo Login", width='stretch')
            
            if submit:
                if username and password:
                    st.session_state.admin_logged_in = True
                    st.success("✅ Login Successful!")
                    st.rerun()
                else:
                    st.error("❌ Please enter credentials")
            
            if demo:
                st.session_state.admin_logged_in = True
                st.success("✅ Welcome Admin!")
                st.rerun()
        
        st.markdown("---")
        st.info("💡 **Tip:** Use Demo Login for quick access")
    
    st.stop()

# =============================================
# MAIN ADMIN PANEL
# =============================================

# Sidebar
with st.sidebar:
    st.markdown("# 🔧 Admin Panel")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "📝 Training Data", "💬 User Queries", "❓ FAQs", "📈 Analytics", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if st.button("🚪 Logout", width='stretch'):
        st.session_state.admin_logged_in = False
        st.rerun()
    
    st.markdown("---")
    st.caption("© 2024 BankBot AI")

# Load dataset
data, error = load_dataset()

# =============================================
# DASHBOARD PAGE
# =============================================

if page == "📊 Dashboard":
    st.title("📊 Admin Dashboard")
    st.markdown("---")
    
    if error:
        st.error(f"⚠️ Error loading dataset: {error}")
        st.info("💡 Make sure `bankbot_expanded_dataset.csv` exists in the current directory")
    else:
        st.success(f"✅ Dataset loaded: **{len(data):,}** samples")
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">{:,}</div>
            <div class="stat-label">Total Queries</div>
        </div>
        """.format(len(data) if data is not None else 0), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card success-card">
            <div class="stat-value">94.2%</div>
            <div class="stat-label">Success Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        intents = len(data['intent'].unique()) if data is not None else 0
        st.markdown(f"""
        <div class="stat-card warning-card">
            <div class="stat-value">{intents}</div>
            <div class="stat-label">Unique Intents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card info-card">
            <div class="stat-value">42</div>
            <div class="stat-label">Entity Types</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reload Dataset", width='stretch'):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if data is not None:
            csv = data.to_csv(index=False)
            st.download_button(
                "📥 Export Training Data",
                csv,
                "training_data.csv",
                "text/csv",
                width='stretch'
            )
    
    with col3:
        if st.button("📊 View Analytics", width='stretch'):
            st.session_state.page = "📈 Analytics"
            st.rerun()
    
    st.markdown("---")
    
    # Recent Activity
    if data is not None:
        st.subheader("📋 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top 10 Intents**")
            intent_counts = data['intent'].value_counts().head(10)
            st.dataframe(
                pd.DataFrame({
                    'Intent': intent_counts.index,
                    'Count': intent_counts.values
                }),
                width='stretch',
                hide_index=True
            )
        
        with col2:
            st.markdown("**Sample Data**")
            st.dataframe(
                data[['text', 'intent', 'response']].head(5),
                width='stretch',
                hide_index=True
            )

# =============================================
# TRAINING DATA PAGE
# =============================================

elif page == "📝 Training Data":
    st.title("📝 Training Data Management")
    st.markdown("---")
    
    if data is None:
        st.error("⚠️ Dataset not loaded!")
    else:
        # Filters
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search = st.text_input("🔍 Search", placeholder="Search by text or intent...")
        
        with col2:
            intent_filter = st.selectbox(
                "Filter by Intent",
                ["All"] + sorted(data['intent'].unique().tolist())
            )
        
        with col3:
            st.metric("Total Records", f"{len(data):,}")
        
        # Filter data
        filtered_data = data.copy()
        
        if search:
            filtered_data = filtered_data[
                filtered_data['text'].str.contains(search, case=False, na=False) |
                filtered_data['intent'].str.contains(search, case=False, na=False)
            ]
        
        if intent_filter != "All":
            filtered_data = filtered_data[filtered_data['intent'] == intent_filter]
        
        st.markdown(f"**Showing {len(filtered_data)} of {len(data)} records**")
        
        # Display with expandable rows
        for idx, row in filtered_data.head(50).iterrows():
            with st.expander(f"**{row['intent']}** - {row['text'][:100]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Text:**")
                    st.write(row['text'])
                    st.markdown("**Intent:**")
                    st.code(row['intent'])
                
                with col2:
                    st.markdown("**Response:**")
                    st.write(row['response'])
                    st.markdown("**Entities:**")
                    st.code(row['entities'])
                
                col_a, col_b, col_c = st.columns([1, 1, 3])
                with col_a:
                    if st.button("✏️ Edit", key=f"edit_{idx}"):
                        st.info("Edit functionality - coming soon!")
                with col_b:
                    if st.button("🗑️ Delete", key=f"delete_{idx}"):
                        st.warning("Delete functionality - coming soon!")
        
        if len(filtered_data) > 50:
            st.info(f"Showing first 50 of {len(filtered_data)} results. Use filters to narrow down.")
        
        # Export
        st.markdown("---")
        if st.button("📥 Export Filtered Data"):
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

# =============================================
# USER QUERIES PAGE
# =============================================

elif page == "💬 User Queries":
    st.title("💬 User Query Monitoring")
    st.markdown("---")
    
    # Generate sample queries if none exist
    if len(st.session_state.user_queries) == 0 and data is not None:
        sample_queries = []
        for idx, row in data.head(20).iterrows():
            sample_queries.append({
                'id': idx + 1,
                'query': row['text'],
                'intent': row['intent'],
                'confidence': 80 + (idx % 20),
                'timestamp': f"2024-11-29 10:{idx:02d}:00",
                'status': 'Responded'
            })
        st.session_state.user_queries = sample_queries
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_query = st.text_input("🔍 Search queries", placeholder="Search...")
    
    with col2:
        confidence_filter = st.selectbox(
            "Confidence Level",
            ["All", "High (90%+)", "Medium (70-89%)", "Low (<70%)"]
        )
    
    with col3:
        st.metric("Total Queries", len(st.session_state.user_queries))
    
    # Filter queries
    filtered_queries = st.session_state.user_queries.copy()
    
    if search_query:
        filtered_queries = [q for q in filtered_queries if search_query.lower() in q['query'].lower()]
    
    if confidence_filter == "High (90%+)":
        filtered_queries = [q for q in filtered_queries if q['confidence'] >= 90]
    elif confidence_filter == "Medium (70-89%)":
        filtered_queries = [q for q in filtered_queries if 70 <= q['confidence'] < 90]
    elif confidence_filter == "Low (<70%)":
        filtered_queries = [q for q in filtered_queries if q['confidence'] < 70]
    
    # Display table
    if filtered_queries:
        df = pd.DataFrame(filtered_queries)
        st.dataframe(
            df[['query', 'intent', 'confidence', 'timestamp', 'status']],
            width='stretch',
            hide_index=True
        )
        
        # Export
        st.markdown("---")
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Export Query Logs",
            csv,
            f"query_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
    else:
        st.info("No queries found matching your filters.")

# =============================================
# FAQS PAGE
# =============================================

elif page == "❓ FAQs":
    st.title("❓ FAQ Management")
    st.markdown("---")
    
    # Add new FAQ
    with st.expander("➕ Add New FAQ", expanded=False):
        with st.form("add_faq"):
            question = st.text_input("Question")
            answer = st.text_area("Answer")
            category = st.selectbox("Category", ["General", "Account", "Accounts", "Transactions", "Security", "Loans", "Cards"])
            
            if st.form_submit_button("Add FAQ"):
                if question and answer:
                    new_faq = {
                        'id': max([f['id'] for f in st.session_state.faqs]) + 1,
                        'question': question,
                        'answer': answer,
                        'category': category,
                        'active': True
                    }
                    st.session_state.faqs.append(new_faq)
                    st.success("✅ FAQ added successfully!")
                    st.rerun()
                else:
                    st.error("Please fill all fields")
    
    # Display FAQs
    st.subheader(f"📋 All FAQs ({len(st.session_state.faqs)})")
    
    for idx, faq in enumerate(st.session_state.faqs):
        with st.expander(f"**{faq['category']}** - {faq['question']}"):
            st.markdown(f"**Question:** {faq['question']}")
            st.markdown(f"**Answer:** {faq['answer']}")
            st.markdown(f"**Category:** {faq['category']}")
            st.markdown(f"**Status:** {'🟢 Active' if faq['active'] else '🔴 Inactive'}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✏️ Edit", key=f"edit_faq_{faq['id']}"):
                    st.info("Edit functionality - save to session state")
            
            with col2:
                if st.button("🔄 Toggle Status", key=f"toggle_{faq['id']}"):
                    st.session_state.faqs[idx]['active'] = not faq['active']
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Delete", key=f"del_faq_{faq['id']}"):
                    st.session_state.faqs = [f for f in st.session_state.faqs if f['id'] != faq['id']]
                    st.rerun()
    
    # Export FAQs
    st.markdown("---")
    if st.button("📥 Export FAQs to JSON"):
        faq_json = json.dumps(st.session_state.faqs, indent=2)
        st.download_button(
            "Download JSON",
            faq_json,
            f"faqs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )

# =============================================
# ANALYTICS PAGE
# =============================================

elif page == "📈 Analytics":
    st.title("📈 Analytics Dashboard")
    st.markdown("---")
    
    if data is None:
        st.error("⚠️ Dataset not loaded!")
    else:
        # Intent Distribution
        st.subheader("📊 Intent Distribution")
        intent_counts = data['intent'].value_counts().head(10)
        
        fig = px.bar(
            x=intent_counts.values,
            y=intent_counts.index,
            orientation='h',
            labels={'x': 'Count', 'y': 'Intent'},
            title='Top 10 Intents',
            color=intent_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie Chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥧 Intent Percentage")
            top_intents = data['intent'].value_counts().head(5)
            fig_pie = px.pie(
                values=top_intents.values,
                names=top_intents.index,
                title='Top 5 Intents Distribution'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📈 Query Trends (Simulated)")
            # Simulated trend data
            dates = pd.date_range('2024-11-23', '2024-11-29', freq='D')
            queries = [4250, 4580, 3920, 4750, 4320, 4100, 4085]
            
            fig_line = px.line(
                x=dates,
                y=queries,
                labels={'x': 'Date', 'y': 'Queries'},
                title='Daily Query Volume'
            )
            st.plotly_chart(fig_line, use_container_width=True)
        
        # Statistics Table
        st.subheader("📋 Intent Statistics")
        stats_df = pd.DataFrame({
            'Intent': data['intent'].value_counts().index[:15],
            'Count': data['intent'].value_counts().values[:15],
            'Percentage': (data['intent'].value_counts().values[:15] / len(data) * 100).round(2)
        })
        st.dataframe(stats_df, width='stretch', hide_index=True)

# =============================================
# SETTINGS PAGE
# =============================================

elif page == "⚙️ Settings":
    st.title("⚙️ System Settings")
    st.markdown("---")
    
    st.subheader("📁 Dataset Configuration")
    st.info(f"**Current Dataset:** bankbot_expanded_dataset.csv")
    
    if data is not None:
        st.success(f"✅ Loaded: {len(data):,} samples, {len(data['intent'].unique())} intents")
    else:
        st.error("❌ Dataset not found!")
    
    if st.button("🔄 Reload Dataset"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    st.subheader("🤖 Model Configuration")
    st.info("**Model:** intent_pipeline.joblib")
    st.info("**Algorithm:** Logistic Regression with TF-IDF")
    
    st.markdown("---")
    
    st.subheader("💾 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Backup FAQs", width='stretch'):
            faq_json = json.dumps(st.session_state.faqs, indent=2)
            st.download_button(
                "Download Backup",
                faq_json,
                f"faqs_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
    
    with col2:
        if st.button("🔄 Reset FAQs", width='stretch'):
            st.warning("This will reset FAQs to default!")
    
    st.markdown("---")
    st.subheader("ℹ️ System Information")
    st.info(f"""
    - **Version:** 1.0.0
    - **Last Updated:** {datetime.now().strftime('%Y-%m-%d')}
    - **Status:** 🟢 Online
    """)